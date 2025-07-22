"""
Deadlock Monitor for Workflow Engine

Detects and resolves deadlocks in workflow execution:
- Dependency cycles in task graphs
- Stuck workflows with no progress
- Resource exhaustion scenarios
- Stale tasks in PENDING/QUEUED states
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Set, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class DeadlockType(Enum):
    """Types of deadlocks that can be detected."""
    DEPENDENCY_CYCLE = "dependency_cycle"
    STALE_WORKFLOW = "stale_workflow"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    ORPHANED_TASKS = "orphaned_tasks"

@dataclass
class DeadlockAlert:
    """Alert information for detected deadlock."""
    type: DeadlockType
    run_id: str
    description: str
    affected_tasks: List[str]
    detected_at: datetime
    severity: str = "medium"  # low, medium, high, critical
    suggested_actions: List[str] = None

class DeadlockMonitor:
    """
    Monitors workflows for deadlock conditions and takes corrective action.
    
    Runs as a background process that periodically checks for:
    - Dependency cycles in task graphs
    - Workflows with no progress for extended periods
    - Tasks stuck in PENDING/QUEUED states beyond thresholds
    """
    
    def __init__(self, config: Dict[str, Any], workflow_manager=None, scheduler=None):
        """
        Initialize deadlock monitor.
        
        Args:
            config: Configuration dict
            workflow_manager: WorkflowManager instance (for database access)
            scheduler: Scheduler instance (for task queue access)
        """
        self.config = config
        self.workflow_manager = workflow_manager
        self.scheduler = scheduler
        
        # Configuration
        self.check_interval_s = config.get("check_interval_s", 60)
        self.pending_stale_s = config.get("pending_stale_s", 900)  # 15 minutes
        self.workflow_stale_s = config.get("workflow_stale_s", 3600)  # 1 hour
        self.max_dependency_depth = config.get("max_dependency_depth", 50)
        
        # Monitoring state
        self.is_running = False
        self.monitor_task = None
        self.alert_callbacks: List[callable] = []
        
        # Statistics
        self.stats = {
            "checks_performed": 0,
            "deadlocks_detected": 0,
            "cycles_found": 0,
            "stale_workflows": 0,
            "last_check": None,
            "alerts_sent": 0
        }
        
        logger.info(f"Deadlock monitor initialized: check_interval={self.check_interval_s}s, "
                   f"pending_threshold={self.pending_stale_s}s")
    
    def add_alert_callback(self, callback: callable):
        """Add callback for deadlock alerts."""
        self.alert_callbacks.append(callback)
    
    async def start(self):
        """Start deadlock monitoring."""
        if self.is_running:
            logger.warning("Deadlock monitor already running")
            return
        
        self.is_running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Deadlock monitor started")
    
    async def stop(self):
        """Stop deadlock monitoring."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Deadlock monitor stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        logger.info("Deadlock monitoring started")
        
        while self.is_running:
            try:
                await self._perform_checks()
                self.stats["checks_performed"] += 1
                self.stats["last_check"] = datetime.utcnow()
                
                await asyncio.sleep(self.check_interval_s)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in deadlock monitor loop: {e}")
                await asyncio.sleep(min(self.check_interval_s * 2, 300))  # Backoff on error
    
    async def _perform_checks(self):
        """Perform all deadlock detection checks."""
        try:
            # Check for dependency cycles
            await self._check_dependency_cycles()
            
            # Check for stale workflows
            await self._check_stale_workflows()
            
            # Check for stale tasks
            await self._check_stale_tasks()
            
            # Check for orphaned tasks
            await self._check_orphaned_tasks()
            
        except Exception as e:
            logger.error(f"Error performing deadlock checks: {e}")
    
    async def _check_dependency_cycles(self):
        """Check for circular dependencies in active workflows."""
        if not self.workflow_manager or not self.workflow_manager.db:
            return
        
        try:
            # Get all active runs
            active_runs = await self.workflow_manager.db.runs.find({
                "status": {"$in": ["RUNNING", "PENDING"]}
            }).to_list(None)
            
            for run in active_runs:
                run_id = run["run_id"]
                
                # Get all tasks for this run
                tasks = await self.workflow_manager.db.tasks.find({
                    "run_id": run_id,
                    "status": {"$in": ["PENDING", "QUEUED", "RUNNING"]}
                }).to_list(None)
                
                if not tasks:
                    continue
                
                # Build dependency graph
                graph = self._build_dependency_graph(tasks)
                
                # Check for cycles
                cycles = self._detect_cycles(graph)
                
                if cycles:
                    self.stats["cycles_found"] += 1
                    self.stats["deadlocks_detected"] += 1
                    
                    affected_tasks = [task_id for cycle in cycles for task_id in cycle]
                    
                    alert = DeadlockAlert(
                        type=DeadlockType.DEPENDENCY_CYCLE,
                        run_id=run_id,
                        description=f"Circular dependency detected involving {len(affected_tasks)} tasks",
                        affected_tasks=affected_tasks,
                        detected_at=datetime.utcnow(),
                        severity="high",
                        suggested_actions=[
                            "Review task dependencies in workflow definition",
                            "Consider breaking dependency cycle manually",
                            "Cancel and restart workflow with corrected dependencies"
                        ]
                    )
                    
                    await self._send_alert(alert)
                    
        except Exception as e:
            logger.error(f"Error checking dependency cycles: {e}")
    
    def _build_dependency_graph(self, tasks: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
        """Build dependency graph from task list."""
        graph = {}
        
        for task in tasks:
            task_id = task["task_id"]
            depends_on = task.get("depends_on", [])
            graph[task_id] = set(depends_on)
        
        return graph
    
    def _detect_cycles(self, graph: Dict[str, Set[str]]) -> List[List[str]]:
        """Detect cycles in dependency graph using DFS."""
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node, path):
            if node in rec_stack:
                # Found cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, set()):
                if neighbor in graph:  # Only consider nodes that exist
                    dfs(neighbor, path + [node])
            
            rec_stack.remove(node)
        
        for node in graph:
            if node not in visited:
                dfs(node, [])
        
        return cycles
    
    async def _check_stale_workflows(self):
        """Check for workflows that haven't made progress."""
        if not self.workflow_manager or not self.workflow_manager.db:
            return
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(seconds=self.workflow_stale_s)
            
            stale_workflows = await self.workflow_manager.db.runs.find({
                "status": "RUNNING",
                "updated_at": {"$lt": cutoff_time}
            }).to_list(None)
            
            for workflow in stale_workflows:
                run_id = workflow["run_id"]
                stale_duration = datetime.utcnow() - workflow["updated_at"]
                
                self.stats["stale_workflows"] += 1
                self.stats["deadlocks_detected"] += 1
                
                alert = DeadlockAlert(
                    type=DeadlockType.STALE_WORKFLOW,
                    run_id=run_id,
                    description=f"Workflow has been inactive for {stale_duration.total_seconds():.0f} seconds",
                    affected_tasks=[],
                    detected_at=datetime.utcnow(),
                    severity="medium",
                    suggested_actions=[
                        "Check agent health and connectivity",
                        "Review task queue for backlogs",
                        "Consider cancelling and restarting workflow"
                    ]
                )
                
                await self._send_alert(alert)
                
        except Exception as e:
            logger.error(f"Error checking stale workflows: {e}")
    
    async def _check_stale_tasks(self):
        """Check for tasks stuck in PENDING/QUEUED states."""
        if not self.workflow_manager or not self.workflow_manager.db:
            return
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(seconds=self.pending_stale_s)
            
            stale_tasks = await self.workflow_manager.db.tasks.find({
                "status": {"$in": ["PENDING", "QUEUED"]},
                "updated_at": {"$lt": cutoff_time}
            }).to_list(None)
            
            if stale_tasks:
                # Group by run_id
                runs_with_stale_tasks = {}
                for task in stale_tasks:
                    run_id = task["run_id"]
                    if run_id not in runs_with_stale_tasks:
                        runs_with_stale_tasks[run_id] = []
                    runs_with_stale_tasks[run_id].append(task["task_id"])
                
                for run_id, task_ids in runs_with_stale_tasks.items():
                    alert = DeadlockAlert(
                        type=DeadlockType.RESOURCE_EXHAUSTION,
                        run_id=run_id,
                        description=f"{len(task_ids)} tasks have been pending/queued for over {self.pending_stale_s}s",
                        affected_tasks=task_ids,
                        detected_at=datetime.utcnow(),
                        severity="medium",
                        suggested_actions=[
                            "Check worker pool capacity",
                            "Verify agent health",
                            "Review scheduler queue depth",
                            "Consider scaling up workers"
                        ]
                    )
                    
                    await self._send_alert(alert)
                    
        except Exception as e:
            logger.error(f"Error checking stale tasks: {e}")
    
    async def _check_orphaned_tasks(self):
        """Check for tasks that have no valid dependencies."""
        if not self.workflow_manager or not self.workflow_manager.db:
            return
        
        try:
            # Get all active runs
            active_runs = await self.workflow_manager.db.runs.find({
                "status": {"$in": ["RUNNING", "PENDING"]}
            }).to_list(None)
            
            for run in active_runs:
                run_id = run["run_id"]
                
                # Get all tasks for this run
                tasks = await self.workflow_manager.db.tasks.find({
                    "run_id": run_id
                }).to_list(None)
                
                if not tasks:
                    continue
                
                # Check for orphaned tasks
                task_ids = {task["task_id"] for task in tasks}
                orphaned = []
                
                for task in tasks:
                    if task["status"] in ["PENDING", "QUEUED"]:
                        depends_on = task.get("depends_on", [])
                        
                        # Check if any dependency doesn't exist
                        missing_deps = [dep for dep in depends_on if dep not in task_ids]
                        
                        if missing_deps:
                            orphaned.append(task["task_id"])
                
                if orphaned:
                    alert = DeadlockAlert(
                        type=DeadlockType.ORPHANED_TASKS,
                        run_id=run_id,
                        description=f"{len(orphaned)} tasks have missing dependencies",
                        affected_tasks=orphaned,
                        detected_at=datetime.utcnow(),
                        severity="high",
                        suggested_actions=[
                            "Review workflow definition for missing tasks",
                            "Check if dependent tasks were incorrectly removed",
                            "Consider manual intervention to fix dependencies"
                        ]
                    )
                    
                    await self._send_alert(alert)
                    
        except Exception as e:
            logger.error(f"Error checking orphaned tasks: {e}")
    
    async def _send_alert(self, alert: DeadlockAlert):
        """Send deadlock alert to registered callbacks."""
        self.stats["alerts_sent"] += 1
        
        logger.warning(f"Deadlock detected - {alert.type.value}: {alert.description} "
                      f"(Run: {alert.run_id}, Tasks: {len(alert.affected_tasks)})")
        
        # Call all registered callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Error in deadlock alert callback: {e}")
    
    async def manual_check(self, run_id: Optional[str] = None) -> List[DeadlockAlert]:
        """
        Perform manual deadlock check for specific run or all runs.
        
        Args:
            run_id: Optional run ID to check, None for all runs
            
        Returns:
            List of detected deadlock alerts
        """
        alerts = []
        
        # Temporarily capture alerts
        captured_alerts = []
        
        def capture_alert(alert):
            captured_alerts.append(alert)
        
        self.add_alert_callback(capture_alert)
        
        try:
            if run_id:
                # Check specific run (implementation would filter by run_id)
                await self._perform_checks()
            else:
                await self._perform_checks()
            
            alerts = captured_alerts.copy()
        finally:
            # Remove capture callback
            if capture_alert in self.alert_callbacks:
                self.alert_callbacks.remove(capture_alert)
        
        return alerts
    
    def get_stats(self) -> Dict[str, Any]:
        """Get deadlock monitor statistics."""
        return {
            **self.stats,
            "config": {
                "check_interval_s": self.check_interval_s,
                "pending_stale_s": self.pending_stale_s,
                "workflow_stale_s": self.workflow_stale_s,
                "max_dependency_depth": self.max_dependency_depth
            },
            "is_running": self.is_running,
            "next_check_in_s": self.check_interval_s if self.is_running else None
        } 