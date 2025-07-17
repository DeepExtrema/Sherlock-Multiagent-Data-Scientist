"""
SLA Monitor for the Master Orchestrator.

Provides monitoring and alerting for workflow and task SLA violations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class SLAViolationType(Enum):
    """Types of SLA violations."""
    TASK_TIMEOUT = "task_timeout"
    WORKFLOW_TIMEOUT = "workflow_timeout"
    QUEUE_STALE = "queue_stale"
    RESOURCE_EXHAUSTION = "resource_exhaustion"

@dataclass
class SLAViolation:
    """SLA violation record."""
    violation_id: str
    violation_type: SLAViolationType
    resource_id: str
    resource_type: str  # "task" or "workflow"
    detected_at: datetime
    age_seconds: float
    threshold_seconds: float
    severity: str  # "warning", "critical"
    context: Dict[str, Any]

class SLAMonitor:
    """Monitor and alert on SLA violations."""
    
    def __init__(self, 
                 config: Dict[str, Any],
                 alert_callback: Optional[Callable] = None):
        """
        Initialize SLA monitor.
        
        Args:
            config: SLA configuration with thresholds
            alert_callback: Function to call when violations occur
        """
        self.config = config
        self.alert_callback = alert_callback
        
        # Monitoring state
        self.is_running = False
        self.monitor_task = None
        
        # Violation tracking
        self.active_violations: Dict[str, SLAViolation] = {}
        self.violation_history: List[SLAViolation] = []
        
        # Callbacks for data access
        self.get_stale_tasks_callback = None
        self.get_stale_workflows_callback = None
        self.broadcast_event_callback = None
        
        # Statistics
        self.stats = {
            "total_violations": 0,
            "violations_by_type": {},
            "last_check": None,
            "monitoring_errors": 0
        }
    
    def set_data_callbacks(self,
                          get_stale_tasks: Callable,
                          get_stale_workflows: Callable,
                          broadcast_event: Callable):
        """
        Set callbacks for accessing external data.
        
        Args:
            get_stale_tasks: Function to get stale tasks
            get_stale_workflows: Function to get stale workflows  
            broadcast_event: Function to broadcast events
        """
        self.get_stale_tasks_callback = get_stale_tasks
        self.get_stale_workflows_callback = get_stale_workflows
        self.broadcast_event_callback = broadcast_event
    
    async def start(self):
        """Start SLA monitoring."""
        if self.is_running:
            logger.warning("SLA monitor is already running")
            return
        
        self.is_running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("SLA monitor started")
    
    async def stop(self):
        """Stop SLA monitoring."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("SLA monitor stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        check_interval = self.config.get("check_interval_seconds", 30)
        
        while self.is_running:
            try:
                await self._check_sla_violations()
                self.stats["last_check"] = datetime.now()
                
                await asyncio.sleep(check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in SLA monitor loop: {e}")
                self.stats["monitoring_errors"] += 1
                await asyncio.sleep(min(check_interval * 2, 300))  # Backoff on error
    
    async def _check_sla_violations(self):
        """Check for SLA violations."""
        current_time = datetime.now()
        
        # Check task SLAs
        if self.get_stale_tasks_callback:
            await self._check_task_slas(current_time)
        
        # Check workflow SLAs
        if self.get_stale_workflows_callback:
            await self._check_workflow_slas(current_time)
        
        # Clean up resolved violations
        await self._cleanup_resolved_violations()
    
    async def _check_task_slas(self, current_time: datetime):
        """Check task SLA violations."""
        try:
            task_sla_seconds = self.config.get("task_timeout_seconds", 600)
            if not self.get_stale_tasks_callback:
                return
            stale_tasks = await self.get_stale_tasks_callback(task_sla_seconds)
            
            for task in stale_tasks:
                task_id = task.get("task_id")
                if not task_id:
                    continue
                
                violation_id = f"task_{task_id}"
                
                # Skip if already tracking this violation
                if violation_id in self.active_violations:
                    continue
                
                age_seconds = (current_time - task["created_at"]).total_seconds()
                
                violation = SLAViolation(
                    violation_id=violation_id,
                    violation_type=SLAViolationType.TASK_TIMEOUT,
                    resource_id=task_id,
                    resource_type="task",
                    detected_at=current_time,
                    age_seconds=age_seconds,
                    threshold_seconds=task_sla_seconds,
                    severity=self._calculate_severity(age_seconds, task_sla_seconds),
                    context={
                        "task_status": task.get("status"),
                        "run_id": task.get("run_id"),
                        "agent": task.get("agent", "unknown")
                    }
                )
                
                await self._handle_violation(violation)
                
        except Exception as e:
            logger.error(f"Error checking task SLAs: {e}")
    
    async def _check_workflow_slas(self, current_time: datetime):
        """Check workflow SLA violations."""
        try:
            workflow_sla_seconds = self.config.get("workflow_timeout_seconds", 3600)
            if not self.get_stale_workflows_callback:
                return
            stale_workflows = await self.get_stale_workflows_callback(workflow_sla_seconds)
            
            for workflow in stale_workflows:
                workflow_id = workflow.get("run_id")
                if not workflow_id:
                    continue
                
                violation_id = f"workflow_{workflow_id}"
                
                # Skip if already tracking this violation
                if violation_id in self.active_violations:
                    continue
                
                age_seconds = (current_time - workflow["created_at"]).total_seconds()
                
                violation = SLAViolation(
                    violation_id=violation_id,
                    violation_type=SLAViolationType.WORKFLOW_TIMEOUT,
                    resource_id=workflow_id,
                    resource_type="workflow",
                    detected_at=current_time,
                    age_seconds=age_seconds,
                    threshold_seconds=workflow_sla_seconds,
                    severity=self._calculate_severity(age_seconds, workflow_sla_seconds),
                    context={
                        "workflow_status": workflow.get("status"),
                        "task_count": workflow.get("task_count", 0)
                    }
                )
                
                await self._handle_violation(violation)
                
        except Exception as e:
            logger.error(f"Error checking workflow SLAs: {e}")
    
    def _calculate_severity(self, age_seconds: float, threshold_seconds: float) -> str:
        """Calculate violation severity based on how much threshold is exceeded."""
        ratio = age_seconds / threshold_seconds
        
        if ratio >= 2.0:
            return "critical"
        elif ratio >= 1.5:
            return "high"
        elif ratio >= 1.2:
            return "medium"
        else:
            return "warning"
    
    async def _handle_violation(self, violation: SLAViolation):
        """Handle a new SLA violation."""
        # Add to active violations
        self.active_violations[violation.violation_id] = violation
        self.violation_history.append(violation)
        
        # Update statistics
        self.stats["total_violations"] += 1
        violation_type = violation.violation_type.value
        if violation_type not in self.stats["violations_by_type"]:
            self.stats["violations_by_type"][violation_type] = 0
        self.stats["violations_by_type"][violation_type] += 1
        
        # Log violation
        logger.warning(
            f"SLA violation detected: {violation.violation_type.value} "
            f"for {violation.resource_type} {violation.resource_id}, "
            f"age: {violation.age_seconds:.0f}s, "
            f"threshold: {violation.threshold_seconds:.0f}s, "
            f"severity: {violation.severity}"
        )
        
        # Broadcast event
        if self.broadcast_event_callback:
            try:
                await self.broadcast_event_callback({
                    "type": "sla_violation",
                    "violation_id": violation.violation_id,
                    "violation_type": violation.violation_type.value,
                    "resource_type": violation.resource_type,
                    "resource_id": violation.resource_id,
                    "severity": violation.severity,
                    "age_seconds": violation.age_seconds,
                    "threshold_seconds": violation.threshold_seconds,
                    "detected_at": violation.detected_at.isoformat(),
                    "context": violation.context
                })
            except Exception as e:
                logger.error(f"Failed to broadcast SLA violation event: {e}")
        
        # Call alert callback
        if self.alert_callback:
            try:
                await self.alert_callback(violation)
            except Exception as e:
                logger.error(f"Failed to call alert callback: {e}")
    
    async def _cleanup_resolved_violations(self):
        """Remove violations that are no longer active."""
        resolved_violations = []
        
        for violation_id, violation in self.active_violations.items():
            # Check if the violation is still active by querying current state
            # This is a simplified approach - in practice, you'd query the database
            # to see if the resource is still in a violating state
            
            # For now, we'll consider violations resolved after a certain time
            time_since_detection = datetime.now() - violation.detected_at
            max_tracking_time = timedelta(hours=1)  # Stop tracking after 1 hour
            
            if time_since_detection > max_tracking_time:
                resolved_violations.append(violation_id)
        
        # Remove resolved violations
        for violation_id in resolved_violations:
            del self.active_violations[violation_id]
            logger.info(f"Removed resolved SLA violation: {violation_id}")
    
    async def manually_resolve_violation(self, violation_id: str) -> bool:
        """
        Manually mark a violation as resolved.
        
        Args:
            violation_id: ID of violation to resolve
            
        Returns:
            True if violation was found and resolved
        """
        if violation_id in self.active_violations:
            violation = self.active_violations[violation_id]
            del self.active_violations[violation_id]
            
            logger.info(f"Manually resolved SLA violation: {violation_id}")
            
            # Broadcast resolution event
            if self.broadcast_event_callback:
                try:
                    await self.broadcast_event_callback({
                        "type": "sla_violation_resolved",
                        "violation_id": violation_id,
                        "resource_type": violation.resource_type,
                        "resource_id": violation.resource_id,
                        "resolved_at": datetime.now().isoformat(),
                        "manually_resolved": True
                    })
                except Exception as e:
                    logger.error(f"Failed to broadcast SLA resolution event: {e}")
            
            return True
        
        return False
    
    def get_active_violations(self) -> List[SLAViolation]:
        """Get list of currently active violations."""
        return list(self.active_violations.values())
    
    def get_violation_history(self, limit: int = 100) -> List[SLAViolation]:
        """Get violation history."""
        return self.violation_history[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get SLA monitoring statistics."""
        return {
            **self.stats,
            "active_violations_count": len(self.active_violations),
            "is_running": self.is_running,
            "config": self.config
        }
    
    def reset_statistics(self):
        """Reset monitoring statistics."""
        self.stats = {
            "total_violations": 0,
            "violations_by_type": {},
            "last_check": None,
            "monitoring_errors": 0
        }
        logger.info("SLA monitor statistics reset") 