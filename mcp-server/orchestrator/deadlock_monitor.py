"""
Deadlock Monitor for Workflow Engine

Monitors MongoDB for stuck workflows and provides automatic cancellation
capabilities. Detects scenarios where all tasks are pending/queued but
none have been updated recently, indicating a potential deadlock.

Features:
- Periodic MongoDB scanning for stuck workflows
- Configurable staleness thresholds
- Optional Slack/PagerDuty alerting
- Automatic workflow cancellation
- Graceful error handling and logging
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from ..config import DeadlockConfig

logger = logging.getLogger(__name__)

class DeadlockMonitor:
    """
    Background monitor for detecting and handling workflow deadlocks.
    
    The monitor periodically scans MongoDB for workflows that appear stuck:
    - All tasks in PENDING/QUEUED state
    - No task updates within the staleness threshold
    - Workflow has been running longer than expected
    
    When a deadlock is detected, the monitor can:
    - Send alerts to configured webhook
    - Automatically cancel the workflow
    - Log detailed information for debugging
    """
    
    def __init__(self, 
                 db: AsyncIOMotorDatabase, 
                 config: DeadlockConfig):
        """
        Initialize deadlock monitor.
        
        Args:
            db: MongoDB database connection
            config: Deadlock configuration settings
        """
        self.db = db
        self.config = config
        self._task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info(f"DeadlockMonitor initialized with check_interval={config.check_interval_s}s, "
                   f"staleness_threshold={config.pending_stale_s}s")
    
    async def start(self) -> None:
        """Start the deadlock monitoring background task."""
        if self._running:
            logger.warning("DeadlockMonitor already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._monitoring_loop())
        logger.info("DeadlockMonitor started")
    
    async def stop(self) -> None:
        """Stop the deadlock monitoring background task."""
        if not self._running:
            return
        
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("DeadlockMonitor stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop that runs continuously."""
        logger.info("Starting deadlock monitoring loop")
        
        while self._running:
            try:
                await self._scan_for_deadlocks()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Deadlock scan failed: {exc}", exc_info=True)
            
            # Wait for next check interval
            try:
                await asyncio.sleep(self.config.check_interval_s)
            except asyncio.CancelledError:
                break
        
        logger.info("Deadlock monitoring loop terminated")
    
    async def _scan_for_deadlocks(self) -> None:
        """
        Scan MongoDB for potential deadlocks.
        
        A workflow is considered deadlocked if:
        1. Status is RUNNING
        2. All tasks are in PENDING or QUEUED state
        3. No task has been updated within the staleness threshold
        4. Workflow has been running longer than expected (optional)
        """
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.config.pending_stale_s)
        workflow_cutoff = datetime.utcnow() - timedelta(seconds=self.config.workflow_stale_s)
        
        # MongoDB aggregation pipeline to find stuck workflows
        pipeline = [
            # Match running workflows
            {"$match": {"status": "RUNNING"}},
            
            # Join with tasks
            {"$lookup": {
                "from": "tasks",
                "localField": "run_id",
                "foreignField": "run_id",
                "as": "tasks"
            }},
            
            # Add computed fields
            {"$addFields": {
                "all_tasks_stuck": {
                    "$allElementsTrue": {
                        "$map": {
                            "input": "$tasks",
                            "as": "task",
                            "in": {"$in": ["$$task.status", ["PENDING", "QUEUED"]]}
                        }
                    }
                },
                "no_recent_updates": {
                    "$lt": [{"$max": "$tasks.updated_at"}, cutoff_time]
                },
                "workflow_too_old": {
                    "$lt": ["$created_at", workflow_cutoff]
                },
                "task_count": {"$size": "$tasks"},
                "pending_count": {
                    "$size": {
                        "$filter": {
                            "input": "$tasks",
                            "cond": {"$in": ["$$this.status", ["PENDING", "QUEUED"]]}
                        }
                    }
                }
            }},
            
            # Match potential deadlocks
            {"$match": {
                "$and": [
                    {"all_tasks_stuck": True},
                    {"no_recent_updates": True},
                    {"task_count": {"$gt": 0}}  # Must have tasks
                ]
            }},
            
            # Project useful fields for handling
            {"$project": {
                "run_id": 1,
                "workflow_name": 1,
                "status": 1,
                "created_at": 1,
                "updated_at": 1,
                "task_count": 1,
                "pending_count": 1,
                "client_id": 1,
                "last_task_update": {"$max": "$tasks.updated_at"},
                "workflow_too_old": 1
            }}
        ]
        
        deadlocked_workflows = []
        
        try:
            async for workflow in self.db.runs.aggregate(pipeline):
                deadlocked_workflows.append(workflow)
                await self._handle_deadlock(workflow)
        
        except Exception as exc:
            logger.error(f"Failed to scan for deadlocks: {exc}", exc_info=True)
            return
        
        if deadlocked_workflows:
            logger.warning(f"Detected {len(deadlocked_workflows)} potentially deadlocked workflows")
        else:
            logger.debug("No deadlocks detected in current scan")
    
    async def _handle_deadlock(self, workflow: Dict[str, Any]) -> None:
        """
        Handle a detected deadlock.
        
        Args:
            workflow: Workflow document from MongoDB with deadlock indicators
        """
        run_id = workflow["run_id"]
        workflow_name = workflow.get("workflow_name", "unknown")
        task_count = workflow.get("task_count", 0)
        last_update = workflow.get("last_task_update")
        
        logger.warning(
            f"Deadlock detected for workflow {run_id} ('{workflow_name}') "
            f"with {task_count} stuck tasks. Last update: {last_update}"
        )
        
        # Send alert if webhook configured
        if self.config.alert_webhook and HTTPX_AVAILABLE:
            await self._send_alert(workflow)
        
        # Auto-cancel if configured
        if self.config.cancel_on_deadlock:
            await self._cancel_workflow(run_id, "auto-cancel: deadlock detected")
        
        # Update workflow status to DEADLOCK for tracking
        try:
            await self.db.runs.update_one(
                {"run_id": run_id},
                {
                    "$set": {
                        "status": "DEADLOCK",
                        "deadlock_detected_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                }
            )
        except Exception as exc:
            logger.error(f"Failed to update workflow {run_id} status to DEADLOCK: {exc}")
    
    async def _send_alert(self, workflow: Dict[str, Any]) -> None:
        """
        Send alert to configured webhook.
        
        Args:
            workflow: Workflow document with deadlock information
        """
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, cannot send webhook alerts")
            return
        
        run_id = workflow["run_id"]
        workflow_name = workflow.get("workflow_name", "unknown")
        task_count = workflow.get("task_count", 0)
        last_update = workflow.get("last_task_update")
        
        alert_payload = {
            "text": f"🔴 Deadlock Alert: Workflow {run_id}",
            "attachments": [
                {
                    "color": "danger",
                    "fields": [
                        {"title": "Workflow ID", "value": run_id, "short": True},
                        {"title": "Workflow Name", "value": workflow_name, "short": True},
                        {"title": "Stuck Tasks", "value": str(task_count), "short": True},
                        {"title": "Last Update", "value": str(last_update), "short": True},
                        {"title": "Auto-Cancel", "value": str(self.config.cancel_on_deadlock), "short": True}
                    ]
                }
            ]
        }
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    self.config.alert_webhook,
                    json=alert_payload
                )
                response.raise_for_status()
                logger.info(f"Deadlock alert sent for workflow {run_id}")
        
        except Exception as exc:
            logger.error(f"Failed to send deadlock alert for workflow {run_id}: {exc}")
    
    async def _cancel_workflow(self, run_id: str, reason: str) -> bool:
        """
        Cancel a deadlocked workflow.
        
        Args:
            run_id: Workflow run ID to cancel
            reason: Cancellation reason for logging
            
        Returns:
            True if cancellation was successful, False otherwise
        """
        try:
            # Import here to avoid circular imports
            from .workflow_manager import cancel_workflow_internal
            
            success = await cancel_workflow_internal(run_id, reason)
            if success:
                logger.info(f"Successfully cancelled deadlocked workflow {run_id}: {reason}")
            else:
                logger.warning(f"Failed to cancel workflow {run_id}: workflow not found or already finished")
            
            return success
        
        except ImportError:
            logger.error("Cannot import cancel_workflow_internal - workflow cancellation unavailable")
            return False
        except Exception as exc:
            logger.error(f"Failed to cancel workflow {run_id}: {exc}", exc_info=True)
            return False
    
    async def manual_scan(self) -> List[Dict[str, Any]]:
        """
        Perform a manual deadlock scan and return results without taking action.
        
        Returns:
            List of workflows that would be considered deadlocked
        """
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.config.pending_stale_s)
        
        pipeline = [
            {"$match": {"status": "RUNNING"}},
            {"$lookup": {
                "from": "tasks",
                "localField": "run_id", 
                "foreignField": "run_id",
                "as": "tasks"
            }},
            {"$addFields": {
                "all_tasks_stuck": {
                    "$allElementsTrue": {
                        "$map": {
                            "input": "$tasks",
                            "as": "task",
                            "in": {"$in": ["$$task.status", ["PENDING", "QUEUED"]]}
                        }
                    }
                },
                "no_recent_updates": {
                    "$lt": [{"$max": "$tasks.updated_at"}, cutoff_time]
                }
            }},
            {"$match": {
                "$and": [
                    {"all_tasks_stuck": True},
                    {"no_recent_updates": True},
                    {"tasks": {"$ne": []}}
                ]
            }}
        ]
        
        results = []
        async for workflow in self.db.runs.aggregate(pipeline):
            results.append(workflow)
        
        logger.info(f"Manual scan found {len(results)} potentially deadlocked workflows")
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get deadlock monitor statistics."""
        return {
            "running": self._running,
            "check_interval_s": self.config.check_interval_s,
            "pending_stale_s": self.config.pending_stale_s,
            "workflow_stale_s": self.config.workflow_stale_s,
            "cancel_on_deadlock": self.config.cancel_on_deadlock,
            "alert_webhook_configured": bool(self.config.alert_webhook),
            "httpx_available": HTTPX_AVAILABLE
        } 