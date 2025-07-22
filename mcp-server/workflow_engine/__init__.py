"""
Workflow Engine Package

Production-ready workflow execution engine with:
- Priority queue scheduler (αβγ scoring)
- Worker pools per agent
- Retry tracker with exponential backoff
- Deadlock detection and monitoring
- Redis-backed state management
"""

import asyncio
import logging
from typing import Dict, Any, Optional

from .scheduler import PriorityQueueScheduler
from .worker_pool import MultiAgentWorkerManager
from .retry_tracker import RetryTracker
from .deadlock_monitor import DeadlockMonitor
from .state import RedisStore

logger = logging.getLogger(__name__)

class WorkflowEngine:
    """
    Main workflow engine class that orchestrates all components.
    """
    
    def __init__(self, config: Dict[str, Any], workflow_manager=None):
        """
        Initialize workflow engine.
        
        Args:
            config: Engine configuration
            workflow_manager: Optional workflow manager for database access
        """
        self.config = config
        self.workflow_manager = workflow_manager
        
        # Core components
        self.scheduler = None
        self.retry_tracker = None
        self.worker_manager = None
        self.deadlock_monitor = None
        
        # State
        self.is_running = False
        
        logger.info("Workflow engine initialized")
    
    async def start(self):
        """Start all engine components."""
        if self.is_running:
            logger.warning("Workflow engine already running")
            return
        
        try:
            # Initialize scheduler
            scheduler_config = {
                **self.config,
                "redis_url": self.config.get("redis_url", "redis://localhost:6379")
            }
            self.scheduler = PriorityQueueScheduler(scheduler_config)
            
            # Initialize retry tracker
            retry_config = self.config.get("retry", {})
            retry_config["redis_url"] = scheduler_config["redis_url"]
            self.retry_tracker = RetryTracker(self.scheduler, retry_config)
            
            # Start retry polling
            await self.retry_tracker.start_polling()
            
            # Initialize worker manager
            self.worker_manager = MultiAgentWorkerManager(
                self.scheduler, 
                self.retry_tracker, 
                self.config
            )
            
            # Start worker pools
            await self.worker_manager.start_all()
            
            # Initialize deadlock monitor
            deadlock_config = self.config.get("deadlock", {})
            self.deadlock_monitor = DeadlockMonitor(
                deadlock_config, 
                self.workflow_manager, 
                self.scheduler
            )
            
            # Start deadlock monitoring
            await self.deadlock_monitor.start()
            
            self.is_running = True
            logger.info("Workflow engine started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start workflow engine: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop all engine components."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        try:
            # Stop components in reverse order
            if self.deadlock_monitor:
                await self.deadlock_monitor.stop()
            
            if self.worker_manager:
                await self.worker_manager.stop_all()
            
            if self.retry_tracker:
                await self.retry_tracker.stop_polling()
            
            logger.info("Workflow engine stopped")
            
        except Exception as e:
            logger.error(f"Error stopping workflow engine: {e}")
    
    def enqueue_task(self, task_meta: Dict[str, Any]) -> bool:
        """
        Enqueue task for execution.
        
        Args:
            task_meta: Task metadata
            
        Returns:
            True if enqueued successfully
        """
        if not self.scheduler:
            logger.error("Scheduler not initialized")
            return False
        
        return self.scheduler.enqueue(task_meta)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics."""
        stats = {
            "is_running": self.is_running,
            "scheduler": self.scheduler.get_stats() if self.scheduler else None,
            "retry_tracker": self.retry_tracker.get_stats() if self.retry_tracker else None,
            "workers": self.worker_manager.get_all_stats() if self.worker_manager else None,
            "deadlock_monitor": self.deadlock_monitor.get_stats() if self.deadlock_monitor else None
        }
        return stats
    
    def add_event_callback(self, callback):
        """Add event callback to worker manager."""
        if self.worker_manager:
            self.worker_manager.add_event_callback(callback)

# Global engine instance
_engine: Optional[WorkflowEngine] = None

def start_engine(config: Dict[str, Any], workflow_manager=None) -> WorkflowEngine:
    """
    Start the global workflow engine.
    
    Args:
        config: Engine configuration
        workflow_manager: Optional workflow manager instance
        
    Returns:
        WorkflowEngine instance
    """
    global _engine
    
    if _engine is not None:
        logger.warning("Engine already started, returning existing instance")
        return _engine
    
    _engine = WorkflowEngine(config, workflow_manager)
    
    # Start engine in event loop
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # If loop is already running, schedule start
        asyncio.create_task(_engine.start())
    else:
        # Start synchronously
        loop.run_until_complete(_engine.start())
    
    logger.info("Global workflow engine started")
    return _engine

async def start_engine_async(config: Dict[str, Any], workflow_manager=None) -> WorkflowEngine:
    """
    Async version of start_engine.
    
    Args:
        config: Engine configuration
        workflow_manager: Optional workflow manager instance
        
    Returns:
        WorkflowEngine instance
    """
    global _engine
    
    if _engine is not None:
        logger.warning("Engine already started, returning existing instance")
        return _engine
    
    _engine = WorkflowEngine(config, workflow_manager)
    await _engine.start()
    
    logger.info("Global workflow engine started (async)")
    return _engine

def get_engine() -> Optional[WorkflowEngine]:
    """Get the global workflow engine instance."""
    return _engine

async def stop_engine():
    """Stop the global workflow engine."""
    global _engine
    
    if _engine:
        await _engine.stop()
        _engine = None
        logger.info("Global workflow engine stopped")

def enqueue_task(task_meta: Dict[str, Any]) -> bool:
    """
    Enqueue task using global engine.
    
    Args:
        task_meta: Task metadata
        
    Returns:
        True if enqueued successfully
    """
    if _engine:
        return _engine.enqueue_task(task_meta)
    else:
        logger.error("Workflow engine not started")
        return False

def get_engine_stats() -> Dict[str, Any]:
    """Get stats from global engine."""
    if _engine:
        return _engine.get_stats()
    else:
        return {"error": "Engine not started"}

# Package exports
__all__ = [
    "WorkflowEngine",
    "start_engine",
    "start_engine_async", 
    "stop_engine",
    "get_engine",
    "enqueue_task",
    "get_engine_stats",
    "PriorityQueueScheduler",
    "MultiAgentWorkerManager",
    "RetryTracker",
    "DeadlockMonitor",
    "RedisStore"
]

# Version info
__version__ = "1.0.0" 