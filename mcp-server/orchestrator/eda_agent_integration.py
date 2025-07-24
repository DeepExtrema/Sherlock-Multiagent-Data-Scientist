"""
EDA Agent Integration Layer

Connects the EDA Agent service to the Master Orchestrator workflow engine.
Handles task routing, result processing, and event emission.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel

from .telemetry import TelemetryManager
from .security import SecurityUtils
from .cache_client import CacheClient

logger = logging.getLogger(__name__)

class EDAAgentTask(BaseModel):
    """EDA Agent task model."""
    run_id: str
    task_id: str
    action: str
    args: Dict[str, Any]
    priority: int = 5
    deadline_ts: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3

class EDAAgentResult(BaseModel):
    """EDA Agent result model."""
    task_id: str
    run_id: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float
    timestamp: float

class EDAAgentIntegration:
    """Integration layer for EDA Agent service."""
    
    def __init__(self, 
                 agent_url: str = "http://localhost:8001",
                 timeout: float = 60.0,
                 max_workers: int = 3,
                 cache_client: Optional[CacheClient] = None,
                 telemetry_manager: Optional[TelemetryManager] = None):
        """
        Initialize EDA Agent integration.
        
        Args:
            agent_url: EDA Agent service URL
            timeout: Request timeout in seconds
            max_workers: Maximum concurrent workers
            cache_client: Cache client for result caching
            telemetry_manager: Telemetry manager for monitoring
        """
        self.agent_url = agent_url.rstrip('/')
        self.timeout = timeout
        self.max_workers = max_workers
        self.cache_client = cache_client
        self.telemetry_manager = telemetry_manager
        
        # HTTP client for agent communication
        self.http_client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_connections=max_workers)
        )
        
        # Worker pool
        self.workers: List[asyncio.Task] = []
        self.is_running = False
        
        # Task queue
        self.task_queue: asyncio.Queue = asyncio.Queue()
        
        # Event callbacks
        self.event_callbacks: List[callable] = []
        
        logger.info(f"EDA Agent integration initialized for {agent_url}")
    
    async def start(self):
        """Start the EDA Agent integration."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start worker pool
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker_loop(f"eda_worker_{i}"))
            self.workers.append(worker)
        
        logger.info(f"EDA Agent integration started with {self.max_workers} workers")
    
    async def stop(self):
        """Stop the EDA Agent integration."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        
        # Close HTTP client
        await self.http_client.aclose()
        
        logger.info("EDA Agent integration stopped")
    
    async def enqueue_task(self, task: EDAAgentTask) -> bool:
        """
        Enqueue a task for processing.
        
        Args:
            task: EDA Agent task to process
            
        Returns:
            True if task was enqueued successfully
        """
        try:
            # Validate task
            if not self._validate_task(task):
                logger.error(f"Invalid task: {task}")
                return False
            
            # Check cache first
            if self.cache_client:
                cache_key = self._get_cache_key(task)
                cached_result = await self.cache_client.get(cache_key)
                if cached_result:
                    logger.info(f"Cache hit for task {task.task_id}")
                    await self._emit_event("TASK_SUCCESS", task, cached_result)
                    return True
            
            # Enqueue task
            await self.task_queue.put(task)
            logger.info(f"Task {task.task_id} enqueued")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enqueue task {task.task_id}: {e}")
            return False
    
    async def _worker_loop(self, worker_id: str):
        """Worker loop for processing tasks."""
        logger.info(f"Worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get task from queue
                task = await asyncio.wait_for(
                    self.task_queue.get(), 
                    timeout=1.0
                )
                
                # Process task
                await self._process_task(task, worker_id)
                
                # Mark task as done
                self.task_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _process_task(self, task: EDAAgentTask, worker_id: str):
        """Process a single EDA Agent task."""
        start_time = time.time()
        
        try:
            # Emit task started event
            await self._emit_event("TASK_STARTED", task, {
                "worker_id": worker_id,
                "timestamp": start_time
            })
            
            # Execute task
            result = await self._execute_task(task)
            
            # Cache result if successful
            if result.success and self.cache_client:
                cache_key = self._get_cache_key(task)
                await self.cache_client.set(cache_key, result.result, ttl=3600)
            
            # Emit task completed event
            await self._emit_event(
                "TASK_SUCCESS" if result.success else "TASK_FAILED",
                task,
                result.dict()
            )
            
            # Record telemetry
            if self.telemetry_manager:
                self.telemetry_manager.record_operation(
                    operation=f"eda_{task.action}",
                    duration=result.execution_time,
                    success=result.success,
                    metadata={
                        "task_id": task.task_id,
                        "run_id": task.run_id,
                        "worker_id": worker_id
                    }
                )
            
        except Exception as e:
            logger.error(f"Failed to process task {task.task_id}: {e}")
            
            # Emit task failed event
            await self._emit_event("TASK_FAILED", task, {
                "error": str(e),
                "execution_time": time.time() - start_time
            })
    
    async def _execute_task(self, task: EDAAgentTask) -> EDAAgentResult:
        """Execute a task against the EDA Agent service."""
        start_time = time.time()
        
        try:
            # Prepare request
            url = f"{self.agent_url}/{task.action}"
            payload = self._sanitize_payload(task.args)
            
            # Make HTTP request
            response = await self.http_client.post(url, json=payload)
            response.raise_for_status()
            
            result_data = response.json()
            execution_time = time.time() - start_time
            
            return EDAAgentResult(
                task_id=task.task_id,
                run_id=task.run_id,
                success=True,
                result=result_data,
                execution_time=execution_time,
                timestamp=time.time()
            )
            
        except httpx.HTTPStatusError as e:
            execution_time = time.time() - start_time
            return EDAAgentResult(
                task_id=task.task_id,
                run_id=task.run_id,
                success=False,
                error=f"HTTP {e.response.status_code}: {e.response.text}",
                execution_time=execution_time,
                timestamp=time.time()
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return EDAAgentResult(
                task_id=task.task_id,
                run_id=task.run_id,
                success=False,
                error=str(e),
                execution_time=execution_time,
                timestamp=time.time()
            )
    
    def _validate_task(self, task: EDAAgentTask) -> bool:
        """Validate task parameters."""
        # Check required fields
        if not task.run_id or not task.task_id or not task.action:
            return False
        
        # Check action is supported
        supported_actions = [
            "load_data", "basic_info", "statistical_summary",
            "missing_data_analysis", "create_visualization",
            "infer_schema", "detect_outliers"
        ]
        
        if task.action not in supported_actions:
            logger.error(f"Unsupported action: {task.action}")
            return False
        
        # Validate action-specific parameters
        if not self._validate_action_params(task.action, task.args):
            return False
        
        return True
    
    def _validate_action_params(self, action: str, args: Dict[str, Any]) -> bool:
        """Validate action-specific parameters."""
        if action == "load_data":
            return "path" in args and "name" in args
        elif action == "basic_info":
            return "name" in args
        elif action == "statistical_summary":
            return "name" in args
        elif action == "missing_data_analysis":
            return "name" in args
        elif action == "create_visualization":
            return "name" in args and "chart_type" in args
        elif action == "infer_schema":
            return "name" in args
        elif action == "detect_outliers":
            return "name" in args and "method" in args
        
        return True
    
    def _sanitize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize payload for security."""
        if not payload:
            return {}
        
        # Use security utils if available
        if hasattr(SecurityUtils, 'sanitize'):
            return SecurityUtils.sanitize(payload)
        
        # Basic sanitization
        sanitized = {}
        for key, value in payload.items():
            if isinstance(value, str):
                # Basic path traversal protection
                if '..' in value or value.startswith('/'):
                    raise ValueError(f"Potentially unsafe value: {value}")
            sanitized[key] = value
        
        return sanitized
    
    def _get_cache_key(self, task: EDAAgentTask) -> str:
        """Generate cache key for task."""
        import hashlib
        key_data = f"eda:{task.action}:{json.dumps(task.args, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _emit_event(self, event_type: str, task: EDAAgentTask, data: Dict[str, Any]):
        """Emit event to registered callbacks."""
        event = {
            "type": event_type,
            "task_id": task.task_id,
            "run_id": task.run_id,
            "agent": "eda_agent",
            "action": task.action,
            "data": data,
            "timestamp": time.time()
        }
        
        for callback in self.event_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")
    
    def add_event_callback(self, callback: callable):
        """Add event callback."""
        self.event_callbacks.append(callback)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check EDA Agent service health."""
        try:
            response = await self.http_client.get(f"{self.agent_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def list_datasets(self) -> Dict[str, Any]:
        """List datasets in EDA Agent."""
        try:
            response = await self.http_client.get(f"{self.agent_url}/datasets")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to list datasets: {e}")
            return {"datasets": []}

# Factory function for creating EDA Agent integration
def create_eda_agent_integration(
    config: Dict[str, Any],
    cache_client: Optional[CacheClient] = None,
    telemetry_manager: Optional[TelemetryManager] = None
) -> EDAAgentIntegration:
    """
    Create EDA Agent integration from configuration.
    
    Args:
        config: Configuration dictionary
        cache_client: Cache client instance
        telemetry_manager: Telemetry manager instance
        
    Returns:
        Configured EDA Agent integration
    """
    agent_config = config.get("eda_agent", {})
    
    return EDAAgentIntegration(
        agent_url=agent_config.get("url", "http://localhost:8001"),
        timeout=agent_config.get("timeout", 60.0),
        max_workers=agent_config.get("max_workers", 3),
        cache_client=cache_client,
        telemetry_manager=telemetry_manager
    ) 