"""
Refinery Agent Integration Layer

Connects the Refinery Agent service to the Master Orchestrator workflow engine.
Handles task routing, result processing, and event emission for data quality validation
and feature engineering tasks.
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

class RefineryTask(BaseModel):
    """Refinery Agent task model."""
    run_id: str
    task_id: str
    action: str
    args: Dict[str, Any]
    priority: int = 5
    deadline_ts: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3

class RefineryResult(BaseModel):
    """Refinery Agent result model."""
    task_id: str
    run_id: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float
    timestamp: float

class RefineryIntegration:
    """Integration layer for Refinery Agent service."""
    
    def __init__(self, 
                 agent_url: str = "http://localhost:8005",
                 timeout: float = 60.0,
                 max_workers: int = 3,
                 cache_client: Optional[CacheClient] = None,
                 telemetry_manager: Optional[TelemetryManager] = None):
        """
        Initialize Refinery Agent integration.
        
        Args:
            agent_url: Refinery Agent service URL
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
        
        logger.info(f"Refinery Agent integration initialized for {agent_url}")
    
    async def start(self):
        """Start the Refinery Agent integration."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start worker pool
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker_loop(f"refinery_worker_{i}"))
            self.workers.append(worker)
        
        logger.info(f"Refinery Agent integration started with {self.max_workers} workers")
    
    async def stop(self):
        """Stop the Refinery Agent integration."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        # Close HTTP client
        await self.http_client.aclose()
        
        logger.info("Refinery Agent integration stopped")
    
    async def enqueue_task(self, task: RefineryTask) -> bool:
        """
        Enqueue a task for processing.
        
        Args:
            task: Task to enqueue
            
        Returns:
            True if task was enqueued successfully, False otherwise
        """
        try:
            if not self._validate_task(task):
                logger.error(f"Invalid task: {task.task_id}")
                return False
            
            # Check cache first
            if self.cache_client:
                cache_key = self._get_cache_key(task)
                cached_result = await self.cache_client.get(cache_key)
                if cached_result:
                    logger.info(f"Task {task.task_id} result found in cache")
                    await self._emit_event("task_completed", task, cached_result)
                    return True
            
            # Add to queue
            await self.task_queue.put(task)
            await self._emit_event("task_enqueued", task, {"queue_size": self.task_queue.qsize()})
            
            logger.info(f"Task {task.task_id} enqueued successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enqueue task {task.task_id}: {e}")
            return False
    
    async def _worker_loop(self, worker_id: str):
        """Worker loop for processing tasks."""
        logger.info(f"Refinery worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get task from queue
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
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
                continue
        
        logger.info(f"Refinery worker {worker_id} stopped")
    
    async def _process_task(self, task: RefineryTask, worker_id: str):
        """Process a single task."""
        logger.info(f"Worker {worker_id} processing task {task.task_id}")
        
        try:
            # Execute task with telemetry tracing
            result = await self._execute_task_with_telemetry(task)
            
            # Cache result if successful
            if result.success and self.cache_client:
                cache_key = self._get_cache_key(task)
                await self.cache_client.set(cache_key, result.dict(), ttl=3600)
            
            # Emit completion event
            await self._emit_event("task_completed", task, result.dict())
            
            logger.info(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            
            # Handle retries
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                await self.task_queue.put(task)
                await self._emit_event("task_retry", task, {"retry_count": task.retry_count})
                logger.info(f"Task {task.task_id} queued for retry {task.retry_count}")
            else:
                await self._emit_event("task_failed", task, {"error": str(e)})
    
    async def _execute_task_with_telemetry(self, task: RefineryTask) -> RefineryResult:
        """Execute a task with telemetry tracing."""
        if self.telemetry_manager:
            # Use telemetry tracing as suggested
            try:
                # Try to use trace_operation method if available
                if hasattr(self.telemetry_manager, 'trace_operation'):
                    async with self.telemetry_manager.trace_operation(
                        f"refinery.{task.action}", 
                        run_id=task.run_id, 
                        task_id=task.task_id
                    ):
                        return await self._execute_task(task)
                else:
                    # Fallback to basic execution
                    return await self._execute_task(task)
            except Exception as e:
                logger.warning(f"Telemetry tracing failed: {e}")
                return await self._execute_task(task)
        else:
            return await self._execute_task(task)
    
    async def _execute_task(self, task: RefineryTask) -> RefineryResult:
        """
        Execute a task by calling the Refinery Agent service.
        
        Args:
            task: Task to execute
            
        Returns:
            Task result
        """
        start_time = time.time()
        
        try:
            # Prepare request payload
            payload = {
                "task_id": task.task_id,
                "action": task.action,
                "params": task.args
            }
            
            # Sanitize payload
            payload = self._sanitize_payload(payload)
            
            # Make HTTP request to /execute endpoint
            url = f"{self.agent_url}/execute"
            response = await self.http_client.post(url, json=payload)
            
            # Handle response
            if response.status_code == 200:
                result_data = response.json()
                execution_time = time.time() - start_time
                
                return RefineryResult(
                    task_id=task.task_id,
                    run_id=task.run_id,
                    success=result_data.get("success", True),
                    result=result_data.get("result"),
                    error=result_data.get("error"),
                    execution_time=execution_time,
                    timestamp=time.time()
                )
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                execution_time = time.time() - start_time
                
                return RefineryResult(
                    task_id=task.task_id,
                    run_id=task.run_id,
                    success=False,
                    error=error_msg,
                    execution_time=execution_time,
                    timestamp=time.time()
                )
                
        except httpx.TimeoutException:
            execution_time = time.time() - start_time
            return RefineryResult(
                task_id=task.task_id,
                run_id=task.run_id,
                success=False,
                error="Request timeout",
                execution_time=execution_time,
                timestamp=time.time()
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return RefineryResult(
                task_id=task.task_id,
                run_id=task.run_id,
                success=False,
                error=str(e),
                execution_time=execution_time,
                timestamp=time.time()
            )
    
    def _validate_task(self, task: RefineryTask) -> bool:
        """
        Validate task parameters.
        
        Args:
            task: Task to validate
            
        Returns:
            True if task is valid, False otherwise
        """
        if not task.task_id or not task.action:
            return False
        
        # Validate action
        valid_actions = [
            # Data Quality actions
            "check_schema_consistency",
            "check_missing_values",
            "check_distributions",
            "check_duplicates",
            "check_leakage",
            "check_drift",
            # Feature Engineering actions
            "assign_feature_roles",
            "impute_missing_values",
            "scale_numeric_features",
            "encode_categorical_features",
            "generate_datetime_features",
            "vectorise_text_features",
            "generate_interactions",
            "select_features",
            "save_fe_pipeline"
        ]
        
        if task.action not in valid_actions:
            logger.error(f"Invalid action: {task.action}")
            return False
        
        # Validate action-specific parameters
        return self._validate_action_params(task.action, task.args)
    
    def _validate_action_params(self, action: str, args: Dict[str, Any]) -> bool:
        """
        Validate action-specific parameters.
        
        Args:
            action: Action name
            args: Action arguments
            
        Returns:
            True if parameters are valid, False otherwise
        """
        try:
            # Data Quality validations
            if action in ["check_schema_consistency", "check_missing_values", "check_distributions", 
                         "check_duplicates", "check_leakage"]:
                if "data_path" not in args:
                    logger.error(f"Missing data_path for action {action}")
                    return False
            
            elif action == "check_drift":
                if "reference_path" not in args or "current_path" not in args:
                    logger.error(f"Missing reference_path or current_path for drift check")
                    return False
            
            # Feature Engineering validations
            elif action in ["assign_feature_roles", "impute_missing_values"]:
                if "input_path" not in args or "run_id" not in args:
                    logger.error(f"Missing input_path or run_id for action {action}")
                    return False
            
            elif action in ["scale_numeric_features", "encode_categorical_features", 
                           "generate_datetime_features", "vectorise_text_features",
                           "generate_interactions", "select_features"]:
                if "run_id" not in args:
                    logger.error(f"Missing run_id for action {action}")
                    return False
            
            elif action == "save_fe_pipeline":
                required_fields = ["input_path", "export_pipeline_path", "export_data_path", "run_id"]
                for field in required_fields:
                    if field not in args:
                        logger.error(f"Missing {field} for save_fe_pipeline")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Parameter validation error for {action}: {e}")
            return False
    
    def _sanitize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize payload for security.
        
        Args:
            payload: Payload to sanitize
            
        Returns:
            Sanitized payload
        """
        try:
            if self.telemetry_manager and hasattr(self.telemetry_manager, 'security_utils'):
                return self.telemetry_manager.security_utils.sanitize_input(payload)
            elif hasattr(SecurityUtils, 'sanitize_input'):
                return SecurityUtils.sanitize_input(payload)
            else:
                # Basic sanitization - remove null bytes and limit string lengths
                def clean_value(value):
                    if isinstance(value, str):
                        return value.replace('\x00', '').replace('\n', ' ')[:1000]
                    elif isinstance(value, dict):
                        return {k: clean_value(v) for k, v in value.items()}
                    elif isinstance(value, list):
                        return [clean_value(v) for v in value]
                    else:
                        return value
                
                return clean_value(payload)
        except Exception as e:
            logger.warning(f"Payload sanitization failed: {e}")
            return payload
    
    def _get_cache_key(self, task: RefineryTask) -> str:
        """Generate cache key for task."""
        return f"refinery_cache:{task.run_id}:{task.task_id}:{task.action}"
    
    async def _emit_event(self, event_type: str, task: RefineryTask, data: Dict[str, Any]):
        """
        Emit event to registered callbacks.
        
        Args:
            event_type: Type of event
            task: Related task
            data: Event data
        """
        event = {
            "type": event_type,
            "task_id": task.task_id,
            "run_id": task.run_id,
            "action": task.action,
            "timestamp": time.time(),
            "data": data
        }
        
        # Call registered callbacks
        for callback in self.event_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")
        
        # Send to telemetry if available
        if self.telemetry_manager:
            try:
                await self.telemetry_manager.record_event("refinery_agent", event)
            except Exception as e:
                logger.error(f"Telemetry recording error: {e}")
    
    def add_event_callback(self, callback: callable):
        """Add event callback function."""
        self.event_callbacks.append(callback)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of the Refinery Agent service.
        
        Returns:
            Health status information
        """
        try:
            response = await self.http_client.get(f"{self.agent_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                return {
                    "status": "healthy",
                    "service": "refinery_agent",
                    "agent_status": health_data.get("status"),
                    "agent_version": health_data.get("version"),
                    "response": health_data
                }
            else:
                return {
                    "status": "unhealthy",
                    "service": "refinery_agent",
                    "error": f"HTTP {response.status_code}"
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "service": "refinery_agent",
                "error": str(e)
            }
    
    async def get_telemetry(self) -> Dict[str, Any]:
        """
        Get telemetry data from the Refinery Agent service.
        
        Returns:
            Telemetry data
        """
        try:
            response = await self.http_client.get(f"{self.agent_url}/telemetry")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

class RedisPipelineCache:
    """Redis-backed pipeline cache for production environments."""
    
    def __init__(self, redis_client, ttl: int = 3600):
        """
        Initialize Redis pipeline cache.
        
        Args:
            redis_client: Redis client instance
            ttl: Time to live for cache entries in seconds
        """
        self.redis_client = redis_client
        self.ttl = ttl
    
    async def get_pipeline_context(self, run_id: str, session_id: str = "default"):
        """Get pipeline context from Redis."""
        key = f"refinery:pipeline:{run_id}:{session_id}"
        try:
            data = await self.redis_client.get(key)
            if data:
                return json.loads(data)
            else:
                # Return default context
                default_context = {
                    "steps": [],
                    "feature_names": [],
                    "metadata": {},
                    "created_at": time.time()
                }
                await self.set_pipeline_context(run_id, session_id, default_context)
                return default_context
        except Exception as e:
            logger.error(f"Failed to get pipeline context: {e}")
            return {
                "steps": [],
                "feature_names": [],
                "metadata": {},
                "created_at": time.time()
            }
    
    async def set_pipeline_context(self, run_id: str, session_id: str, context: Dict[str, Any]):
        """Set pipeline context in Redis."""
        key = f"refinery:pipeline:{run_id}:{session_id}"
        try:
            context["updated_at"] = time.time()
            await self.redis_client.setex(key, self.ttl, json.dumps(context))
        except Exception as e:
            logger.error(f"Failed to set pipeline context: {e}")
    
    async def delete_pipeline_context(self, run_id: str, session_id: str = "default"):
        """Delete pipeline context from Redis."""
        key = f"refinery:pipeline:{run_id}:{session_id}"
        try:
            await self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Failed to delete pipeline context: {e}")

def create_refinery_agent_integration(
    config: Dict[str, Any],
    cache_client: Optional[CacheClient] = None,
    telemetry_manager: Optional[TelemetryManager] = None
) -> RefineryIntegration:
    """
    Create Refinery Agent integration from configuration.
    
    Args:
        config: Configuration dictionary
        cache_client: Cache client instance
        telemetry_manager: Telemetry manager instance
        
    Returns:
        Refinery Agent integration instance
    """
    refinery_config = config.get("workflow_engine", {}).get("refinery_agent", {})
    
    return RefineryIntegration(
        agent_url=refinery_config.get("url", "http://localhost:8005"),
        timeout=refinery_config.get("timeout", 60.0),
        max_workers=refinery_config.get("max_workers", 3),
        cache_client=cache_client,
        telemetry_manager=telemetry_manager
    )