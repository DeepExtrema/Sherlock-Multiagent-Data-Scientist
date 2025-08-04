#!/usr/bin/env python3
"""
Refinery Agent Integration Layer
Connects the refinery agent to the Master Orchestrator.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

import httpx
from pydantic import BaseModel, Field
import redis.asyncio as redis

# Configure logging
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class RefineryTask(BaseModel):
    """Task model for refinery agent operations."""
    run_id: str
    task_id: str
    action: str  # Must be in AGENT_MATRIX['refinery_agent']
    params: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=5, ge=1, le=10)
    timeout: int = Field(default=300)  # 5 minutes
    retries: int = Field(default=3)
    created_at: float = Field(default_factory=time.time)

class RefineryResult(BaseModel):
    """Result model for refinery agent operations."""
    task_id: str
    run_id: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float
    timestamp: float
    worker_id: Optional[str] = None

class RedisPipelineCache:
    """Redis-backed pipeline state cache."""
    
    def __init__(self, redis_url: str, namespace: str = "refinery_pipeline"):
        self.redis_url = redis_url
        self.namespace = namespace
        self.redis_client: Optional[redis.Redis] = None
        self.default_ttl = 3600  # 1 hour
    
    async def connect(self):
        """Connect to Redis."""
        if not self.redis_client:
            self.redis_client = redis.from_url(
                self.redis_url,
                retry_on_timeout=True,
                socket_timeout=3,
                socket_connect_timeout=3,
                socket_keepalive=True
            )
            await self.redis_client.ping()
            logger.info(f"Connected to Redis pipeline cache: {self.redis_url}")
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
    
    def _get_key(self, run_id: str, session_id: str = "default") -> str:
        """Generate Redis key for pipeline."""
        return f"{self.namespace}:{run_id}:{session_id}"
    
    async def get_pipeline(self, run_id: str, session_id: str = "default") -> Optional[Dict[str, Any]]:
        """Get pipeline state from Redis."""
        await self.connect()
        key = self._get_key(run_id, session_id)
        data = await self.redis_client.get(key)
        if data:
            return json.loads(data)
        return None
    
    async def set_pipeline(self, run_id: str, session_id: str, pipeline_data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set pipeline state in Redis."""
        await self.connect()
        key = self._get_key(run_id, session_id)
        ttl = ttl or self.default_ttl
        return await self.redis_client.setex(key, ttl, json.dumps(pipeline_data))
    
    async def delete_pipeline(self, run_id: str, session_id: str = "default") -> bool:
        """Delete pipeline state from Redis."""
        await self.connect()
        key = self._get_key(run_id, session_id)
        return bool(await self.redis_client.delete(key))
    
    async def list_pipelines(self) -> List[str]:
        """List all pipeline keys."""
        await self.connect()
        pattern = f"{self.namespace}:*"
        keys = await self.redis_client.keys(pattern)
        return [key.decode() for key in keys]

class RefineryIntegration:
    """Integration layer for refinery agent."""
    
    def __init__(
        self,
        agent_url: str = "http://localhost:8005",
        max_workers: int = 3,
        timeout: int = 300,
        cache_enabled: bool = True,
        cache_ttl: int = 3600,
        telemetry_manager: Optional[Any] = None,
        redis_url: str = "redis://localhost:6379"
    ):
        self.agent_url = agent_url.rstrip('/')
        self.max_workers = max_workers
        self.timeout = timeout
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self.telemetry_manager = telemetry_manager
        
        # Worker pool
        self.workers: List[str] = []
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.result_cache: Dict[str, RefineryResult] = {}
        self.active_tasks: Dict[str, RefineryTask] = {}
        
        # Redis pipeline cache
        self.pipeline_cache = RedisPipelineCache(redis_url)
        
        # HTTP client
        self.http_client: Optional[httpx.AsyncClient] = None
        
        # Worker tasks
        self.worker_tasks: List[asyncio.Task] = []
        self.running = False
    
    async def start(self):
        """Start the integration layer."""
        if self.running:
            return
        
        # Initialize HTTP client
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=5)
        )
        
        # Connect to Redis
        await self.pipeline_cache.connect()
        
        # Start worker pool
        self.running = True
        for i in range(self.max_workers):
            worker_id = f"refinery_worker_{i+1}"
            self.workers.append(worker_id)
            task = asyncio.create_task(self._worker_loop(worker_id))
            self.worker_tasks.append(task)
        
        logger.info(f"Refinery integration started with {self.max_workers} workers")
    
    async def stop(self):
        """Stop the integration layer."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for workers to finish
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # Close HTTP client
        if self.http_client:
            await self.http_client.aclose()
        
        # Disconnect from Redis
        await self.pipeline_cache.disconnect()
        
        logger.info("Refinery integration stopped")
    
    async def enqueue_task(self, task: RefineryTask) -> bool:
        """Enqueue a task for processing."""
        if not self.running:
            raise RuntimeError("Integration layer not started")
        
        # Check cache first
        if self.cache_enabled and task.task_id in self.result_cache:
            logger.info(f"Task {task.task_id} found in cache")
            return True
        
        # Add to queue
        await self.task_queue.put(task)
        self.active_tasks[task.task_id] = task
        
        logger.info(f"Task {task.task_id} enqueued")
        return True
    
    async def get_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[RefineryResult]:
        """Get result for a task."""
        start_time = time.time()
        
        while timeout is None or (time.time() - start_time) < timeout:
            if task_id in self.result_cache:
                return self.result_cache[task_id]
            
            if task_id not in self.active_tasks:
                return None
            
            await asyncio.sleep(0.1)
        
        return None
    
    async def _worker_loop(self, worker_id: str):
        """Worker loop for processing tasks."""
        logger.info(f"Worker {worker_id} started")
        
        while self.running:
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
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _process_task(self, task: RefineryTask, worker_id: str):
        """Process a single task."""
        logger.info(f"Worker {worker_id} processing task {task.task_id}")
        
        try:
            # Execute task with telemetry
            result = await self._execute_task_with_telemetry(task)
            
            # Cache result
            if self.cache_enabled:
                self.result_cache[task.task_id] = result
            
            # Remove from active tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            
            logger.info(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            
            # Create error result
            error_result = RefineryResult(
                task_id=task.task_id,
                run_id=task.run_id,
                success=False,
                error=str(e),
                execution_time=0.0,
                timestamp=time.time(),
                worker_id=worker_id
            )
            
            # Cache error result
            if self.cache_enabled:
                self.result_cache[task.task_id] = error_result
            
            # Remove from active tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
    
    async def _execute_task_with_telemetry(self, task: RefineryTask) -> RefineryResult:
        """Execute task with telemetry tracing."""
        if self.telemetry_manager:
            try:
                if hasattr(self.telemetry_manager, 'trace_operation'):
                    async with self.telemetry_manager.trace_operation(
                        f"refinery.{task.action}", 
                        run_id=task.run_id, 
                        task_id=task.task_id
                    ):
                        return await self._execute_task(task)
                else:
                    return await self._execute_task(task)
            except Exception as e:
                logger.warning(f"Telemetry tracing failed: {e}")
                return await self._execute_task(task)
        else:
            return await self._execute_task(task)
    
    async def _execute_task(self, task: RefineryTask) -> RefineryResult:
        """Execute task by calling the refinery agent."""
        start_time = time.time()
        
        # Prepare request
        request_data = {
            "task_id": task.task_id,
            "action": task.action,
            "params": self._sanitize_payload(task.params)
        }
        
        # Make HTTP request
        url = f"{self.agent_url}/execute"
        
        try:
            response = await self.http_client.post(
                url,
                json=request_data,
                timeout=task.timeout
            )
            response.raise_for_status()
            
            response_data = response.json()
            
            execution_time = time.time() - start_time
            
            return RefineryResult(
                task_id=task.task_id,
                run_id=task.run_id,
                success=response_data.get("success", False),
                result=response_data.get("result"),
                error=response_data.get("error"),
                execution_time=execution_time,
                timestamp=time.time()
            )
            
        except httpx.HTTPStatusError as e:
            execution_time = time.time() - start_time
            logger.error(f"HTTP error for task {task.task_id}: {e}")
            
            return RefineryResult(
                task_id=task.task_id,
                run_id=task.run_id,
                success=False,
                error=f"HTTP {e.response.status_code}: {e.response.text}",
                execution_time=execution_time,
                timestamp=time.time()
            )
            
        except httpx.TimeoutException as e:
            execution_time = time.time() - start_time
            logger.error(f"Timeout for task {task.task_id}: {e}")
            
            return RefineryResult(
                task_id=task.task_id,
                run_id=task.run_id,
                success=False,
                error=f"Timeout after {task.timeout}s",
                execution_time=execution_time,
                timestamp=time.time()
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Unexpected error for task {task.task_id}: {e}")
            
            return RefineryResult(
                task_id=task.task_id,
                run_id=task.run_id,
                success=False,
                error=str(e),
                execution_time=execution_time,
                timestamp=time.time()
            )
    
    def _sanitize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize payload for security."""
        try:
            # Deep copy to avoid modifying original
            sanitized = json.loads(json.dumps(payload))
            
            # Remove potentially dangerous keys
            dangerous_keys = ['__class__', '__dict__', '__module__', '__reduce__']
            for key in dangerous_keys:
                if key in sanitized:
                    del sanitized[key]
            
            return sanitized
            
        except Exception as e:
            logger.warning(f"Payload sanitization failed: {e}")
            # Fallback: return empty dict
            return {}
    
    async def get_pipeline_state(self, run_id: str, session_id: str = "default") -> Optional[Dict[str, Any]]:
        """Get pipeline state from Redis cache."""
        return await self.pipeline_cache.get_pipeline(run_id, session_id)
    
    async def set_pipeline_state(self, run_id: str, session_id: str, pipeline_data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set pipeline state in Redis cache."""
        return await self.pipeline_cache.set_pipeline(run_id, session_id, pipeline_data, ttl)
    
    async def clear_pipeline_state(self, run_id: str, session_id: str = "default") -> bool:
        """Clear pipeline state from Redis cache."""
        return await self.pipeline_cache.delete_pipeline(run_id, session_id)
    
    async def list_active_pipelines(self) -> List[str]:
        """List all active pipeline keys."""
        return await self.pipeline_cache.list_pipelines()
    
    async def health_check(self) -> bool:
        """Check if the refinery agent is healthy."""
        try:
            response = await self.http_client.get(f"{self.agent_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get metrics from the refinery agent."""
        try:
            response = await self.http_client.get(f"{self.agent_url}/metrics")
            if response.status_code == 200:
                return {"metrics": response.text}
            return None
        except Exception as e:
            logger.error(f"Metrics retrieval failed: {e}")
            return None

# Factory function for easy integration
async def create_refinery_integration(
    agent_url: str = "http://localhost:8005",
    max_workers: int = 3,
    timeout: int = 300,
    cache_enabled: bool = True,
    telemetry_manager: Optional[Any] = None,
    redis_url: str = "redis://localhost:6379"
) -> RefineryIntegration:
    """Create and start a refinery integration instance."""
    integration = RefineryIntegration(
        agent_url=agent_url,
        max_workers=max_workers,
        timeout=timeout,
        cache_enabled=cache_enabled,
        telemetry_manager=telemetry_manager,
        redis_url=redis_url
    )
    
    await integration.start()
    return integration 