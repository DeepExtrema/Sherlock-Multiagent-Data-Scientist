#!/usr/bin/env python3
"""
Redis Pipeline Cache

Production-ready Redis-backed pipeline cache for feature engineering contexts.
Replaces in-memory pipeline storage with distributed, fault-tolerant storage.
"""

import json
import logging
import time
from typing import Any, Dict, Optional, List

import redis.asyncio as redis

logger = logging.getLogger(__name__)

class RedisPipelineCache:
    """
    Redis-backed pipeline cache for production environments.
    
    Provides distributed storage for feature engineering pipeline contexts
    with automatic serialization, TTL management, and fallback handling.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: Optional[int] = None):
        """
        Initialize Redis pipeline cache.
        
        Args:
            redis_url: Redis connection URL
            ttl: Time to live for cache entries in seconds (None = no expiration)
        """
        self.redis_url = redis_url
        self.ttl = ttl
        self.redis_client: Optional[redis.Redis] = None
        self.connected = False
        
        # In-memory fallback for when Redis is unavailable
        self._fallback_store: Dict[str, Any] = {}
        
    async def connect(self):
        """Establish Redis connection."""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            self.connected = True
            logger.info("Redis pipeline cache connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.connected = False
            
    async def disconnect(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self.connected = False
            logger.info("Redis pipeline cache disconnected")
    
    def _get_key(self, run_id: str, session_id: str = "default") -> str:
        """Generate cache key for pipeline context."""
        return f"refinery:pipeline:{run_id}:{session_id}"
    
    async def get_pipeline_context(self, run_id: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Get pipeline context from Redis with fallback.
        
        Args:
            run_id: Run identifier
            session_id: Session identifier
            
        Returns:
            Pipeline context dictionary
        """
        key = self._get_key(run_id, session_id)
        
        # Try Redis first
        if self.connected and self.redis_client:
            try:
                data = await self.redis_client.get(key)
                if data:
                    context = json.loads(data)
                    logger.debug(f"Retrieved pipeline context from Redis: {key}")
                    return context
            except Exception as e:
                logger.warning(f"Failed to get pipeline context from Redis: {e}")
        
        # Fallback to in-memory
        if key in self._fallback_store:
            logger.debug(f"Retrieved pipeline context from fallback: {key}")
            return self._fallback_store[key]
        
        # Create new context
        default_context = {
            "steps": [],
            "feature_names": [],
            "metadata": {},
            "roles": {},
            "created_at": time.time(),
            "run_id": run_id,
            "session_id": session_id
        }
        
        # Store the new context
        await self.set_pipeline_context(run_id, session_id, default_context)
        logger.info(f"Created new pipeline context: {key}")
        return default_context
    
    async def set_pipeline_context(self, run_id: str, session_id: str, context: Dict[str, Any]):
        """
        Set pipeline context in Redis with fallback.
        
        Args:
            run_id: Run identifier
            session_id: Session identifier
            context: Pipeline context to store
        """
        key = self._get_key(run_id, session_id)
        context["updated_at"] = time.time()
        
        # Try Redis first
        if self.connected and self.redis_client:
            try:
                serialized = json.dumps(context)
                if self.ttl:
                    await self.redis_client.setex(key, self.ttl, serialized)
                else:
                    await self.redis_client.set(key, serialized)
                logger.debug(f"Stored pipeline context in Redis: {key}")
            except Exception as e:
                logger.warning(f"Failed to set pipeline context in Redis: {e}")
        
        # Always store in fallback
        self._fallback_store[key] = context
        logger.debug(f"Stored pipeline context in fallback: {key}")
    
    async def update_pipeline_step(self, run_id: str, session_id: str, step_name: str, step_data: Any):
        """
        Update a specific step in the pipeline context.
        
        Args:
            run_id: Run identifier
            session_id: Session identifier
            step_name: Name of the pipeline step
            step_data: Data for the step
        """
        context = await self.get_pipeline_context(run_id, session_id)
        
        # Add step to pipeline
        context["steps"].append({
            "name": step_name,
            "data": step_data,
            "timestamp": time.time()
        })
        
        # Store specific step data
        context[step_name] = step_data
        
        await self.set_pipeline_context(run_id, session_id, context)
        logger.info(f"Updated pipeline step '{step_name}' for {run_id}:{session_id}")
    
    async def get_pipeline_step(self, run_id: str, session_id: str, step_name: str) -> Optional[Any]:
        """
        Get data for a specific pipeline step.
        
        Args:
            run_id: Run identifier
            session_id: Session identifier
            step_name: Name of the pipeline step
            
        Returns:
            Step data or None if not found
        """
        context = await self.get_pipeline_context(run_id, session_id)
        return context.get(step_name)
    
    async def delete_pipeline_context(self, run_id: str, session_id: str = "default"):
        """
        Delete pipeline context from storage.
        
        Args:
            run_id: Run identifier
            session_id: Session identifier
        """
        key = self._get_key(run_id, session_id)
        
        # Remove from Redis
        if self.connected and self.redis_client:
            try:
                await self.redis_client.delete(key)
                logger.debug(f"Deleted pipeline context from Redis: {key}")
            except Exception as e:
                logger.warning(f"Failed to delete pipeline context from Redis: {e}")
        
        # Remove from fallback
        self._fallback_store.pop(key, None)
        logger.debug(f"Deleted pipeline context from fallback: {key}")
    
    async def list_pipelines(self, pattern: str = "refinery:pipeline:*") -> List[str]:
        """
        List all pipeline keys matching pattern.
        
        Args:
            pattern: Redis key pattern
            
        Returns:
            List of matching keys
        """
        if self.connected and self.redis_client:
            try:
                keys = await self.redis_client.keys(pattern)
                return [key.decode() for key in keys]
            except Exception as e:
                logger.warning(f"Failed to list pipeline keys: {e}")
        
        # Fallback to in-memory keys
        return [key for key in self._fallback_store.keys() if key.startswith("refinery:pipeline:")]
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "connected": self.connected,
            "redis_url": self.redis_url,
            "ttl": self.ttl,
            "fallback_entries": len(self._fallback_store)
        }
        
        if self.connected and self.redis_client:
            try:
                info = await self.redis_client.info("memory")
                stats.update({
                    "redis_memory_used": info.get("used_memory", 0),
                    "redis_memory_human": info.get("used_memory_human", "0B")
                })
            except Exception as e:
                logger.warning(f"Failed to get Redis stats: {e}")
        
        return stats

# Convenience functions for backward compatibility
_default_cache: Optional[RedisPipelineCache] = None

async def init_default_cache(redis_url: str = "redis://localhost:6379", ttl: Optional[int] = None):
    """Initialize default pipeline cache."""
    global _default_cache
    _default_cache = RedisPipelineCache(redis_url, ttl)
    await _default_cache.connect()

async def get_pipeline_context(run_id: str, session_id: str = "default") -> Dict[str, Any]:
    """Get pipeline context using default cache."""
    if not _default_cache:
        await init_default_cache()
    return await _default_cache.get_pipeline_context(run_id, session_id)

async def set_pipeline_context(run_id: str, session_id: str, context: Dict[str, Any]):
    """Set pipeline context using default cache."""
    if not _default_cache:
        await init_default_cache()
    await _default_cache.set_pipeline_context(run_id, session_id, context)