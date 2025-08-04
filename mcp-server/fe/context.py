"""
Pipeline Context Management for Feature Engineering

This module provides Redis-backed pipeline state management for
feature engineering workflows.
"""

import dill
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import pandas as pd
import aioredis

logger = logging.getLogger(__name__)


@dataclass
class PipelineCtx:
    """Pipeline context for feature engineering workflows."""
    
    df: Optional[pd.DataFrame] = None
    roles: Dict[str, str] = field(default_factory=dict)
    steps: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate context after initialization."""
        if self.df is not None and not isinstance(self.df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame or None")


class PipelineContextManager:
    """Manages Redis-backed pipeline context for feature engineering."""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.context_prefix = "refinery:fe:ctx:"
        self.default_ttl = 3600  # 1 hour default TTL
    
    async def get_ctx(self, run_id: str) -> PipelineCtx:
        """
        Get pipeline context from Redis.
        
        Args:
            run_id: Unique run identifier
            
        Returns:
            PipelineCtx object, or new empty context if not found
        """
        try:
            key = f"{self.context_prefix}{run_id}"
            blob = await self.redis.get(key)
            
            if blob is None:
                logger.info(f"No existing context found for run_id: {run_id}")
                return PipelineCtx(df=None, roles={})
            
            ctx = dill.loads(blob)
            logger.debug(f"Loaded context for run_id: {run_id} with {len(ctx.steps)} steps")
            return ctx
            
        except Exception as e:
            logger.error(f"Failed to load context for run_id {run_id}: {e}")
            return PipelineCtx(df=None, roles={})
    
    async def save_ctx(self, run_id: str, ctx: PipelineCtx, ttl: Optional[int] = None) -> bool:
        """
        Save pipeline context to Redis.
        
        Args:
            run_id: Unique run identifier
            ctx: Pipeline context to save
            ttl: Time to live in seconds (default: 1 hour)
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            key = f"{self.context_prefix}{run_id}"
            blob = dill.dumps(ctx)
            
            await self.redis.setex(
                key, 
                ttl or self.default_ttl, 
                blob
            )
            
            logger.debug(f"Saved context for run_id: {run_id} with {len(ctx.steps)} steps")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save context for run_id {run_id}: {e}")
            return False
    
    async def delete_ctx(self, run_id: str) -> bool:
        """
        Delete pipeline context from Redis.
        
        Args:
            run_id: Unique run identifier
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            key = f"{self.context_prefix}{run_id}"
            deleted = await self.redis.delete(key)
            logger.info(f"Deleted context for run_id: {run_id}")
            return deleted > 0
            
        except Exception as e:
            logger.error(f"Failed to delete context for run_id {run_id}: {e}")
            return False
    
    async def list_contexts(self) -> List[str]:
        """
        List all pipeline contexts.
        
        Returns:
            List of run IDs with active contexts
        """
        try:
            keys = await self.redis.keys(f"{self.context_prefix}*")
            run_ids = [key.decode().replace(self.context_prefix, "") for key in keys]
            return run_ids
            
        except Exception as e:
            logger.error(f"Failed to list contexts: {e}")
            return []
    
    async def get_context_info(self, run_id: str) -> Dict[str, Any]:
        """
        Get context information without loading the full DataFrame.
        
        Args:
            run_id: Unique run identifier
            
        Returns:
            Dictionary with context metadata
        """
        try:
            ctx = await self.get_ctx(run_id)
            
            info = {
                "run_id": run_id,
                "steps": ctx.steps,
                "roles": ctx.roles,
                "meta": ctx.meta,
                "has_data": ctx.df is not None,
                "data_shape": ctx.df.shape if ctx.df is not None else None,
                "step_count": len(ctx.steps)
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get context info for run_id {run_id}: {e}")
            return {"run_id": run_id, "error": str(e)}
    
    async def cleanup_expired_contexts(self) -> int:
        """
        Clean up expired contexts (Redis TTL handles this automatically).
        This method can be used for additional cleanup logic if needed.
        
        Returns:
            Number of contexts cleaned up
        """
        try:
            # Redis automatically handles TTL expiration
            # This method can be extended for additional cleanup logic
            logger.info("Context cleanup completed (handled by Redis TTL)")
            return 0
            
        except Exception as e:
            logger.error(f"Failed to cleanup contexts: {e}")
            return 0


# Utility functions for common context operations
async def get_ctx(run_id: str, redis_client: aioredis.Redis) -> PipelineCtx:
    """Convenience function to get pipeline context."""
    manager = PipelineContextManager(redis_client)
    return await manager.get_ctx(run_id)


async def save_ctx(run_id: str, ctx: PipelineCtx, redis_client: aioredis.Redis, ttl: int = 3600) -> bool:
    """Convenience function to save pipeline context."""
    manager = PipelineContextManager(redis_client)
    return await manager.save_ctx(run_id, ctx, ttl)


async def delete_ctx(run_id: str, redis_client: aioredis.Redis) -> bool:
    """Convenience function to delete pipeline context."""
    manager = PipelineContextManager(redis_client)
    return await manager.delete_ctx(run_id) 