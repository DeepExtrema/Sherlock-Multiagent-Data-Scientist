"""
Decorators for Feature Engineering

This module provides decorators for conditional execution, metrics tracking,
and other utility functions for feature engineering workflows.
"""

import time
import logging
import functools
from typing import Callable, Any, Dict, Optional
from prometheus_client import Histogram, Counter

logger = logging.getLogger(__name__)

# Prometheus metrics for feature engineering
fe_step_latency = Histogram(
    "refinery_fe_step_seconds", 
    "Feature engineering step latency", 
    ["step"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10]
)
fe_step_total = Counter("refinery_fe_step_total", "Feature engineering steps", ["step", "status"])


def skip_if(predicate: Callable) -> Callable:
    """
    Decorator to conditionally skip function execution.
    
    Args:
        predicate: Function that takes params and returns True to skip
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(params: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
            try:
                if predicate(params):
                    logger.info(f"Skipping {func.__name__}: {predicate.__name__} returned True")
                    fe_step_total.labels(step=func.__name__, status="skipped").inc()
                    return {
                        "status": "success",
                        "skipped": True,
                        "reason": predicate.__name__
                    }
                
                # Execute the function
                start_time = time.time()
                result = await func(params, *args, **kwargs)
                duration = time.time() - start_time
                
                # Record metrics
                fe_step_latency.labels(step=func.__name__).observe(duration)
                fe_step_total.labels(step=func.__name__, status="success").inc()
                
                return result
                
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                fe_step_total.labels(step=func.__name__, status="error").inc()
                return {"status": "error", "error": str(e)}
        
        return wrapper
    return decorator


def track_metrics(step_name: Optional[str] = None) -> Callable:
    """
    Decorator to track metrics for feature engineering steps.
    
    Args:
        step_name: Optional custom step name for metrics
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(params: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
            step = step_name or func.__name__
            
            try:
                start_time = time.time()
                result = await func(params, *args, **kwargs)
                duration = time.time() - start_time
                
                # Record metrics
                fe_step_latency.labels(step=step).observe(duration)
                fe_step_total.labels(step=step, status="success").inc()
                
                return result
                
            except Exception as e:
                logger.error(f"Error in {step}: {e}")
                fe_step_total.labels(step=step, status="error").inc()
                return {"status": "error", "error": str(e)}
        
        return wrapper
    return decorator


def validate_context(func: Callable) -> Callable:
    """
    Decorator to validate pipeline context before execution.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    async def wrapper(params: Dict[str, Any], ctx: Any, *args, **kwargs) -> Dict[str, Any]:
        try:
            # Check if context has data
            if ctx is None or ctx.df is None:
                return {
                    "status": "error",
                    "error": "No data in pipeline context"
                }
            
            # Check if run_id is provided
            if 'run_id' not in params:
                return {
                    "status": "error",
                    "error": "run_id is required"
                }
            
            return await func(params, ctx, *args, **kwargs)
            
        except Exception as e:
            logger.error(f"Context validation failed in {func.__name__}: {e}")
            return {"status": "error", "error": str(e)}
    
    return wrapper


# Predicate functions for conditional execution
def no_text_columns(params: Dict[str, Any]) -> bool:
    """Check if there are no text columns in the data."""
    # This would need to be implemented based on the actual data
    # For now, return False to always execute
    return False


def no_categorical_columns(params: Dict[str, Any]) -> bool:
    """Check if there are no categorical columns in the data."""
    # This would need to be implemented based on the actual data
    # For now, return False to always execute
    return False


def no_datetime_columns(params: Dict[str, Any]) -> bool:
    """Check if there are no datetime columns in the data."""
    # This would need to be implemented based on the actual data
    # For now, return False to always execute
    return False


def no_missing_values(params: Dict[str, Any]) -> bool:
    """Check if there are no missing values in the data."""
    # This would need to be implemented based on the actual data
    # For now, return False to always execute
    return False


def high_cardinality_detected(params: Dict[str, Any]) -> bool:
    """Check if high cardinality categorical columns are detected."""
    # This would need to be implemented based on the actual data
    # For now, return False to always execute
    return False


# Example usage decorators
@skip_if(no_text_columns)
async def vectorise_text_features(params: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    """Example function that would be skipped if no text columns exist."""
    # Implementation would go here
    return {"status": "success", "message": "Text features vectorized"}


@skip_if(no_categorical_columns)
async def encode_categorical_features(params: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    """Example function that would be skipped if no categorical columns exist."""
    # Implementation would go here
    return {"status": "success", "message": "Categorical features encoded"}


@skip_if(no_datetime_columns)
async def generate_datetime_features(params: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    """Example function that would be skipped if no datetime columns exist."""
    # Implementation would go here
    return {"status": "success", "message": "Datetime features generated"}


@skip_if(no_missing_values)
async def impute_missing_values(params: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    """Example function that would be skipped if no missing values exist."""
    # Implementation would go here
    return {"status": "success", "message": "Missing values imputed"} 