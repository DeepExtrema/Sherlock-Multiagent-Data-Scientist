"""
Enhanced Data Quality FastAPI Routes

This module provides FastAPI routes for the enhanced data quality
checks with Prometheus metrics integration.
"""

import time
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge

from .handler import EnhancedDataQualityHandler
from .baseline import BaselineRegistry

logger = logging.getLogger(__name__)

# Prometheus metrics
dq_pass = Counter("refinery_dq_pass_total", "DQ pass", ["check"])
dq_fail = Counter("refinery_dq_fail_total", "DQ fail", ["check"])
pipeline_latency = Histogram(
    "refinery_pipeline_seconds", 
    "Wall time per pipeline", 
    ["step"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10]
)
drift_rate = Gauge("refinery_drift_rate", "Share of columns drifted", ["run_id"])


class DQRequest(BaseModel):
    """Base request model for data quality checks."""
    data_path: str
    run_id: Optional[str] = None
    baseline_id: Optional[str] = None


class SchemaConsistencyRequest(DQRequest):
    """Request model for schema consistency check."""
    pass


class MissingValuesRequest(DQRequest):
    """Request model for missing values check."""
    critical_columns: Optional[list[str]] = None


class DistributionsRequest(DQRequest):
    """Request model for distributions check."""
    baseline_id: str  # Required for distributions


class DuplicatesRequest(DQRequest):
    """Request model for duplicates check."""
    key_columns: Optional[list[str]] = None


class LeakageRequest(DQRequest):
    """Request model for leakage check."""
    target_column: str


class DriftRequest(DQRequest):
    """Request model for drift check."""
    baseline_id: str  # Required for drift
    drift_type: str = "both"  # 'data', 'target', or 'both'


class FreshnessRequest(DQRequest):
    """Request model for freshness check."""
    datetime_column: str
    max_age_hours: int = 24


class TargetBalanceRequest(DQRequest):
    """Request model for target balance check."""
    target_column: str


class BaselineRequest(BaseModel):
    """Request model for baseline operations."""
    data_path: str
    target_column: Optional[str] = None


class DQResponse(BaseModel):
    """Base response model for data quality checks."""
    status: str
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    run_id: Optional[str] = None


def get_dq_handler() -> EnhancedDataQualityHandler:
    """Dependency to get the data quality handler."""
    # This would be initialized with proper Redis client and config
    # For now, we'll create a placeholder
    from config import load_config
    config = load_config()
    
    # Initialize Redis client (this should be done properly in production)
    import aioredis
    redis_client = aioredis.from_url("redis://localhost:6379", decode_responses=False)
    
    baseline_registry = BaselineRegistry(redis_client)
    return EnhancedDataQualityHandler(baseline_registry, config)


def get_baseline_registry() -> BaselineRegistry:
    """Dependency to get the baseline registry."""
    import aioredis
    redis_client = aioredis.from_url("redis://localhost:6379", decode_responses=False)
    return BaselineRegistry(redis_client)


router = APIRouter(prefix="/dq", tags=["data-quality"])


@router.post("/check_schema_consistency", response_model=DQResponse)
async def check_schema_consistency(
    request: SchemaConsistencyRequest,
    handler: EnhancedDataQualityHandler = Depends(get_dq_handler)
) -> DQResponse:
    """Enhanced schema consistency check with baseline comparison."""
    start_time = time.time()
    
    try:
        params = request.dict()
        result = await handler.check_schema_consistency(params)
        
        duration = time.time() - start_time
        pipeline_latency.labels("check_schema_consistency").observe(duration)
        
        if result.get("status") == "success":
            dq_pass.labels("schema").inc()
        else:
            dq_fail.labels("schema").inc()
        
        return DQResponse(
            status=result.get("status", "error"),
            data=result,
            run_id=request.run_id
        )
        
    except Exception as e:
        duration = time.time() - start_time
        pipeline_latency.labels("check_schema_consistency").observe(duration)
        dq_fail.labels("schema").inc()
        
        logger.error(f"Schema consistency check failed: {e}")
        # Log only status and first 2 keys to reduce log volume
        if 'result' in locals():
            log_keys = list(result.keys())[:2] if isinstance(result, dict) else []
            logger.info(f"Result keys: {log_keys}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check_missing_values", response_model=DQResponse)
async def check_missing_values(
    request: MissingValuesRequest,
    handler: EnhancedDataQualityHandler = Depends(get_dq_handler)
) -> DQResponse:
    """Enhanced missing values check with row-level analysis."""
    start_time = time.time()
    
    try:
        params = request.dict()
        result = await handler.check_missing_values(params)
        
        duration = time.time() - start_time
        pipeline_latency.labels("check_missing_values").observe(duration)
        
        if result.get("status") == "success":
            dq_pass.labels("missing").inc()
        else:
            dq_fail.labels("missing").inc()
        
        return DQResponse(
            status=result.get("status", "error"),
            data=result,
            run_id=request.run_id
        )
        
    except Exception as e:
        duration = time.time() - start_time
        pipeline_latency.labels("check_missing_values").observe(duration)
        dq_fail.labels("missing").inc()
        
        logger.error(f"Missing values check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check_distributions", response_model=DQResponse)
async def check_distributions(
    request: DistributionsRequest,
    handler: EnhancedDataQualityHandler = Depends(get_dq_handler)
) -> DQResponse:
    """Enhanced distribution analysis with drift detection."""
    start_time = time.time()
    
    try:
        params = request.dict()
        result = await handler.check_distributions(params)
        
        duration = time.time() - start_time
        pipeline_latency.labels("check_distributions").observe(duration)
        
        if result.get("status") == "success":
            dq_pass.labels("distributions").inc()
            # Set drift rate metric
            drift_rate_value = result.get("drift_rate", 0.0)
            drift_rate.labels(run_id=request.run_id).set(drift_rate_value)
        else:
            dq_fail.labels("distributions").inc()
        
        return DQResponse(
            status=result.get("status", "error"),
            data=result,
            run_id=request.run_id
        )
        
    except Exception as e:
        duration = time.time() - start_time
        pipeline_latency.labels("check_distributions").observe(duration)
        dq_fail.labels("distributions").inc()
        
        logger.error(f"Distributions check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check_duplicates", response_model=DQResponse)
async def check_duplicates(
    request: DuplicatesRequest,
    handler: EnhancedDataQualityHandler = Depends(get_dq_handler)
) -> DQResponse:
    """Enhanced duplicate detection."""
    start_time = time.time()
    
    try:
        params = request.dict()
        result = await handler.check_duplicates(params)
        
        duration = time.time() - start_time
        pipeline_latency.labels("check_duplicates").observe(duration)
        
        if result.get("status") == "success":
            dq_pass.labels("duplicates").inc()
        else:
            dq_fail.labels("duplicates").inc()
        
        return DQResponse(
            status=result.get("status", "error"),
            data=result,
            run_id=request.run_id
        )
        
    except Exception as e:
        duration = time.time() - start_time
        pipeline_latency.labels("check_duplicates").observe(duration)
        dq_fail.labels("duplicates").inc()
        
        logger.error(f"Duplicates check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check_leakage", response_model=DQResponse)
async def check_leakage(
    request: LeakageRequest,
    handler: EnhancedDataQualityHandler = Depends(get_dq_handler)
) -> DQResponse:
    """Enhanced leakage detection."""
    start_time = time.time()
    
    try:
        params = request.dict()
        result = await handler.check_leakage(params)
        
        duration = time.time() - start_time
        pipeline_latency.labels("check_leakage").observe(duration)
        
        if result.get("status") == "success":
            dq_pass.labels("leakage").inc()
        else:
            dq_fail.labels("leakage").inc()
        
        return DQResponse(
            status=result.get("status", "error"),
            data=result,
            run_id=request.run_id
        )
        
    except Exception as e:
        duration = time.time() - start_time
        pipeline_latency.labels("check_leakage").observe(duration)
        dq_fail.labels("leakage").inc()
        
        logger.error(f"Leakage check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check_drift", response_model=DQResponse)
async def check_drift(
    request: DriftRequest,
    handler: EnhancedDataQualityHandler = Depends(get_dq_handler)
) -> DQResponse:
    """Enhanced drift detection."""
    start_time = time.time()
    
    try:
        params = request.dict()
        result = await handler.check_drift(params)
        
        duration = time.time() - start_time
        pipeline_latency.labels("check_drift").observe(duration)
        
        if result.get("status") == "success":
            dq_pass.labels("drift").inc()
        else:
            dq_fail.labels("drift").inc()
        
        return DQResponse(
            status=result.get("status", "error"),
            data=result,
            run_id=request.run_id
        )
        
    except Exception as e:
        duration = time.time() - start_time
        pipeline_latency.labels("check_drift").observe(duration)
        dq_fail.labels("drift").inc()
        
        logger.error(f"Drift check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check_freshness", response_model=DQResponse)
async def check_freshness(
    request: FreshnessRequest,
    handler: EnhancedDataQualityHandler = Depends(get_dq_handler)
) -> DQResponse:
    """Check data freshness and latency."""
    start_time = time.time()
    
    try:
        params = request.dict()
        result = await handler.check_freshness(params)
        
        duration = time.time() - start_time
        pipeline_latency.labels("check_freshness").observe(duration)
        
        if result.get("status") == "success":
            dq_pass.labels("freshness").inc()
        else:
            dq_fail.labels("freshness").inc()
        
        return DQResponse(
            status=result.get("status", "error"),
            data=result,
            run_id=request.run_id
        )
        
    except Exception as e:
        duration = time.time() - start_time
        pipeline_latency.labels("check_freshness").observe(duration)
        dq_fail.labels("freshness").inc()
        
        logger.error(f"Freshness check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check_target_balance", response_model=DQResponse)
async def check_target_balance(
    request: TargetBalanceRequest,
    handler: EnhancedDataQualityHandler = Depends(get_dq_handler)
) -> DQResponse:
    """Check target class balance."""
    start_time = time.time()
    
    try:
        params = request.dict()
        result = await handler.check_target_balance(params)
        
        duration = time.time() - start_time
        pipeline_latency.labels("check_target_balance").observe(duration)
        
        if result.get("status") == "success":
            dq_pass.labels("target_balance").inc()
        else:
            dq_fail.labels("target_balance").inc()
        
        return DQResponse(
            status=result.get("status", "error"),
            data=result,
            run_id=request.run_id
        )
        
    except Exception as e:
        duration = time.time() - start_time
        pipeline_latency.labels("check_target_balance").observe(duration)
        dq_fail.labels("target_balance").inc()
        
        logger.error(f"Target balance check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/set_baseline")
async def set_baseline(
    request: BaselineRequest,
    registry: BaselineRegistry = Depends(get_baseline_registry)
) -> Dict[str, Any]:
    """Save a baseline dataset."""
    try:
        import pandas as pd
        
        # Load data
        if request.data_path.endswith('.csv'):
            df = pd.read_csv(request.data_path)
        elif request.data_path.endswith('.parquet'):
            df = pd.read_parquet(request.data_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Save baseline
        baseline_id = await registry.save(df, request.target_column)
        
        return {
            "status": "success",
            "baseline_id": baseline_id,
            "data_shape": df.shape,
            "target_column": request.target_column
        }
        
    except Exception as e:
        logger.error(f"Failed to set baseline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list_baselines")
async def list_baselines(
    registry: BaselineRegistry = Depends(get_baseline_registry)
) -> Dict[str, Any]:
    """List all available baselines."""
    try:
        baselines = await registry.list_baselines()
        return {
            "status": "success",
            "baselines": baselines,
            "count": len(baselines)
        }
        
    except Exception as e:
        logger.error(f"Failed to list baselines: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/baseline/{baseline_id}")
async def delete_baseline(
    baseline_id: str,
    registry: BaselineRegistry = Depends(get_baseline_registry)
) -> Dict[str, Any]:
    """Delete a baseline dataset."""
    try:
        deleted = await registry.delete(baseline_id)
        return {
            "status": "success" if deleted else "not_found",
            "baseline_id": baseline_id,
            "deleted": deleted
        }
        
    except Exception as e:
        logger.error(f"Failed to delete baseline {baseline_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 