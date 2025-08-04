#!/usr/bin/env python3
"""
Refinery Agent - Data Quality and Feature Engineering Service
Provides 15 actions for DQ validation and FE pipeline management.
"""

import asyncio
import json
import logging
import time
import os
from typing import Dict, Any, Optional, Literal
from pathlib import Path

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
import redis.asyncio as redis
from motor.motor_asyncio import AsyncIOMotorClient
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, multiprocess
from fastapi.responses import Response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Refinery Agent",
    description="Data Quality and Feature Engineering Service",
    version="0.1.0"
)

# Add security middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])  # Configure per environment
# app.add_middleware(HTTPSRedirectMiddleware)  # Uncomment for HTTPS-only

# Prometheus metrics (multiprocess-safe for gunicorn)
REQUEST_COUNT = Counter('refinery_requests_total', 'Total requests', ['action', 'status'], multiprocess_mode='livesum')
REQUEST_DURATION = Histogram('refinery_request_duration_seconds', 'Request duration', ['action'], multiprocess_mode='livesum')
ACTIVE_PIPELINES = Gauge('refinery_active_pipelines', 'Number of active pipelines', multiprocess_mode='livesum')
DATASET_SIZE = Histogram('refinery_dataset_size_rows', 'Dataset size in rows', ['action'], multiprocess_mode='livesum')

# Configuration from environment
DRIFT_NUMERIC_P95_THRESHOLD = float(os.getenv('DRIFT_NUMERIC_P95_THRESHOLD', '0.1'))
DRIFT_CATEGORICAL_PSI_THRESHOLD = float(os.getenv('DRIFT_CATEGORICAL_PSI_THRESHOLD', '0.25'))
MISSING_VALUES_THRESHOLD = float(os.getenv('MISSING_VALUES_THRESHOLD', '0.5'))
CORRELATION_THRESHOLD = float(os.getenv('CORRELATION_THRESHOLD', '0.95'))

# In-memory pipeline cache (will be replaced with Redis)
_PIPELINES: Dict[str, Dict[str, Any]] = {}

# Action types for validation
DQ_ACTIONS = Literal[
    "check_schema_consistency",
    "check_missing_values", 
    "check_distributions",
    "check_duplicates",
    "check_leakage",
    "check_drift"
]

FE_ACTIONS = Literal[
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

ALL_ACTIONS = Literal[DQ_ACTIONS, FE_ACTIONS]

# Pydantic models
class TaskRequest(BaseModel):
    task_id: str
    action: ALL_ACTIONS
    params: Dict[str, Any] = Field(default_factory=dict)

class TaskResponse(BaseModel):
    task_id: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float
    timestamp: float

# Utility functions
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from various file formats."""
    path = Path(file_path)
    if path.suffix == '.parquet':
        return pd.read_parquet(file_path)
    elif path.suffix == '.csv':
        return pd.read_csv(file_path)
    elif path.suffix == '.json':
        return pd.read_json(file_path)
    elif path.suffix in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

def get_pipeline_context(run_id: str, session_id: str = "default") -> Dict[str, Any]:
    """Get or create pipeline context."""
    key = f"{run_id}:{session_id}"
    if key not in _PIPELINES:
        _PIPELINES[key] = {
            "steps": [],
            "features": [],
            "metadata": {},
            "created_at": time.time()
        }
        ACTIVE_PIPELINES.inc()
    return _PIPELINES[key]

# Data Quality Handlers
async def check_schema_consistency(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check schema consistency against expected schema."""
    data_path = params["data_path"]
    expected_schema = params.get("expected_schema", {})
    
    df = load_data(data_path)
    DATASET_SIZE.labels(action="check_schema_consistency").observe(len(df))
    
    actual_schema = {
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "shape": df.shape
    }
    
    if expected_schema:
        # Compare schemas
        missing_cols = set(expected_schema.get("columns", [])) - set(actual_schema["columns"])
        extra_cols = set(actual_schema["columns"]) - set(expected_schema.get("columns", []))
        
        return {
            "status": "pass" if not missing_cols and not extra_cols else "fail",
            "actual_schema": actual_schema,
            "missing_columns": list(missing_cols),
            "extra_columns": list(extra_cols),
            "dtype_mismatches": []
        }
    
    return {
        "status": "pass",
        "actual_schema": actual_schema
    }

async def check_missing_values(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check for missing values in the dataset."""
    data_path = params["data_path"]
    threshold_pct = params.get("threshold_pct", MISSING_VALUES_THRESHOLD)
    
    df = load_data(data_path)
    DATASET_SIZE.labels(action="check_missing_values").observe(len(df))
    
    missing_pct = df.isnull().mean() * 100
    cols_over_threshold = missing_pct[missing_pct > threshold_pct * 100].index.tolist()
    
    return {
        "cols_over_threshold": cols_over_threshold,
        "missing_summary": missing_pct.to_dict(),
        "total_missing": df.isnull().sum().sum(),
        "threshold_exceeded": len(cols_over_threshold) > 0
    }

async def check_distributions(params: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze data distributions and detect outliers."""
    data_path = params["data_path"]
    
    df = load_data(data_path)
    DATASET_SIZE.labels(action="check_distributions").observe(len(df))
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    results = {
        "numeric_distributions": {},
        "categorical_distributions": {},
        "outliers": {}
    }
    
    # Numeric distributions
    for col in numeric_cols:
        results["numeric_distributions"][col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "q25": float(df[col].quantile(0.25)),
            "q75": float(df[col].quantile(0.75))
        }
        
        # Simple outlier detection (IQR method)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        results["outliers"][col] = len(outliers)
    
    # Categorical distributions
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        results["categorical_distributions"][col] = {
            "unique_count": len(value_counts),
            "top_values": value_counts.head(5).to_dict(),
            "null_count": df[col].isnull().sum()
        }
    
    return results

async def check_duplicates(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check for duplicate rows and high correlation pairs."""
    data_path = params["data_path"]
    id_cols = params.get("id_cols", [])
    
    df = load_data(data_path)
    DATASET_SIZE.labels(action="check_duplicates").observe(len(df))
    
    # Duplicate rows
    dup_rows = df.duplicated().sum()
    
    # Duplicate IDs if specified
    dup_ids = 0
    if id_cols:
        dup_ids = df.duplicated(subset=id_cols).sum()
    
    # High correlation pairs
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr().abs()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > CORRELATION_THRESHOLD:
                    high_corr_pairs.append({
                        "col1": corr_matrix.columns[i],
                        "col2": corr_matrix.columns[j],
                        "correlation": float(corr_matrix.iloc[i, j])
                    })
    else:
        high_corr_pairs = []
    
    return {
        "duplicate_rows": int(dup_rows),
        "duplicate_ids": int(dup_ids),
        "high_correlation_pairs": high_corr_pairs,
        "total_rows": len(df)
    }

async def check_leakage(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check for data leakage between features and target."""
    data_path = params["data_path"]
    target_col = params["target_col"]
    
    df = load_data(data_path)
    DATASET_SIZE.labels(action="check_leakage").observe(len(df))
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    # Calculate correlations with target
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    target_correlations = {}
    suspicious_cols = []
    
    for col in numeric_cols:
        if col != target_col:
            corr = df[col].corr(df[target_col])
            if abs(corr) > CORRELATION_THRESHOLD:  # Very high correlation
                suspicious_cols.append(col)
            target_correlations[col] = float(corr)
    
    return {
        "suspicious_cols": suspicious_cols,
        "target_correlations": target_correlations,
        "high_correlation_threshold": CORRELATION_THRESHOLD
    }

async def check_drift(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check for data drift between reference and current datasets."""
    reference_path = params["reference_path"]
    current_path = params["current_path"]
    
    ref_df = load_data(reference_path)
    curr_df = load_data(current_path)
    
    DATASET_SIZE.labels(action="check_drift").observe(len(ref_df) + len(curr_df))
    
    # Enhanced drift detection using configurable thresholds
    numeric_cols = ref_df.select_dtypes(include=[np.number]).columns
    drift_results = {}
    
    for col in numeric_cols:
        if col in curr_df.columns:
            ref_mean = ref_df[col].mean()
            curr_mean = curr_df[col].mean()
            ref_std = ref_df[col].std()
            
            # Calculate drift score (simplified)
            drift_score = abs(curr_mean - ref_mean) / ref_std if ref_std > 0 else 0
            
            # Use configurable threshold
            threshold = params.get("drift_threshold", DRIFT_NUMERIC_P95_THRESHOLD)
            
            drift_results[col] = {
                "drift_score": float(drift_score),
                "reference_mean": float(ref_mean),
                "current_mean": float(curr_mean),
                "drift_detected": drift_score > threshold,
                "threshold_used": threshold
            }
    
    return {
        "drift_results": drift_results,
        "drift_threshold": DRIFT_NUMERIC_P95_THRESHOLD,
        "columns_analyzed": list(drift_results.keys())
    }

# Feature Engineering Handlers
async def assign_feature_roles(params: Dict[str, Any]) -> Dict[str, Any]:
    """Assign feature roles based on data types and patterns."""
    data_path = params["input_path"]
    run_id = params["run_id"]
    session_id = params.get("session_id", "default")
    overrides = params.get("overrides", {})
    
    df = load_data(data_path)
    DATASET_SIZE.labels(action="assign_feature_roles").observe(len(df))
    
    pipeline = get_pipeline_context(run_id, session_id)
    
    # Auto-detect feature roles
    roles = {}
    for col in df.columns:
        if col in overrides:
            roles[col] = overrides[col]
        elif df[col].dtype in ['int64', 'float64']:
            if col.lower() in ['id', 'index', 'key']:
                roles[col] = 'id'
            elif df[col].nunique() == 2:
                roles[col] = 'binary'
            else:
                roles[col] = 'numeric'
        elif df[col].dtype == 'object':
            if df[col].str.len().mean() > 50:
                roles[col] = 'text'
            else:
                roles[col] = 'categorical'
        elif df[col].dtype == 'datetime64[ns]':
            roles[col] = 'datetime'
        else:
            roles[col] = 'unknown'
    
    pipeline["roles"] = roles
    pipeline["features"] = list(df.columns)
    
    return {
        "assigned_roles": roles,
        "feature_count": len(roles)
    }

async def impute_missing_values(params: Dict[str, Any]) -> Dict[str, Any]:
    """Impute missing values using various strategies."""
    data_path = params["input_path"]
    run_id = params["run_id"]
    session_id = params.get("session_id", "default")
    strategy = params.get("strategy", "auto")
    
    df = load_data(data_path)
    DATASET_SIZE.labels(action="impute_missing_values").observe(len(df))
    
    pipeline = get_pipeline_context(run_id, session_id)
    
    # Simple imputation strategies
    df_imputed = df.copy()
    imputation_summary = {}
    
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if strategy == "auto":
                if df[col].dtype in ['int64', 'float64']:
                    df_imputed[col] = df[col].fillna(df[col].median())
                    imputation_summary[col] = "median"
                else:
                    df_imputed[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "unknown")
                    imputation_summary[col] = "mode"
            elif strategy == "mean" and df[col].dtype in ['int64', 'float64']:
                df_imputed[col] = df[col].fillna(df[col].mean())
                imputation_summary[col] = "mean"
            elif strategy == "zero" and df[col].dtype in ['int64', 'float64']:
                df_imputed[col] = df[col].fillna(0)
                imputation_summary[col] = "zero"
    
    pipeline["imputation_strategy"] = imputation_summary
    pipeline["steps"].append("impute_missing_values")
    
    return {
        "imputed_columns": list(imputation_summary.keys()),
        "imputation_strategy": imputation_summary,
        "remaining_nulls": df_imputed.isnull().sum().sum()
    }

async def scale_numeric_features(params: Dict[str, Any]) -> Dict[str, Any]:
    """Scale numeric features using various methods."""
    run_id = params["run_id"]
    session_id = params.get("session_id", "default")
    method = params.get("method", "standard")
    
    pipeline = get_pipeline_context(run_id, session_id)
    
    pipeline["scaling_method"] = method
    pipeline["steps"].append("scale_numeric_features")
    
    return {
        "scaling_method": method,
        "scaled_features": "numeric_features"
    }

async def encode_categorical_features(params: Dict[str, Any]) -> Dict[str, Any]:
    """Encode categorical features."""
    run_id = params["run_id"]
    session_id = params.get("session_id", "default")
    strategy = params.get("strategy", "auto")
    
    pipeline = get_pipeline_context(run_id, session_id)
    
    pipeline["encoding_strategy"] = strategy
    pipeline["steps"].append("encode_categorical_features")
    
    return {
        "encoding_strategy": strategy,
        "encoded_features": "categorical_features"
    }

async def generate_datetime_features(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate datetime features."""
    run_id = params["run_id"]
    session_id = params.get("session_id", "default")
    country = params.get("country", "US")
    
    pipeline = get_pipeline_context(run_id, session_id)
    
    pipeline["datetime_country"] = country
    pipeline["steps"].append("generate_datetime_features")
    
    return {
        "country": country,
        "generated_features": ["hour", "day", "month", "year", "dayofweek"]
    }

async def vectorise_text_features(params: Dict[str, Any]) -> Dict[str, Any]:
    """Vectorize text features."""
    run_id = params["run_id"]
    session_id = params.get("session_id", "default")
    model = params.get("model", "mini-lm")
    max_features = params.get("max_features", 5000)
    
    pipeline = get_pipeline_context(run_id, session_id)
    
    pipeline["text_model"] = model
    pipeline["max_features"] = max_features
    pipeline["steps"].append("vectorise_text_features")
    
    return {
        "text_model": model,
        "max_features": max_features,
        "vectorized_features": "text_features"
    }

async def generate_interactions(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate feature interactions."""
    run_id = params["run_id"]
    session_id = params.get("session_id", "default")
    max_degree = params.get("max_degree", 2)
    
    pipeline = get_pipeline_context(run_id, session_id)
    
    pipeline["interaction_degree"] = max_degree
    pipeline["steps"].append("generate_interactions")
    
    return {
        "interaction_degree": max_degree,
        "new_features": f"interaction_features_degree_{max_degree}"
    }

async def select_features(params: Dict[str, Any]) -> Dict[str, Any]:
    """Select features using various methods."""
    run_id = params["run_id"]
    session_id = params.get("session_id", "default")
    method = params.get("method", "shap_top_k")
    k = params.get("k", 100)
    
    pipeline = get_pipeline_context(run_id, session_id)
    
    # Wrap feature selection in try/except for robustness
    try:
        # In a real implementation, this would use sklearn.feature_selection.SelectKBest
        # and handle all-NaN columns gracefully
        pipeline["feature_selection_method"] = method
        pipeline["selected_k"] = k
        pipeline["steps"].append("select_features")
        
        return {
            "selection_method": method,
            "selected_count": k,
            "selected_features": f"top_{k}_features"
        }
    except Exception as e:
        logger.error(f"Feature selection failed: {e}")
        raise ValueError(f"Feature selection failed: {str(e)}")

async def save_fe_pipeline(params: Dict[str, Any]) -> Dict[str, Any]:
    """Save the feature engineering pipeline."""
    input_path = params["input_path"]
    export_data_path = params["export_data_path"]
    export_pipeline_path = params["export_pipeline_path"]
    run_id = params["run_id"]
    session_id = params.get("session_id", "default")
    
    df = load_data(input_path)
    DATASET_SIZE.labels(action="save_fe_pipeline").observe(len(df))
    
    pipeline = get_pipeline_context(run_id, session_id)
    
    # Save pipeline metadata
    pipeline["finalized_at"] = time.time()
    pipeline["export_paths"] = {
        "data": export_data_path,
        "pipeline": export_pipeline_path
    }
    
    # Generate a unique identifier for the pipeline
    pipeline_id = f"pipeline_{run_id}_{int(time.time())}"
    
    # In a real implementation, you would:
    # 1. Apply all pipeline steps to the data
    # 2. Save the transformed data
    # 3. Save the pipeline object
    # 4. Return a storage-agnostic identifier
    
    return {
        "pipeline_id": pipeline_id,
        "data_path": export_data_path,
        "pipeline_path": export_pipeline_path,
        "metadata": pipeline,
        "final_shape": df.shape,
        "storage_url": f"storage://pipelines/{pipeline_id}"  # Storage-agnostic identifier
    }

# Main execution endpoint
@app.post("/execute", response_model=TaskResponse)
async def execute(req: TaskRequest):
    start_time = time.time()
    
    try:
        # Action dispatcher
        dispatcher = {
            # Data Quality actions
            "check_schema_consistency": check_schema_consistency,
            "check_missing_values": check_missing_values,
            "check_distributions": check_distributions,
            "check_duplicates": check_duplicates,
            "check_leakage": check_leakage,
            "check_drift": check_drift,
            # Feature Engineering actions
            "assign_feature_roles": assign_feature_roles,
            "impute_missing_values": impute_missing_values,
            "scale_numeric_features": scale_numeric_features,
            "encode_categorical_features": encode_categorical_features,
            "generate_datetime_features": generate_datetime_features,
            "vectorise_text_features": vectorise_text_features,
            "generate_interactions": generate_interactions,
            "select_features": select_features,
            "save_fe_pipeline": save_fe_pipeline
        }
        
        if req.action not in dispatcher:
            raise ValueError(f"Unsupported action: {req.action}")
        
        # Execute the action
        handler = dispatcher[req.action]
        result = await handler(req.params)
        
        execution_time = time.time() - start_time
        
        # Record metrics
        REQUEST_COUNT.labels(action=req.action, status="success").inc()
        REQUEST_DURATION.labels(action=req.action).observe(execution_time)
        
        return TaskResponse(
            task_id=req.task_id,
            success=True,
            result=result,
            execution_time=execution_time,
            timestamp=time.time()
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        
        # Record error metrics
        REQUEST_COUNT.labels(action=req.action, status="error").inc()
        REQUEST_DURATION.labels(action=req.action).observe(execution_time)
        
        logger.error(f"Task {req.task_id} failed: {str(e)}")
        
        return TaskResponse(
            task_id=req.task_id,
            success=False,
            error=str(e),
            execution_time=execution_time,
            timestamp=time.time()
        )

# Health endpoint
@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}

# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Pipeline status endpoint
@app.get("/pipelines")
async def list_pipelines():
    return {
        "active_pipelines": len(_PIPELINES),
        "pipelines": list(_PIPELINES.keys())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005) 