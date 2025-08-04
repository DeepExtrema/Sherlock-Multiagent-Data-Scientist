#!/usr/bin/env python3
"""
Refinery Agent - Universal Data Quality & Feature Engineering Service
Provides comprehensive data quality validation and feature engineering pipeline management.
Clear separation between DQ checking (read-only) and FE processing (transformative) modes.
Enhanced with FE module integration for advanced capabilities.
"""

import asyncio
import json
import logging
import time
import os
from typing import Dict, Any, Optional, Literal, Union, List
from pathlib import Path
from enum import Enum

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, validator
import redis.asyncio as redis
from motor.motor_asyncio import AsyncIOMotorClient
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, multiprocess
from fastapi.responses import Response

# FE Module Integration
try:
    from fe.imputation import impute_missing_values as fe_impute
    from fe.encoding import encode_categorical_features as fe_encode
    from fe.pruning import prune_multicollinear_features as fe_prune, generate_interactions as fe_interactions
    from fe.context import PipelineCtx, PipelineContextManager
    FE_MODULE_AVAILABLE = True
except ImportError:
    FE_MODULE_AVAILABLE = False
    logging.warning("FE module not available - advanced features disabled")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Refinery Agent",
    description="Universal Data Quality & Feature Engineering Service with Advanced FE Module Integration",
    version="2.0.0"
)

# Add security middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])  # Configure per environment
# app.add_middleware(HTTPSRedirectMiddleware)  # Uncomment for HTTPS-only

# Prometheus metrics (multiprocess-safe for gunicorn)
REQUEST_COUNT = Counter('refinery_requests_total', 'Total requests', ['action', 'mode', 'backend', 'status'], multiprocess_mode='livesum')
REQUEST_DURATION = Histogram('refinery_request_duration_seconds', 'Request duration', ['action', 'mode', 'backend'], multiprocess_mode='livesum')
ACTIVE_PIPELINES = Gauge('refinery_active_pipelines', 'Number of active pipelines', multiprocess_mode='livesum')
DATASET_SIZE = Histogram('refinery_dataset_size_rows', 'Dataset size in rows', ['action', 'mode', 'backend'], multiprocess_mode='livesum')
FE_MODULE_USAGE = Counter('fe_module_usage_total', 'FE module usage', ['action', 'complexity'], multiprocess_mode='livesum')

# Configuration from environment
DRIFT_NUMERIC_P95_THRESHOLD = float(os.getenv('DRIFT_NUMERIC_P95_THRESHOLD', '0.1'))
DRIFT_CATEGORICAL_PSI_THRESHOLD = float(os.getenv('DRIFT_CATEGORICAL_PSI_THRESHOLD', '0.25'))
MISSING_VALUES_THRESHOLD = float(os.getenv('MISSING_VALUES_THRESHOLD', '0.5'))
CORRELATION_THRESHOLD = float(os.getenv('CORRELATION_THRESHOLD', '0.95'))

# FE Module Integration Configuration
FE_MODULE_ENABLED = os.getenv('FE_MODULE_ENABLED', 'true').lower() == 'true'
AUTO_ADVANCED_ROUTING = os.getenv('AUTO_ADVANCED_ROUTING', 'true').lower() == 'true'
FE_CONTEXT_TTL = int(os.getenv('FE_CONTEXT_TTL', '3600'))
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')

# In-memory pipeline cache (will be replaced with Redis)
_PIPELINES: Dict[str, Dict[str, Any]] = {}

# Mode definitions for clear separation
class AgentMode(Enum):
    """Agent operation modes to prevent accidental transformations."""
    DATA_QUALITY = "data_quality"      # Read-only quality checks
    FEATURE_ENGINEERING = "feature_engineering"  # Transformative operations

# Backend types for routing
class BackendType(Enum):
    """Backend types for processing."""
    REFINERY_BASIC = "refinery_basic"  # Basic refinery agent capabilities
    FE_MODULE = "fe_module"           # Advanced FE module capabilities

# Action types with mode classification
DQ_ACTIONS = Literal[
    "check_schema_consistency",
    "check_missing_values", 
    "check_distributions",
    "check_duplicates",
    "check_leakage",
    "check_drift",
    "comprehensive_quality_report"
]

BASIC_FE_ACTIONS = Literal[
    "assign_feature_roles",
    "basic_impute_missing_values",
    "basic_scale_numeric_features",
    "basic_encode_categorical_features",
    "basic_generate_datetime_features",
    "basic_vectorise_text_features",
    "basic_generate_interactions",
    "basic_select_features",
    "save_fe_pipeline",
    "execute_feature_pipeline"
]

ADVANCED_FE_ACTIONS = Literal[
    "advanced_impute_missing_values",
    "advanced_encode_categorical_features",
    "advanced_feature_selection",
    "feature_interactions",
    "pipeline_persistence"
]

ALL_ACTIONS = Literal[DQ_ACTIONS, BASIC_FE_ACTIONS, ADVANCED_FE_ACTIONS]

# Action mode mapping for validation
ACTION_MODE_MAP = {
    # Data Quality Actions (Read-only)
    "check_schema_consistency": AgentMode.DATA_QUALITY,
    "check_missing_values": AgentMode.DATA_QUALITY,
    "check_distributions": AgentMode.DATA_QUALITY,
    "check_duplicates": AgentMode.DATA_QUALITY,
    "check_leakage": AgentMode.DATA_QUALITY,
    "check_drift": AgentMode.DATA_QUALITY,
    "comprehensive_quality_report": AgentMode.DATA_QUALITY,
    
    # Basic Feature Engineering Actions (Refinery Agent)
    "assign_feature_roles": AgentMode.FEATURE_ENGINEERING,
    "basic_impute_missing_values": AgentMode.FEATURE_ENGINEERING,
    "basic_scale_numeric_features": AgentMode.FEATURE_ENGINEERING,
    "basic_encode_categorical_features": AgentMode.FEATURE_ENGINEERING,
    "basic_generate_datetime_features": AgentMode.FEATURE_ENGINEERING,
    "basic_vectorise_text_features": AgentMode.FEATURE_ENGINEERING,
    "basic_generate_interactions": AgentMode.FEATURE_ENGINEERING,
    "basic_select_features": AgentMode.FEATURE_ENGINEERING,
    "save_fe_pipeline": AgentMode.FEATURE_ENGINEERING,
    "execute_feature_pipeline": AgentMode.FEATURE_ENGINEERING,
    
    # Advanced Feature Engineering Actions (FE Module)
    "advanced_impute_missing_values": AgentMode.FEATURE_ENGINEERING,
    "advanced_encode_categorical_features": AgentMode.FEATURE_ENGINEERING,
    "advanced_feature_selection": AgentMode.FEATURE_ENGINEERING,
    "feature_interactions": AgentMode.FEATURE_ENGINEERING,
    "pipeline_persistence": AgentMode.FEATURE_ENGINEERING
}

# Backend routing mapping
ACTION_BACKEND_MAP = {
    # Data Quality Actions (Refinery Agent)
    "check_schema_consistency": BackendType.REFINERY_BASIC,
    "check_missing_values": BackendType.REFINERY_BASIC,
    "check_distributions": BackendType.REFINERY_BASIC,
    "check_duplicates": BackendType.REFINERY_BASIC,
    "check_leakage": BackendType.REFINERY_BASIC,
    "check_drift": BackendType.REFINERY_BASIC,
    "comprehensive_quality_report": BackendType.REFINERY_BASIC,
    
    # Basic Feature Engineering Actions (Refinery Agent)
    "assign_feature_roles": BackendType.REFINERY_BASIC,
    "basic_impute_missing_values": BackendType.REFINERY_BASIC,
    "basic_scale_numeric_features": BackendType.REFINERY_BASIC,
    "basic_encode_categorical_features": BackendType.REFINERY_BASIC,
    "basic_generate_datetime_features": BackendType.REFINERY_BASIC,
    "basic_vectorise_text_features": BackendType.REFINERY_BASIC,
    "basic_generate_interactions": BackendType.REFINERY_BASIC,
    "basic_select_features": BackendType.REFINERY_BASIC,
    "save_fe_pipeline": BackendType.REFINERY_BASIC,
    "execute_feature_pipeline": BackendType.REFINERY_BASIC,
    
    # Advanced Feature Engineering Actions (FE Module)
    "advanced_impute_missing_values": BackendType.FE_MODULE,
    "advanced_encode_categorical_features": BackendType.FE_MODULE,
    "advanced_feature_selection": BackendType.FE_MODULE,
    "feature_interactions": BackendType.FE_MODULE,
    "pipeline_persistence": BackendType.FE_MODULE
}

# Pydantic models
class TaskRequest(BaseModel):
    task_id: str
    action: ALL_ACTIONS
    mode: Optional[AgentMode] = None  # Auto-detected if not provided
    backend: Optional[BackendType] = None  # Auto-detected if not provided
    params: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('mode', pre=True, always=True)
    def set_mode(cls, v, values):
        """Auto-detect mode based on action if not provided."""
        if v is None and 'action' in values:
            return ACTION_MODE_MAP.get(values['action'], AgentMode.DATA_QUALITY)
        return v
    
    @validator('backend', pre=True, always=True)
    def set_backend(cls, v, values):
        """Auto-detect backend based on action if not provided."""
        if v is None and 'action' in values:
            return ACTION_BACKEND_MAP.get(values['action'], BackendType.REFINERY_BASIC)
        return v

class TaskResponse(BaseModel):
    task_id: str
    success: bool
    mode: AgentMode
    backend: BackendType
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float
    timestamp: float

class QualityReport(BaseModel):
    """Comprehensive data quality report."""
    overall_score: float
    checks_passed: int
    checks_failed: int
    warnings: List[str]
    recommendations: List[str]
    details: Dict[str, Any]

# Unified Pipeline Context
class UnifiedPipelineContext:
    """Unified context management for both refinery and FE module."""
    
    def __init__(self, run_id: str, redis_client: Optional[redis.Redis] = None):
        self.run_id = run_id
        self.refinery_context = {
            "steps": [],
            "features": [],
            "metadata": {},
            "mode": None,
            "created_at": time.time()
        }
        self.fe_context = None
        self.redis_client = redis_client
        self.fe_context_manager = None
        
        if redis_client and FE_MODULE_AVAILABLE:
            self.fe_context_manager = PipelineContextManager(redis_client)
    
    async def get_or_create_fe_context(self) -> Optional[PipelineCtx]:
        """Get or create FE module context."""
        if not FE_MODULE_AVAILABLE or not self.fe_context_manager:
            return None
            
        if self.fe_context is None:
            self.fe_context = await self.fe_context_manager.get_ctx(self.run_id)
        return self.fe_context
    
    async def save_contexts(self):
        """Save both contexts."""
        if self.fe_context and self.fe_context_manager:
            await self.fe_context_manager.save_ctx(self.run_id, self.fe_context, FE_CONTEXT_TTL)
    
    def update_refinery_context(self, key: str, value: Any):
        """Update refinery context."""
        self.refinery_context[key] = value

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

def get_unified_pipeline_context(run_id: str, redis_client: Optional[redis.Redis] = None) -> UnifiedPipelineContext:
    """Get or create unified pipeline context."""
    key = f"{run_id}:unified"
    if key not in _PIPELINES:
        _PIPELINES[key] = UnifiedPipelineContext(run_id, redis_client)
        ACTIVE_PIPELINES.inc()
    return _PIPELINES[key]

def validate_mode_consistency(action: str, mode: AgentMode) -> None:
    """Validate that action and mode are consistent."""
    expected_mode = ACTION_MODE_MAP.get(action)
    if expected_mode != mode:
        raise ValueError(f"Action '{action}' requires mode '{expected_mode.value}', but got '{mode.value}'")

def requires_advanced_processing(action: str, params: Dict[str, Any]) -> bool:
    """Determine if advanced processing is needed."""
    if not FE_MODULE_AVAILABLE or not FE_MODULE_ENABLED:
        return False
    
    # Check for complexity indicators
    complexity_indicators = {
        "high_cardinality": params.get("cardinality", 0) > 50,
        "missing_pattern_analysis": params.get("pattern_analysis", False),
        "target_encoding": params.get("encoding_strategy") == "target",
        "multicollinearity_detection": params.get("vif_analysis", False),
        "cross_validation": params.get("cv_folds", 0) > 0,
        "advanced_imputation": params.get("imputation_strategy") in ["knn", "mice"],
        "hash_encoding": params.get("encoding_strategy") == "hash",
        "feature_interactions": params.get("interaction_degree", 0) > 2
    }
    
    return any(complexity_indicators.values())

# Data Quality Handlers (Read-only operations)
async def check_schema_consistency(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check schema consistency against expected schema."""
    data_path = params["data_path"]
    expected_schema = params.get("expected_schema", {})
    
    df = load_data(data_path)
    DATASET_SIZE.labels(action="check_schema_consistency", mode="data_quality", backend="refinery_basic").observe(len(df))
    
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
        "actual_schema": actual_schema,
        "missing_columns": [],
        "extra_columns": [],
        "dtype_mismatches": []
    }

async def check_missing_values(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check missing values without modifying data."""
    data_path = params["data_path"]
    threshold = params.get("threshold", MISSING_VALUES_THRESHOLD)
    
    df = load_data(data_path)
    DATASET_SIZE.labels(action="check_missing_values", mode="data_quality", backend="refinery_basic").observe(len(df))
    
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df) * 100).round(2)
    
    problematic_cols = missing_percentages[missing_percentages > threshold * 100].to_dict()
    
    return {
        "total_missing": int(missing_counts.sum()),
        "missing_percentages": missing_percentages.to_dict(),
        "problematic_columns": problematic_cols,
        "recommendations": [
            f"Column '{col}' has {pct}% missing values" 
            for col, pct in problematic_cols.items()
        ]
    }

async def check_distributions(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check data distributions without modifying data."""
    data_path = params["data_path"]
    
    df = load_data(data_path)
    DATASET_SIZE.labels(action="check_distributions", mode="data_quality", backend="refinery_basic").observe(len(df))
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    distribution_report = {
        "numeric_distributions": {},
        "categorical_distributions": {},
        "outliers_detected": {},
        "recommendations": []
    }
    
    # Analyze numeric distributions
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            distribution_report["numeric_distributions"][col] = {
                "mean": float(col_data.mean()),
                "std": float(col_data.std()),
                "skewness": float(col_data.skew()),
                "kurtosis": float(col_data.kurtosis()),
                "q25": float(col_data.quantile(0.25)),
                "q75": float(col_data.quantile(0.75))
            }
            
            # Detect outliers using IQR
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = col_data[(col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)]
            
            if len(outliers) > 0:
                distribution_report["outliers_detected"][col] = {
                    "count": int(len(outliers)),
                    "percentage": float(len(outliers) / len(col_data) * 100)
                }
    
    # Analyze categorical distributions
    for col in categorical_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            value_counts = col_data.value_counts()
            distribution_report["categorical_distributions"][col] = {
                "unique_count": int(len(value_counts)),
                "most_common": value_counts.head(5).to_dict(),
                "cardinality": "high" if len(value_counts) > 50 else "medium" if len(value_counts) > 10 else "low"
            }
    
    return distribution_report

async def check_duplicates(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check for duplicate records without modifying data."""
    data_path = params["data_path"]
    subset = params.get("subset", None)  # Columns to check for duplicates
    
    df = load_data(data_path)
    DATASET_SIZE.labels(action="check_duplicates", mode="data_quality", backend="refinery_basic").observe(len(df))
    
    if subset:
        duplicates = df.duplicated(subset=subset)
    else:
        duplicates = df.duplicated()
    
    duplicate_count = int(duplicates.sum())
    duplicate_percentage = float(duplicate_count / len(df) * 100)
    
    return {
        "duplicate_count": duplicate_count,
        "duplicate_percentage": duplicate_percentage,
        "subset_checked": subset or "all_columns",
        "recommendations": [
            f"Found {duplicate_count} duplicate records ({duplicate_percentage:.2f}%)"
        ] if duplicate_count > 0 else ["No duplicates found"]
    }

async def check_leakage(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check for data leakage without modifying data."""
    data_path = params["data_path"]
    target_col = params.get("target_col")
    
    df = load_data(data_path)
    DATASET_SIZE.labels(action="check_leakage", mode="data_quality", backend="refinery_basic").observe(len(df))
    
    if not target_col or target_col not in df.columns:
        return {
            "status": "error",
            "message": "Target column not specified or not found"
        }
    
    # Check for perfect correlation with target
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    leakage_candidates = []
    
    for col in numeric_cols:
        if col != target_col:
            correlation = abs(df[col].corr(df[target_col]))
            if correlation > 0.95:  # Very high correlation might indicate leakage
                leakage_candidates.append({
                    "column": col,
                    "correlation": float(correlation)
                })
    
    return {
        "target_column": target_col,
        "leakage_candidates": leakage_candidates,
        "recommendations": [
            f"Column '{c['column']}' has {c['correlation']:.3f} correlation with target - potential leakage"
            for c in leakage_candidates
        ] if leakage_candidates else ["No obvious data leakage detected"]
    }

async def check_drift(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check for data drift without modifying data."""
    current_data_path = params["current_data_path"]
    reference_data_path = params.get("reference_data_path")
    
    current_df = load_data(current_data_path)
    DATASET_SIZE.labels(action="check_drift", mode="data_quality", backend="refinery_basic").observe(len(current_df))
    
    if not reference_data_path:
        return {
            "status": "error",
            "message": "Reference data path not provided"
        }
    
    reference_df = load_data(reference_data_path)
    
    # Simple drift detection using basic statistics
    numeric_cols = current_df.select_dtypes(include=[np.number]).columns
    drift_report = {
        "drift_detected": False,
        "drift_details": {},
        "recommendations": []
    }
    
    for col in numeric_cols:
        if col in reference_df.columns:
            current_mean = current_df[col].mean()
            reference_mean = reference_df[col].mean()
            drift_ratio = abs(current_mean - reference_mean) / abs(reference_mean) if reference_mean != 0 else 0
            
            if drift_ratio > DRIFT_NUMERIC_P95_THRESHOLD:
                drift_report["drift_detected"] = True
                drift_report["drift_details"][col] = {
                    "current_mean": float(current_mean),
                    "reference_mean": float(reference_mean),
                    "drift_ratio": float(drift_ratio)
                }
                drift_report["recommendations"].append(
                    f"Column '{col}' shows {drift_ratio:.2%} drift from reference"
                )
    
    return drift_report

async def comprehensive_quality_report(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive data quality report without modifying data."""
    data_path = params["data_path"]
    target_col = params.get("target_col")
    
    df = load_data(data_path)
    DATASET_SIZE.labels(action="comprehensive_quality_report", mode="data_quality", backend="refinery_basic").observe(len(df))
    
    # Run all quality checks
    schema_result = await check_schema_consistency({"data_path": data_path})
    missing_result = await check_missing_values({"data_path": data_path})
    distribution_result = await check_distributions({"data_path": data_path})
    duplicate_result = await check_duplicates({"data_path": data_path})
    
    # Calculate overall quality score
    checks_passed = 0
    checks_failed = 0
    warnings = []
    recommendations = []
    
    # Schema check
    if schema_result["status"] == "pass":
        checks_passed += 1
    else:
        checks_failed += 1
        warnings.append("Schema inconsistencies detected")
    
    # Missing values check
    if not missing_result["problematic_columns"]:
        checks_passed += 1
    else:
        checks_failed += 1
        warnings.append(f"High missing values in {len(missing_result['problematic_columns'])} columns")
    
    # Duplicate check
    if duplicate_result["duplicate_count"] == 0:
        checks_passed += 1
    else:
        checks_failed += 1
        warnings.append(f"{duplicate_result['duplicate_count']} duplicate records found")
    
    # Distribution check
    if not distribution_result["outliers_detected"]:
        checks_passed += 1
    else:
        checks_failed += 1
        warnings.append(f"Outliers detected in {len(distribution_result['outliers_detected'])} columns")
    
    total_checks = checks_passed + checks_failed
    overall_score = (checks_passed / total_checks * 100) if total_checks > 0 else 100
    
    return {
        "overall_score": float(overall_score),
        "checks_passed": checks_passed,
        "checks_failed": checks_failed,
        "warnings": warnings,
        "recommendations": recommendations,
        "details": {
            "schema": schema_result,
            "missing_values": missing_result,
            "distributions": distribution_result,
            "duplicates": duplicate_result
        }
    }

# Basic Feature Engineering Handlers (Refinery Agent)
async def assign_feature_roles(params: Dict[str, Any]) -> Dict[str, Any]:
    """Assign feature roles for feature engineering pipeline."""
    data_path = params["data_path"]
    target_col = params.get("target_col")
    
    df = load_data(data_path)
    DATASET_SIZE.labels(action="assign_feature_roles", mode="feature_engineering", backend="refinery_basic").observe(len(df))
    
    run_id = params["run_id"]
    redis_client = params.get("redis_client")
    
    context = get_unified_pipeline_context(run_id, redis_client)
    context.update_refinery_context("mode", AgentMode.FEATURE_ENGINEERING.value)
    
    # Auto-detect feature roles
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    feature_roles = {
        "target": target_col,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "datetime_features": datetime_cols,
        "text_features": [],  # Would need NLP detection
        "id_features": []     # Would need ID detection
    }
    
    context.update_refinery_context("feature_roles", feature_roles)
    context.refinery_context["steps"].append("assign_feature_roles")
    
    return {
        "feature_roles": feature_roles,
        "total_features": len(numeric_cols) + len(categorical_cols) + len(datetime_cols)
    }

async def basic_impute_missing_values(params: Dict[str, Any]) -> Dict[str, Any]:
    """Basic imputation using simple strategies."""
    data_path = params["data_path"]
    run_id = params["run_id"]
    strategy = params.get("strategy", "auto")
    
    df = load_data(data_path)
    DATASET_SIZE.labels(action="basic_impute_missing_values", mode="feature_engineering", backend="refinery_basic").observe(len(df))
    
    redis_client = params.get("redis_client")
    context = get_unified_pipeline_context(run_id, redis_client)
    context.update_refinery_context("mode", AgentMode.FEATURE_ENGINEERING.value)
    
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
    
    context.update_refinery_context("imputation_strategy", imputation_summary)
    context.refinery_context["steps"].append("basic_impute_missing_values")
    
    return {
        "imputed_columns": list(imputation_summary.keys()),
        "imputation_strategy": imputation_summary,
        "remaining_nulls": df_imputed.isnull().sum().sum()
    }

# Advanced Feature Engineering Handlers (FE Module Integration)
async def advanced_impute_missing_values(params: Dict[str, Any]) -> Dict[str, Any]:
    """Advanced imputation using FE module capabilities."""
    if not FE_MODULE_AVAILABLE:
        raise ValueError("FE module not available for advanced imputation")
    
    data_path = params["data_path"]
    run_id = params["run_id"]
    
    df = load_data(data_path)
    DATASET_SIZE.labels(action="advanced_impute_missing_values", mode="feature_engineering", backend="fe_module").observe(len(df))
    
    redis_client = params.get("redis_client")
    context = get_unified_pipeline_context(run_id, redis_client)
    
    # Get or create FE context
    fe_context = await context.get_or_create_fe_context()
    if fe_context is None:
        raise ValueError("Failed to create FE context")
    
    # Set data in FE context
    fe_context.df = df
    
    # Use FE module's advanced imputation
    result = await fe_impute(params, fe_context, {})
    
    # Update refinery context
    context.update_refinery_context("advanced_imputation", result)
    context.refinery_context["steps"].append("advanced_impute_missing_values")
    
    # Save contexts
    await context.save_contexts()
    
    # Record FE module usage
    FE_MODULE_USAGE.labels(action="advanced_impute_missing_values", complexity="high").inc()
    
    return result

async def advanced_encode_categorical_features(params: Dict[str, Any]) -> Dict[str, Any]:
    """Advanced categorical encoding using FE module capabilities."""
    if not FE_MODULE_AVAILABLE:
        raise ValueError("FE module not available for advanced encoding")
    
    data_path = params["data_path"]
    run_id = params["run_id"]
    
    df = load_data(data_path)
    DATASET_SIZE.labels(action="advanced_encode_categorical_features", mode="feature_engineering", backend="fe_module").observe(len(df))
    
    redis_client = params.get("redis_client")
    context = get_unified_pipeline_context(run_id, redis_client)
    
    # Get or create FE context
    fe_context = await context.get_or_create_fe_context()
    if fe_context is None:
        raise ValueError("Failed to create FE context")
    
    # Set data in FE context
    fe_context.df = df
    
    # Use FE module's advanced encoding
    result = await fe_encode(params, fe_context, {})
    
    # Update refinery context
    context.update_refinery_context("advanced_encoding", result)
    context.refinery_context["steps"].append("advanced_encode_categorical_features")
    
    # Save contexts
    await context.save_contexts()
    
    # Record FE module usage
    FE_MODULE_USAGE.labels(action="advanced_encode_categorical_features", complexity="high").inc()
    
    return result

async def advanced_feature_selection(params: Dict[str, Any]) -> Dict[str, Any]:
    """Advanced feature selection using FE module capabilities."""
    if not FE_MODULE_AVAILABLE:
        raise ValueError("FE module not available for advanced feature selection")
    
    data_path = params["data_path"]
    run_id = params["run_id"]
    
    df = load_data(data_path)
    DATASET_SIZE.labels(action="advanced_feature_selection", mode="feature_engineering", backend="fe_module").observe(len(df))
    
    redis_client = params.get("redis_client")
    context = get_unified_pipeline_context(run_id, redis_client)
    
    # Get or create FE context
    fe_context = await context.get_or_create_fe_context()
    if fe_context is None:
        raise ValueError("Failed to create FE context")
    
    # Set data in FE context
    fe_context.df = df
    
    # Use FE module's advanced feature selection
    result = await fe_prune(params, fe_context, {})
    
    # Update refinery context
    context.update_refinery_context("advanced_feature_selection", result)
    context.refinery_context["steps"].append("advanced_feature_selection")
    
    # Save contexts
    await context.save_contexts()
    
    # Record FE module usage
    FE_MODULE_USAGE.labels(action="advanced_feature_selection", complexity="high").inc()
    
    return result

async def feature_interactions(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate feature interactions using FE module capabilities."""
    if not FE_MODULE_AVAILABLE:
        raise ValueError("FE module not available for feature interactions")
    
    data_path = params["data_path"]
    run_id = params["run_id"]
    
    df = load_data(data_path)
    DATASET_SIZE.labels(action="feature_interactions", mode="feature_engineering", backend="fe_module").observe(len(df))
    
    redis_client = params.get("redis_client")
    context = get_unified_pipeline_context(run_id, redis_client)
    
    # Get or create FE context
    fe_context = await context.get_or_create_fe_context()
    if fe_context is None:
        raise ValueError("Failed to create FE context")
    
    # Set data in FE context
    fe_context.df = df
    
    # Use FE module's feature interactions
    result = await fe_interactions(params, fe_context)
    
    # Update refinery context
    context.update_refinery_context("feature_interactions", result)
    context.refinery_context["steps"].append("feature_interactions")
    
    # Save contexts
    await context.save_contexts()
    
    # Record FE module usage
    FE_MODULE_USAGE.labels(action="feature_interactions", complexity="medium").inc()
    
    return result

# Main execution endpoint
@app.post("/execute", response_model=TaskResponse)
async def execute(req: TaskRequest):
    start_time = time.time()
    
    try:
        # Validate mode consistency
        validate_mode_consistency(req.action, req.mode)
        
        # Determine if advanced processing is needed
        if AUTO_ADVANCED_ROUTING and requires_advanced_processing(req.action, req.params):
            req.backend = BackendType.FE_MODULE
            logger.info(f"Auto-routing to FE module for action: {req.action}")
        
        # Action dispatcher
        dispatcher = {
            # Data Quality actions (Read-only)
            "check_schema_consistency": check_schema_consistency,
            "check_missing_values": check_missing_values,
            "check_distributions": check_distributions,
            "check_duplicates": check_duplicates,
            "check_leakage": check_leakage,
            "check_drift": check_drift,
            "comprehensive_quality_report": comprehensive_quality_report,
            
            # Basic Feature Engineering actions (Refinery Agent)
            "assign_feature_roles": assign_feature_roles,
            "basic_impute_missing_values": basic_impute_missing_values,
            
            # Advanced Feature Engineering actions (FE Module)
            "advanced_impute_missing_values": advanced_impute_missing_values,
            "advanced_encode_categorical_features": advanced_encode_categorical_features,
            "advanced_feature_selection": advanced_feature_selection,
            "feature_interactions": feature_interactions,
        }
        
        if req.action not in dispatcher:
            raise ValueError(f"Unsupported action: {req.action}")
        
        # Add Redis client to params for context management
        req.params["redis_client"] = redis.from_url(REDIS_URL) if REDIS_URL else None
        
        # Execute the action
        handler = dispatcher[req.action]
        result = await handler(req.params)
        
        execution_time = time.time() - start_time
        
        # Record metrics with backend
        REQUEST_COUNT.labels(action=req.action, mode=req.mode.value, backend=req.backend.value, status="success").inc()
        REQUEST_DURATION.labels(action=req.action, mode=req.mode.value, backend=req.backend.value).observe(execution_time)
        
        return TaskResponse(
            task_id=req.task_id,
            success=True,
            mode=req.mode,
            backend=req.backend,
            result=result,
            execution_time=execution_time,
            timestamp=time.time()
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        
        # Record error metrics with backend
        REQUEST_COUNT.labels(action=req.action, mode=req.mode.value, backend=req.backend.value, status="error").inc()
        REQUEST_DURATION.labels(action=req.action, mode=req.mode.value, backend=req.backend.value).observe(execution_time)
        
        logger.error(f"Task {req.task_id} failed: {str(e)}")
        
        return TaskResponse(
            task_id=req.task_id,
            success=False,
            mode=req.mode,
            backend=req.backend,
            error=str(e),
            execution_time=execution_time,
            timestamp=time.time()
        )

# Health endpoint
@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "version": "2.0.0", 
        "modes": [mode.value for mode in AgentMode],
        "backends": [backend.value for backend in BackendType],
        "fe_module_available": FE_MODULE_AVAILABLE,
        "fe_module_enabled": FE_MODULE_ENABLED
    }

# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Pipeline status endpoint
@app.get("/pipelines")
async def list_pipelines():
    return {
        "active_pipelines": len(_PIPELINES),
        "pipelines": list(_PIPELINES.keys()),
        "modes": {
            "data_quality": len([p for p in _PIPELINES.values() if hasattr(p, 'refinery_context') and p.refinery_context.get("mode") == AgentMode.DATA_QUALITY.value]),
            "feature_engineering": len([p for p in _PIPELINES.values() if hasattr(p, 'refinery_context') and p.refinery_context.get("mode") == AgentMode.FEATURE_ENGINEERING.value])
        },
        "fe_module_available": FE_MODULE_AVAILABLE,
        "fe_module_enabled": FE_MODULE_ENABLED
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005) 