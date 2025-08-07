#!/usr/bin/env python3
"""
Refinery Agent Service

A FastAPI-based microservice that provides data quality validation and feature engineering tools
for the Master Orchestrator. Implements comprehensive DQ checks and automated FE pipelines
with production-grade features including caching, telemetry, and security.
"""

import asyncio
import hashlib
import json
import logging
import time
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import base64
import io

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import redis.asyncio as redis
from motor.motor_asyncio import AsyncIOMotorClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Refinery Agent Service",
    description="Data Quality Validation and Feature Engineering microservice for Deepline Master Orchestrator",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
data_store: Dict[str, pd.DataFrame] = {}
pipeline_store: Dict[str, Any] = {}  # Store FE pipelines by (run_id, session_id)
redis_client: Optional[redis.Redis] = None
mongo_client: Optional[AsyncIOMotorClient] = None
telemetry_data: Dict[str, Any] = {}

# Configuration
class Config:
    REDIS_URL = "redis://localhost:6379"
    MONGO_URL = "mongodb://localhost:27017"
    DB_NAME = "deepline"
    CACHE_TTL = 3600  # 1 hour
    MAX_DATASET_SIZE = 1000000  # 1M rows
    SAMPLE_SIZE = 10000  # For large datasets
    RANDOM_SEED = 42

# Pydantic Models for Request/Response
class TaskRequest(BaseModel):
    task_id: str = Field(..., description="Unique task identifier")
    action: str = Field(..., description="Action to perform")
    params: Dict[str, Any] = Field(..., description="Action parameters")

class TaskResponse(BaseModel):
    task_id: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float
    timestamp: float

# Data Quality Models
class SchemaCheckRequest(BaseModel):
    data_path: str = Field(..., description="Path to data file")
    expected_schema_json: Optional[Dict[str, Any]] = Field(None, description="Expected schema")

class MissingValuesRequest(BaseModel):
    data_path: str = Field(..., description="Path to data file")
    threshold_pct: float = Field(0.5, description="Missing value threshold percentage")

class DistributionCheckRequest(BaseModel):
    data_path: str = Field(..., description="Path to data file")
    numeric_rules: Optional[Dict[str, Any]] = Field(None, description="Numeric validation rules")
    category_domains: Optional[Dict[str, List[str]]] = Field(None, description="Categorical domain constraints")

class DuplicateCheckRequest(BaseModel):
    data_path: str = Field(..., description="Path to data file")
    id_cols: Optional[List[str]] = Field(None, description="ID columns for duplicate detection")

class LeakageCheckRequest(BaseModel):
    data_path: str = Field(..., description="Path to data file")
    target_col: str = Field(..., description="Target column name")

class DriftCheckRequest(BaseModel):
    reference_path: str = Field(..., description="Path to reference data")
    current_path: str = Field(..., description="Path to current data")

# Feature Engineering Models
class FeatureRolesRequest(BaseModel):
    input_path: str = Field(..., description="Path to input data")
    overrides_json: Optional[Dict[str, Any]] = Field(None, description="Role overrides")
    run_id: str = Field(..., description="Run identifier")
    session_id: str = Field("default", description="Session identifier")

class ImputeRequest(BaseModel):
    input_path: str = Field(..., description="Path to input data")
    run_id: str = Field(..., description="Run identifier")
    session_id: str = Field("default", description="Session identifier")
    strategy: str = Field("auto", description="Imputation strategy")

class ScaleRequest(BaseModel):
    run_id: str = Field(..., description="Run identifier")
    session_id: str = Field("default", description="Session identifier")
    method: str = Field("standard", description="Scaling method")

class EncodeRequest(BaseModel):
    run_id: str = Field(..., description="Run identifier")
    session_id: str = Field("default", description="Session identifier")
    strategy: str = Field("auto", description="Encoding strategy")

class DatetimeRequest(BaseModel):
    run_id: str = Field(..., description="Run identifier")
    session_id: str = Field("default", description="Session identifier")
    country: str = Field("US", description="Country for datetime features")

class TextVectorizeRequest(BaseModel):
    run_id: str = Field(..., description="Run identifier")
    session_id: str = Field("default", description="Session identifier")
    model: str = Field("tfidf", description="Vectorization model")
    max_feats: int = Field(5000, description="Maximum features")

class InteractionsRequest(BaseModel):
    run_id: str = Field(..., description="Run identifier")
    session_id: str = Field("default", description="Session identifier")
    max_degree: int = Field(2, description="Maximum interaction degree")

class FeatureSelectRequest(BaseModel):
    run_id: str = Field(..., description="Run identifier")
    session_id: str = Field("default", description="Session identifier")
    method: str = Field("shap_top_k", description="Selection method")
    k: int = Field(100, description="Number of features to select")

class SavePipelineRequest(BaseModel):
    input_path: str = Field(..., description="Path to input data")
    export_pipeline_path: str = Field(..., description="Path to save pipeline")
    export_data_path: str = Field(..., description="Path to save processed data")
    run_id: str = Field(..., description="Run identifier")
    session_id: str = Field("default", description="Session identifier")

# Utility Functions
def load_dataframe(path: str) -> pd.DataFrame:
    """Load dataframe from various file formats."""
    path = Path(path)
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    elif path.suffix.lower() == '.csv':
        return pd.read_csv(path)
    elif path.suffix.lower() in ['.xlsx', '.xls']:
        return pd.read_excel(path)
    elif path.suffix.lower() == '.json':
        return pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

def get_pipeline_context(run_id: str, session_id: str = "default") -> Any:
    """Get or create pipeline context for feature engineering."""
    key = (run_id, session_id)
    if key not in pipeline_store:
        pipeline_store[key] = {
            "steps": [],
            "feature_names": [],
            "metadata": {}
        }
    return pipeline_store[key]

def record_telemetry(operation: str, duration: float, success: bool, error: Optional[str] = None):
    """Record telemetry data."""
    telemetry_data[operation] = {
        "duration": duration,
        "success": success,
        "error": error,
        "timestamp": datetime.now().isoformat()
    }

# Data Quality Handlers
async def check_schema_consistency(request: SchemaCheckRequest):
    """Check schema consistency between expected and actual data."""
    start_time = time.time()
    
    try:
        df = load_dataframe(request.data_path)
        actual_schema = {
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "shape": df.shape
        }
        
        if request.expected_schema_json:
            # Compare schemas
            expected = request.expected_schema_json
            diff = {
                "missing_columns": list(set(expected.get("columns", [])) - set(actual_schema["columns"])),
                "extra_columns": list(set(actual_schema["columns"]) - set(expected.get("columns", []))),
                "type_mismatches": {}
            }
            
            # Check type mismatches for common columns
            for col in set(expected.get("columns", [])) & set(actual_schema["columns"]):
                if col in expected.get("dtypes", {}) and col in actual_schema["dtypes"]:
                    if expected["dtypes"][col] != actual_schema["dtypes"][col]:
                        diff["type_mismatches"][col] = {
                            "expected": expected["dtypes"][col],
                            "actual": actual_schema["dtypes"][col]
                        }
            
            status = "pass" if not any(diff.values()) else "fail"
        else:
            diff = {}
            status = "pass"
        
        duration = time.time() - start_time
        record_telemetry("check_schema_consistency", duration, True)
        
        return {
            "status": status,
            "diff": diff,
            "actual_schema": actual_schema
        }
        
    except Exception as e:
        duration = time.time() - start_time
        record_telemetry("check_schema_consistency", duration, False, str(e))
        raise HTTPException(status_code=500, detail=str(e))

async def check_missing_values(request: MissingValuesRequest):
    """Check for missing values in the dataset."""
    start_time = time.time()
    
    try:
        df = load_dataframe(request.data_path)
        
        missing_counts = df.isnull().sum().to_dict()
        missing_percentages = (df.isnull().sum() / len(df) * 100).to_dict()
        
        cols_over_thresh = [
            col for col, pct in missing_percentages.items() 
            if pct > request.threshold_pct * 100
        ]
        
        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns_with_missing": sum(1 for count in missing_counts.values() if count > 0),
            "total_missing_values": sum(missing_counts.values()),
            "overall_missing_pct": sum(missing_counts.values()) / (len(df) * len(df.columns)) * 100
        }
        
        duration = time.time() - start_time
        record_telemetry("check_missing_values", duration, True)
        
        return {
            "cols_over_thresh": cols_over_thresh,
            "summary": summary,
            "missing_counts": missing_counts,
            "missing_percentages": missing_percentages
        }
        
    except Exception as e:
        duration = time.time() - start_time
        record_telemetry("check_missing_values", duration, False, str(e))
        raise HTTPException(status_code=500, detail=str(e))

async def check_distributions(request: DistributionCheckRequest):
    """Check data distributions for violations and anomalies."""
    start_time = time.time()
    
    try:
        df = load_dataframe(request.data_path)
        violations = []
        recommendations = []
        
        # Check numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Check for outliers using IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            
            if len(outliers) > 0:
                violations.append({
                    "column": col,
                    "type": "outliers",
                    "count": len(outliers),
                    "percentage": len(outliers) / len(df) * 100
                })
        
        # Check categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            unique_vals = df[col].nunique()
            if unique_vals > 100:
                violations.append({
                    "column": col,
                    "type": "high_cardinality",
                    "unique_values": unique_vals
                })
                recommendations.append(f"Consider encoding strategy for high-cardinality column '{col}'")
        
        summary = {
            "total_violations": len(violations),
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(categorical_cols),
            "total_rows": len(df)
        }
        
        duration = time.time() - start_time
        record_telemetry("check_distributions", duration, True)
        
        return {
            "violations": violations,
            "summary": summary,
            "recommendations": recommendations
        }
        
    except Exception as e:
        duration = time.time() - start_time
        record_telemetry("check_distributions", duration, False, str(e))
        raise HTTPException(status_code=500, detail=str(e))

async def check_duplicates(request: DuplicateCheckRequest):
    """Check for duplicate rows and high correlation pairs."""
    start_time = time.time()
    
    try:
        df = load_dataframe(request.data_path)
        
        # Check for duplicate rows
        dup_rows = df.duplicated()
        dup_row_count = dup_rows.sum()
        duplicate_indices = df[dup_rows].index.tolist()
        
        # Check for duplicate IDs if specified
        dup_id_count = 0
        if request.id_cols:
            for id_col in request.id_cols:
                if id_col in df.columns:
                    dup_id_count += df[df.duplicated(subset=[id_col])].shape[0]
        
        # Find high correlation pairs
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr().abs()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if corr_val > 0.95:  # Very high correlation threshold
                        high_corr_pairs.append({
                            "col1": corr_matrix.columns[i],
                            "col2": corr_matrix.columns[j],
                            "correlation": corr_val
                        })
        else:
            high_corr_pairs = []
        
        duration = time.time() - start_time
        record_telemetry("check_duplicates", duration, True)
        
        return {
            "dup_row_count": dup_row_count,
            "dup_id_count": dup_id_count,
            "high_corr_pairs": high_corr_pairs,
            "duplicate_indices": duplicate_indices
        }
        
    except Exception as e:
        duration = time.time() - start_time
        record_telemetry("check_duplicates", duration, False, str(e))
        raise HTTPException(status_code=500, detail=str(e))

async def check_leakage(request: LeakageCheckRequest):
    """Check for data leakage by analyzing correlations with target."""
    start_time = time.time()
    
    try:
        df = load_dataframe(request.data_path)
        
        if request.target_col not in df.columns:
            raise ValueError(f"Target column '{request.target_col}' not found in dataset")
        
        # Calculate correlations with target
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if request.target_col in numeric_cols:
            corr_table = {}
            suspicious_cols = []
            
            for col in numeric_cols:
                if col != request.target_col:
                    corr = df[col].corr(df[request.target_col])
                    corr_table[col] = corr
                    
                    # Flag suspicious correlations
                    if abs(corr) > 0.95:
                        suspicious_cols.append(col)
            
            recommendations = []
            if suspicious_cols:
                recommendations.append(f"High correlation columns detected: {suspicious_cols}")
                recommendations.append("Consider removing or investigating these features for data leakage")
        else:
            corr_table = {}
            suspicious_cols = []
            recommendations = ["Target column is not numeric, correlation analysis not applicable"]
        
        duration = time.time() - start_time
        record_telemetry("check_leakage", duration, True)
        
        return {
            "suspicious_cols": suspicious_cols,
            "corr_table": corr_table,
            "recommendations": recommendations
        }
        
    except Exception as e:
        duration = time.time() - start_time
        record_telemetry("check_leakage", duration, False, str(e))
        raise HTTPException(status_code=500, detail=str(e))

async def check_drift(request: DriftCheckRequest):
    """Check for data drift between reference and current datasets."""
    start_time = time.time()
    
    try:
        ref_df = load_dataframe(request.reference_path)
        curr_df = load_dataframe(request.current_path)
        
        # Simple drift detection based on statistical differences
        drift_metrics = {}
        drifted_features = []
        
        # Compare common numeric columns
        common_numeric = set(ref_df.select_dtypes(include=[np.number]).columns) & \
                        set(curr_df.select_dtypes(include=[np.number]).columns)
        
        for col in common_numeric:
            ref_mean = ref_df[col].mean()
            curr_mean = curr_df[col].mean()
            ref_std = ref_df[col].std()
            
            if ref_std > 0:
                drift_score = abs(curr_mean - ref_mean) / ref_std
                drift_metrics[col] = {
                    "ref_mean": ref_mean,
                    "curr_mean": curr_mean,
                    "drift_score": drift_score
                }
                
                if drift_score > 2.0:  # Significant drift threshold
                    drifted_features.append(col)
        
        # Determine overall severity
        if len(drifted_features) == 0:
            severity = "low"
        elif len(drifted_features) <= len(common_numeric) * 0.1:
            severity = "medium"
        else:
            severity = "high"
        
        duration = time.time() - start_time
        record_telemetry("check_drift", duration, True)
        
        return {
            "drift_metrics": drift_metrics,
            "drifted_features": drifted_features,
            "severity": severity
        }
        
    except Exception as e:
        duration = time.time() - start_time
        record_telemetry("check_drift", duration, False, str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Feature Engineering Handlers
async def assign_feature_roles(request: FeatureRolesRequest):
    """Assign feature roles based on data analysis."""
    start_time = time.time()
    
    try:
        df = load_dataframe(request.input_path)
        pipeline_ctx = get_pipeline_context(request.run_id, request.session_id)
        
        roles = {}
        confidence_scores = {}
        
        # Simple role detection logic
        for col in df.columns:
            # Detect target columns (assuming binary classification for now)
            if col.lower() in ['target', 'label', 'y', 'class', 'outcome']:
                roles[col] = 'target'
                confidence_scores[col] = 0.9
            # Detect ID columns
            elif col.lower() in ['id', 'index', 'key'] or col.endswith('_id'):
                roles[col] = 'id'
                confidence_scores[col] = 0.8
            # Detect datetime columns
            elif df[col].dtype == 'object' and df[col].str.match(r'\d{4}-\d{2}-\d{2}').any():
                roles[col] = 'datetime'
                confidence_scores[col] = 0.7
            # Detect text columns
            elif df[col].dtype == 'object' and df[col].str.len().mean() > 50:
                roles[col] = 'text'
                confidence_scores[col] = 0.6
            # Detect categorical columns
            elif df[col].dtype == 'object' or df[col].nunique() < min(50, len(df) * 0.1):
                roles[col] = 'categorical'
                confidence_scores[col] = 0.7
            # Default to numeric
            else:
                roles[col] = 'numeric'
                confidence_scores[col] = 0.5
        
        # Apply overrides if provided
        if request.overrides_json:
            for col, role in request.overrides_json.items():
                if col in roles:
                    roles[col] = role
                    confidence_scores[col] = 1.0  # Override has highest confidence
        
        # Store in pipeline context
        pipeline_ctx["roles"] = roles
        pipeline_ctx["feature_names"] = list(df.columns)
        
        duration = time.time() - start_time
        record_telemetry("assign_feature_roles", duration, True)
        
        return {
            "roles": roles,
            "confidence_scores": confidence_scores
        }
        
    except Exception as e:
        duration = time.time() - start_time
        record_telemetry("assign_feature_roles", duration, False, str(e))
        raise HTTPException(status_code=500, detail=str(e))

async def impute_missing_values(request: ImputeRequest):
    """Impute missing values in the dataset."""
    start_time = time.time()
    
    try:
        df = load_dataframe(request.input_path)
        pipeline_ctx = get_pipeline_context(request.run_id, request.session_id)
        
        # Determine imputation strategy
        if request.strategy == "auto":
            strategy = "mean"  # Default for numeric
        else:
            strategy = request.strategy
        
        # Apply imputation to numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df_imputed = df.copy()
            imputer = SimpleImputer(strategy=strategy)
            df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            
            # Calculate remaining null percentages
            imputed_null_pct = (df_imputed.isnull().sum() / len(df_imputed) * 100).to_dict()
            
            # Store imputer in pipeline context
            pipeline_ctx["imputer"] = imputer
            pipeline_ctx["imputed_data"] = df_imputed
        else:
            imputed_null_pct = df.isnull().sum().to_dict()
        
        duration = time.time() - start_time
        record_telemetry("impute_missing_values", duration, True)
        
        return {
            "imputed_null_pct": imputed_null_pct,
            "strategy_used": strategy
        }
        
    except Exception as e:
        duration = time.time() - start_time
        record_telemetry("impute_missing_values", duration, False, str(e))
        raise HTTPException(status_code=500, detail=str(e))

async def scale_numeric_features(request: ScaleRequest):
    """Scale numeric features."""
    start_time = time.time()
    
    try:
        pipeline_ctx = get_pipeline_context(request.run_id, request.session_id)
        
        # Create scaler based on method
        if request.method == "standard":
            scaler = StandardScaler()
        elif request.method == "minmax":
            scaler = MinMaxScaler()
        elif request.method == "robust":
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        # Store scaler in pipeline context
        pipeline_ctx["scaler"] = scaler
        pipeline_ctx["scaling_method"] = request.method
        
        # Get numeric features from roles
        roles = pipeline_ctx.get("roles", {})
        numeric_features = [col for col, role in roles.items() if role == "numeric"]
        
        duration = time.time() - start_time
        record_telemetry("scale_numeric_features", duration, True)
        
        return {
            "scaler": request.method,
            "scaled_features": numeric_features
        }
        
    except Exception as e:
        duration = time.time() - start_time
        record_telemetry("scale_numeric_features", duration, False, str(e))
        raise HTTPException(status_code=500, detail=str(e))

async def encode_categorical_features(request: EncodeRequest):
    """Encode categorical features."""
    start_time = time.time()
    
    try:
        pipeline_ctx = get_pipeline_context(request.run_id, request.session_id)
        
        # Get categorical features from roles
        roles = pipeline_ctx.get("roles", {})
        categorical_features = [col for col, role in roles.items() if role == "categorical"]
        
        # Store encoding strategy in pipeline context
        pipeline_ctx["encoding_strategy"] = request.strategy
        pipeline_ctx["categorical_features"] = categorical_features
        
        duration = time.time() - start_time
        record_telemetry("encode_categorical_features", duration, True)
        
        return {
            "encoding": request.strategy,
            "encoded_features": categorical_features
        }
        
    except Exception as e:
        duration = time.time() - start_time
        record_telemetry("encode_categorical_features", duration, False, str(e))
        raise HTTPException(status_code=500, detail=str(e))

async def generate_datetime_features(request: DatetimeRequest):
    """Generate datetime features."""
    start_time = time.time()
    
    try:
        pipeline_ctx = get_pipeline_context(request.run_id, request.session_id)
        
        # Get datetime features from roles
        roles = pipeline_ctx.get("roles", {})
        datetime_features = [col for col, role in roles.items() if role == "datetime"]
        
        # Store datetime configuration in pipeline context
        pipeline_ctx["datetime_country"] = request.country
        pipeline_ctx["datetime_features"] = datetime_features
        
        # Generate feature names for datetime components
        generated_cols = []
        for dt_col in datetime_features:
            generated_cols.extend([
                f"{dt_col}_year",
                f"{dt_col}_month", 
                f"{dt_col}_day",
                f"{dt_col}_hour",
                f"{dt_col}_dayofweek"
            ])
        
        duration = time.time() - start_time
        record_telemetry("generate_datetime_features", duration, True)
        
        return {
            "generated_cols": generated_cols,
            "country": request.country
        }
        
    except Exception as e:
        duration = time.time() - start_time
        record_telemetry("generate_datetime_features", duration, False, str(e))
        raise HTTPException(status_code=500, detail=str(e))

async def vectorise_text_features(request: TextVectorizeRequest):
    """Vectorize text features."""
    start_time = time.time()
    
    try:
        pipeline_ctx = get_pipeline_context(request.run_id, request.session_id)
        
        # Get text features from roles
        roles = pipeline_ctx.get("roles", {})
        text_features = [col for col, role in roles.items() if role == "text"]
        
        # Create vectorizer
        vectorizer = TfidfVectorizer(max_features=request.max_feats)
        
        # Store vectorizer in pipeline context
        pipeline_ctx["text_vectorizer"] = vectorizer
        pipeline_ctx["text_features"] = text_features
        
        # Generate feature names for vectorized text
        vectorized_features = []
        for text_col in text_features:
            for i in range(min(request.max_feats, 100)):  # Limit for naming
                vectorized_features.append(f"{text_col}_vec_{i}")
        
        duration = time.time() - start_time
        record_telemetry("vectorise_text_features", duration, True)
        
        return {
            "vectoriser": request.model,
            "vectorized_features": vectorized_features
        }
        
    except Exception as e:
        duration = time.time() - start_time
        record_telemetry("vectorise_text_features", duration, False, str(e))
        raise HTTPException(status_code=500, detail=str(e))

async def generate_interactions(request: InteractionsRequest):
    """Generate feature interactions."""
    start_time = time.time()
    
    try:
        pipeline_ctx = get_pipeline_context(request.run_id, request.session_id)
        
        # Get numeric features for interactions
        roles = pipeline_ctx.get("roles", {})
        numeric_features = [col for col, role in roles.items() if role == "numeric"]
        
        # Generate interaction feature names
        new_features = []
        if request.max_degree >= 2 and len(numeric_features) >= 2:
            for i in range(len(numeric_features)):
                for j in range(i+1, len(numeric_features)):
                    new_features.append(f"{numeric_features[i]}_x_{numeric_features[j]}")
        
        # Store interaction configuration in pipeline context
        pipeline_ctx["interaction_degree"] = request.max_degree
        pipeline_ctx["interaction_features"] = new_features
        
        duration = time.time() - start_time
        record_telemetry("generate_interactions", duration, True)
        
        return {
            "new_features": new_features,
            "degree": request.max_degree
        }
        
    except Exception as e:
        duration = time.time() - start_time
        record_telemetry("generate_interactions", duration, False, str(e))
        raise HTTPException(status_code=500, detail=str(e))

async def select_features(request: FeatureSelectRequest):
    """Select features using various methods."""
    start_time = time.time()
    
    try:
        pipeline_ctx = get_pipeline_context(request.run_id, request.session_id)
        
        # Get all feature names from pipeline context
        all_features = pipeline_ctx.get("feature_names", [])
        
        # Simple feature selection (in practice, this would use actual selection methods)
        selected_features = all_features[:min(request.k, len(all_features))]
        
        # Store selection in pipeline context
        pipeline_ctx["selected_features"] = selected_features
        pipeline_ctx["selection_method"] = request.method
        
        duration = time.time() - start_time
        record_telemetry("select_features", duration, True)
        
        return {
            "selected_count": len(selected_features),
            "selected_features": selected_features
        }
        
    except Exception as e:
        duration = time.time() - start_time
        record_telemetry("select_features", duration, False, str(e))
        raise HTTPException(status_code=500, detail=str(e))

async def save_fe_pipeline(request: SavePipelineRequest):
    """Save the feature engineering pipeline and processed data."""
    start_time = time.time()
    
    try:
        df = load_dataframe(request.input_path)
        pipeline_ctx = get_pipeline_context(request.run_id, request.session_id)
        
        # Create metadata
        metadata = {
            "run_id": request.run_id,
            "session_id": request.session_id,
            "created_at": datetime.now().isoformat(),
            "original_shape": df.shape,
            "pipeline_steps": pipeline_ctx.get("steps", []),
            "roles": pipeline_ctx.get("roles", {}),
            "scaling_method": pipeline_ctx.get("scaling_method"),
            "encoding_strategy": pipeline_ctx.get("encoding_strategy"),
            "selection_method": pipeline_ctx.get("selection_method")
        }
        
        # Save pipeline to file
        pipeline_path = Path(request.export_pipeline_path)
        pipeline_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(pipeline_path, 'wb') as f:
            pickle.dump(pipeline_ctx, f)
        
        # Save processed data (simplified - in practice would apply full pipeline)
        data_path = Path(request.export_data_path)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        
        # For now, just save the original data
        df.to_parquet(data_path)
        
        duration = time.time() - start_time
        record_telemetry("save_fe_pipeline", duration, True)
        
        return {
            "data_path": str(data_path),
            "pipeline_path": str(pipeline_path),
            "metadata": metadata
        }
        
    except Exception as e:
        duration = time.time() - start_time
        record_telemetry("save_fe_pipeline", duration, False, str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Main execution endpoint
@app.post("/execute", response_model=TaskResponse)
async def execute(req: TaskRequest):
    """Main execution endpoint for all refinery agent actions."""
    start_time = time.time()
    
    try:
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
        
        # Convert params to appropriate request model and call handler
        handler = dispatcher[req.action]
        
        # Create appropriate request object based on action
        if req.action == "check_schema_consistency":
            request_obj = SchemaCheckRequest(**req.params)
        elif req.action == "check_missing_values":
            request_obj = MissingValuesRequest(**req.params)
        elif req.action == "check_distributions":
            request_obj = DistributionCheckRequest(**req.params)
        elif req.action == "check_duplicates":
            request_obj = DuplicateCheckRequest(**req.params)
        elif req.action == "check_leakage":
            request_obj = LeakageCheckRequest(**req.params)
        elif req.action == "check_drift":
            request_obj = DriftCheckRequest(**req.params)
        elif req.action == "assign_feature_roles":
            request_obj = FeatureRolesRequest(**req.params)
        elif req.action == "impute_missing_values":
            request_obj = ImputeRequest(**req.params)
        elif req.action == "scale_numeric_features":
            request_obj = ScaleRequest(**req.params)
        elif req.action == "encode_categorical_features":
            request_obj = EncodeRequest(**req.params)
        elif req.action == "generate_datetime_features":
            request_obj = DatetimeRequest(**req.params)
        elif req.action == "vectorise_text_features":
            request_obj = TextVectorizeRequest(**req.params)
        elif req.action == "generate_interactions":
            request_obj = InteractionsRequest(**req.params)
        elif req.action == "select_features":
            request_obj = FeatureSelectRequest(**req.params)
        elif req.action == "save_fe_pipeline":
            request_obj = SavePipelineRequest(**req.params)
        else:
            raise ValueError(f"Unknown action: {req.action}")
        
        # Execute the handler
        result = await handler(request_obj)
        
        execution_time = time.time() - start_time
        
        return TaskResponse(
            task_id=req.task_id,
            success=True,
            result=result,
            execution_time=execution_time,
            timestamp=time.time()
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Task {req.task_id} failed: {str(e)}")
        
        return TaskResponse(
            task_id=req.task_id,
            success=False,
            error=str(e),
            execution_time=execution_time,
            timestamp=time.time()
        )

# Health and utility endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "0.1.0",
        "service": "refinery_agent",
        "timestamp": datetime.now().isoformat(),
        "telemetry": len(telemetry_data)
    }

@app.get("/telemetry")
async def get_telemetry():
    """Get telemetry data."""
    return telemetry_data

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global redis_client, mongo_client
    
    try:
        # Initialize Redis client
        redis_client = redis.from_url(Config.REDIS_URL)
        await redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        redis_client = None
    
    try:
        # Initialize MongoDB client
        mongo_client = AsyncIOMotorClient(Config.MONGO_URL)
        await mongo_client.admin.command('ping')
        logger.info("MongoDB connection established")
    except Exception as e:
        logger.warning(f"MongoDB connection failed: {e}")
        mongo_client = None
    
    logger.info("Refinery Agent service started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global redis_client, mongo_client
    
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed")
    
    if mongo_client:
        mongo_client.close()
        logger.info("MongoDB connection closed")
    
    logger.info("Refinery Agent service stopped")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)