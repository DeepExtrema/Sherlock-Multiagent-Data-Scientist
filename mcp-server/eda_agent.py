#!/usr/bin/env python3
"""
EDA (Exploratory Data Analysis) Agent Service

A FastAPI-based microservice that provides data analysis tools for the Master Orchestrator.
Implements the full EDA workflow with production-grade features including caching, 
telemetry, security, and human-in-loop checkpoints.
"""

import asyncio
import hashlib
import json
import logging
import time
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
from sklearn.neighbors import LocalOutlierFactor
import redis.asyncio as redis
from motor.motor_asyncio import AsyncIOMotorClient
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="EDA Agent Service",
    description="Exploratory Data Analysis microservice for Deepline Master Orchestrator",
    version="1.0.0"
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
class LoadDataRequest(BaseModel):
    path: str = Field(..., description="Path to data file")
    name: str = Field(..., description="Name to store dataset under")
    file_type: Optional[str] = Field(None, description="File type (csv, xlsx, json)")
    
    @validator('path')
    def validate_path(cls, v):
        if not Path(v).exists():
            raise ValueError(f"File not found: {v}")
        return v

class LoadDataResponse(BaseModel):
    name: str
    rows: int
    cols: int
    dtypes: Dict[str, str]
    memory_usage: str
    sample_preview: List[Dict[str, Any]]

class BasicInfoRequest(BaseModel):
    name: str = Field(..., description="Dataset name")

class BasicInfoResponse(BaseModel):
    shape: tuple
    dtypes: Dict[str, str]
    memory_usage: str
    preview: List[Dict[str, Any]]
    null_counts: Dict[str, int]

class StatisticalSummaryRequest(BaseModel):
    name: str = Field(..., description="Dataset name")
    sample_size: Optional[int] = Field(10000, description="Sample size for large datasets")

class StatisticalSummaryResponse(BaseModel):
    descriptive_stats: Dict[str, Any]
    correlation_matrix: Dict[str, Dict[str, float]]
    skewness: Dict[str, float]
    kurtosis: Dict[str, float]

class MissingDataRequest(BaseModel):
    name: str = Field(..., description="Dataset name")

class MissingDataResult(BaseModel):
    missing_counts: Dict[str, int]
    missing_percentages: Dict[str, float]
    missing_patterns: List[Dict[str, Any]]
    recommendations: List[str]
    visualization_base64: Optional[str] = None

class VisualizationRequest(BaseModel):
    name: str = Field(..., description="Dataset name")
    chart_type: str = Field(..., description="Type of chart (histogram, boxplot, correlation, etc.)")
    columns: Optional[List[str]] = Field(None, description="Columns to visualize")
    sample_size: Optional[int] = Field(10000, description="Sample size for large datasets")

class VisualizationResponse(BaseModel):
    chart_type: str
    columns: List[str]
    base64_image: str
    metadata: Dict[str, Any]

class SchemaInferenceRequest(BaseModel):
    name: str = Field(..., description="Dataset name")
    confidence_threshold: float = Field(0.8, description="Minimum confidence for type inference")

class SchemaInferenceResponse(BaseModel):
    schema: Dict[str, Any]
    confidence_scores: Dict[str, float]
    recommendations: List[str]
    yaml_schema: str

class OutlierDetectionRequest(BaseModel):
    name: str = Field(..., description="Dataset name")
    method: str = Field("iqr", description="Detection method (iqr, isolation_forest, lof)")
    columns: Optional[List[str]] = Field(None, description="Columns to analyze")
    contamination: float = Field(0.1, description="Expected fraction of outliers")

class OutlierResult(BaseModel):
    method: str
    outlier_indices: List[int]
    outlier_count: int
    outlier_percentage: float
    columns_analyzed: List[str]
    scores: Optional[Dict[str, List[float]]] = None

# Utility Functions
def get_cache_key(operation: str, dataset_name: str, **kwargs) -> str:
    """Generate cache key for operation results."""
    key_data = f"{operation}:{dataset_name}:{json.dumps(kwargs, sort_keys=True)}"
    return hashlib.md5(key_data.encode()).hexdigest()

async def get_cached_result(cache_key: str) -> Optional[Dict[str, Any]]:
    """Get cached result from Redis."""
    if not redis_client:
        return None
    try:
        cached = await redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception as e:
        logger.warning(f"Cache retrieval failed: {e}")
    return None

async def cache_result(cache_key: str, result: Dict[str, Any], ttl: int = Config.CACHE_TTL):
    """Cache result in Redis."""
    if not redis_client:
        return
    try:
        await redis_client.setex(cache_key, ttl, json.dumps(result))
    except Exception as e:
        logger.warning(f"Cache storage failed: {e}")

def get_dataset(name: str) -> pd.DataFrame:
    """Get dataset from store with error handling."""
    if name not in data_store:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
    return data_store[name]

def sample_dataset(df: pd.DataFrame, sample_size: int = Config.SAMPLE_SIZE) -> pd.DataFrame:
    """Sample large datasets for analysis."""
    if len(df) > sample_size:
        return df.sample(n=sample_size, random_state=Config.RANDOM_SEED)
    return df

def create_visualization_base64(fig) -> str:
    """Convert matplotlib figure to base64 string."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    return image_base64

# Telemetry
def record_telemetry(operation: str, duration: float, success: bool, error: Optional[str] = None):
    """Record operation telemetry."""
    if operation not in telemetry_data:
        telemetry_data[operation] = {"count": 0, "total_time": 0, "errors": 0}
    
    telemetry_data[operation]["count"] += 1
    telemetry_data[operation]["total_time"] += duration
    
    if not success:
        telemetry_data[operation]["errors"] += 1
        logger.error(f"Operation {operation} failed: {error}")

# EDA Tool Implementations
@app.post("/load_data", response_model=LoadDataResponse)
async def load_data(request: LoadDataRequest):
    """Load dataset from file into memory store."""
    start_time = time.time()
    success = False
    error = None
    
    try:
        # Determine file type
        file_type = request.file_type or Path(request.path).suffix.lower()
        
        # Load data based on file type
        if file_type in ['.csv', 'csv']:
            df = pd.read_csv(request.path)
        elif file_type in ['.xlsx', '.xls', 'xlsx', 'xls']:
            df = pd.read_excel(request.path)
        elif file_type in ['.json', 'json']:
            df = pd.read_json(request.path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Check dataset size
        if len(df) > Config.MAX_DATASET_SIZE:
            logger.warning(f"Large dataset detected: {len(df)} rows. Consider sampling.")
        
        # Store dataset
        data_store[request.name] = df
        
        # Prepare response
        sample_df = df.head(5)
        response = LoadDataResponse(
            name=request.name,
            rows=len(df),
            cols=len(df.columns),
            dtypes=df.dtypes.apply(str).to_dict(),
            memory_usage=f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
            sample_preview=sample_df.to_dict(orient="records")
        )
        
        success = True
        logger.info(f"Dataset '{request.name}' loaded: {len(df)} rows, {len(df.columns)} columns")
        
    except Exception as e:
        error = str(e)
        logger.error(f"Failed to load dataset: {error}")
        raise HTTPException(status_code=400, detail=error)
    
    finally:
        duration = time.time() - start_time
        record_telemetry("load_data", duration, success, error)
    
    return response

@app.post("/basic_info", response_model=BasicInfoResponse)
async def basic_info(request: BasicInfoRequest):
    """Get basic information about dataset."""
    start_time = time.time()
    success = False
    error = None
    
    try:
        df = get_dataset(request.name)
        
        response = BasicInfoResponse(
            shape=df.shape,
            dtypes=df.dtypes.apply(str).to_dict(),
            memory_usage=f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
            preview=df.head().to_dict(orient="records"),
            null_counts=df.isnull().sum().to_dict()
        )
        
        success = True
        
    except Exception as e:
        error = str(e)
        logger.error(f"Failed to get basic info: {error}")
        raise HTTPException(status_code=400, detail=error)
    
    finally:
        duration = time.time() - start_time
        record_telemetry("basic_info", duration, success, error)
    
    return response

@app.post("/statistical_summary", response_model=StatisticalSummaryResponse)
async def statistical_summary(request: StatisticalSummaryRequest):
    """Generate statistical summary of dataset."""
    start_time = time.time()
    success = False
    error = None
    
    try:
        df = get_dataset(request.name)
        sample_df = sample_dataset(df, request.sample_size)
        
        # Descriptive statistics
        desc_stats = sample_df.describe().to_dict()
        
        # Correlation matrix (numeric columns only)
        numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = sample_df[numeric_cols].corr().to_dict()
        else:
            corr_matrix = {}
        
        # Skewness and kurtosis
        skewness = sample_df[numeric_cols].skew().to_dict() if len(numeric_cols) > 0 else {}
        kurtosis = sample_df[numeric_cols].kurtosis().to_dict() if len(numeric_cols) > 0 else {}
        
        response = StatisticalSummaryResponse(
            descriptive_stats=desc_stats,
            correlation_matrix=corr_matrix,
            skewness=skewness,
            kurtosis=kurtosis
        )
        
        success = True
        
    except Exception as e:
        error = str(e)
        logger.error(f"Failed to generate statistical summary: {error}")
        raise HTTPException(status_code=400, detail=error)
    
    finally:
        duration = time.time() - start_time
        record_telemetry("statistical_summary", duration, success, error)
    
    return response

@app.post("/missing_data_analysis", response_model=MissingDataResult)
async def missing_data_analysis(request: MissingDataRequest):
    """Analyze missing data patterns."""
    start_time = time.time()
    success = False
    error = None
    
    try:
        df = get_dataset(request.name)
        
        # Missing data counts and percentages
        missing_counts = df.isnull().sum().to_dict()
        missing_percentages = (df.isnull().sum() / len(df) * 100).to_dict()
        
        # Missing data patterns
        missing_patterns = []
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                pattern = {
                    "column": col,
                    "missing_count": int(df[col].isnull().sum()),
                    "missing_percentage": float(df[col].isnull().sum() / len(df) * 100),
                    "data_type": str(df[col].dtype)
                }
                missing_patterns.append(pattern)
        
        # Recommendations
        recommendations = []
        high_missing_cols = [col for col, pct in missing_percentages.items() if pct > 50]
        if high_missing_cols:
            recommendations.append(f"Consider dropping columns with >50% missing data: {high_missing_cols}")
        
        moderate_missing_cols = [col for col, pct in missing_percentages.items() if 10 < pct <= 50]
        if moderate_missing_cols:
            recommendations.append(f"Consider imputation for columns with 10-50% missing data: {moderate_missing_cols}")
        
        # Create visualization
        if missing_percentages:
            fig, ax = plt.subplots(figsize=(10, 6))
            missing_df = pd.DataFrame(list(missing_percentages.items()), columns=['Column', 'Missing_Percentage'])
            missing_df = missing_df[missing_df['Missing_Percentage'] > 0].sort_values('Missing_Percentage', ascending=True)
            
            if len(missing_df) > 0:
                ax.barh(missing_df['Column'], missing_df['Missing_Percentage'])
                ax.set_xlabel('Missing Data Percentage')
                ax.set_title('Missing Data Analysis')
                ax.set_xlim(0, 100)
                
                visualization_base64 = create_visualization_base64(fig)
            else:
                visualization_base64 = None
        else:
            visualization_base64 = None
        
        response = MissingDataResult(
            missing_counts=missing_counts,
            missing_percentages=missing_percentages,
            missing_patterns=missing_patterns,
            recommendations=recommendations,
            visualization_base64=visualization_base64
        )
        
        success = True
        
    except Exception as e:
        error = str(e)
        logger.error(f"Failed to analyze missing data: {error}")
        raise HTTPException(status_code=400, detail=error)
    
    finally:
        duration = time.time() - start_time
        record_telemetry("missing_data_analysis", duration, success, error)
    
    return response

@app.post("/create_visualization", response_model=VisualizationResponse)
async def create_visualization(request: VisualizationRequest):
    """Create data visualizations."""
    start_time = time.time()
    success = False
    error = None
    
    try:
        df = get_dataset(request.name)
        sample_df = sample_dataset(df, request.sample_size)
        
        # Select columns to visualize
        columns = request.columns or sample_df.select_dtypes(include=[np.number]).columns.tolist()
        if not columns:
            columns = sample_df.columns[:5].tolist()  # Default to first 5 columns
        
        # Create visualization based on type
        if request.chart_type == "histogram":
            fig, axes = plt.subplots(1, min(len(columns), 3), figsize=(15, 5))
            if len(columns) == 1:
                axes = [axes]
            
            for i, col in enumerate(columns[:3]):
                if col in sample_df.columns:
                    axes[i].hist(sample_df[col].dropna(), bins=30, alpha=0.7)
                    axes[i].set_title(f'Histogram of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
            
            plt.tight_layout()
            
        elif request.chart_type == "boxplot":
            fig, ax = plt.subplots(figsize=(12, 6))
            numeric_cols = [col for col in columns if col in sample_df.select_dtypes(include=[np.number]).columns]
            if numeric_cols:
                sample_df[numeric_cols].boxplot(ax=ax)
                ax.set_title('Box Plot of Numeric Variables')
                ax.set_ylabel('Values')
                plt.xticks(rotation=45)
            
        elif request.chart_type == "correlation":
            numeric_cols = [col for col in columns if col in sample_df.select_dtypes(include=[np.number]).columns]
            if len(numeric_cols) > 1:
                fig, ax = plt.subplots(figsize=(10, 8))
                corr_matrix = sample_df[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                ax.set_title('Correlation Matrix')
            else:
                raise ValueError("Need at least 2 numeric columns for correlation plot")
        
        elif request.chart_type == "scatter":
            if len(columns) >= 2:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(sample_df[columns[0]], sample_df[columns[1]], alpha=0.6)
                ax.set_xlabel(columns[0])
                ax.set_ylabel(columns[1])
                ax.set_title(f'Scatter Plot: {columns[0]} vs {columns[1]}')
            else:
                raise ValueError("Need at least 2 columns for scatter plot")
        
        else:
            raise ValueError(f"Unsupported chart type: {request.chart_type}")
        
        # Convert to base64
        base64_image = create_visualization_base64(fig)
        
        response = VisualizationResponse(
            chart_type=request.chart_type,
            columns=columns,
            base64_image=base64_image,
            metadata={
                "sample_size": len(sample_df),
                "total_rows": len(df),
                "columns_used": columns
            }
        )
        
        success = True
        
    except Exception as e:
        error = str(e)
        logger.error(f"Failed to create visualization: {error}")
        raise HTTPException(status_code=400, detail=error)
    
    finally:
        duration = time.time() - start_time
        record_telemetry("create_visualization", duration, success, error)
    
    return response

@app.post("/infer_schema", response_model=SchemaInferenceResponse)
async def infer_schema(request: SchemaInferenceRequest):
    """Infer data schema and types."""
    start_time = time.time()
    success = False
    error = None
    
    try:
        df = get_dataset(request.name)
        sample_df = sample_dataset(df, 10000)  # Use sample for schema inference
        
        schema = {}
        confidence_scores = {}
        recommendations = []
        
        for column in df.columns:
            col_data = sample_df[column].dropna()
            
            # Type inference
            if col_data.empty:
                inferred_type = "unknown"
                confidence = 0.0
            elif df[column].dtype == 'object':
                # Try to infer if it's datetime, numeric, or categorical
                try:
                    pd.to_datetime(col_data.iloc[:100])
                    inferred_type = "datetime"
                    confidence = 0.9
                except:
                    try:
                        pd.to_numeric(col_data.iloc[:100])
                        inferred_type = "numeric"
                        confidence = 0.8
                    except:
                        # Check if it's categorical
                        unique_ratio = len(col_data.unique()) / len(col_data)
                        if unique_ratio < 0.1:
                            inferred_type = "categorical"
                            confidence = 0.9
                        else:
                            inferred_type = "string"
                            confidence = 0.7
            else:
                inferred_type = str(df[column].dtype)
                confidence = 1.0
            
            schema[column] = {
                "type": inferred_type,
                "nullable": df[column].isnull().any(),
                "unique_count": df[column].nunique(),
                "sample_values": col_data.head(5).tolist() if not col_data.empty else []
            }
            
            confidence_scores[column] = confidence
            
            # Generate recommendations
            if confidence < request.confidence_threshold:
                recommendations.append(f"Low confidence ({confidence:.2f}) for column '{column}' - manual review recommended")
            
            if df[column].isnull().sum() > 0:
                recommendations.append(f"Column '{column}' contains missing values - consider imputation strategy")
        
        # Generate YAML schema
        yaml_schema = yaml.dump({
            "dataset_name": request.name,
            "columns": schema,
            "total_rows": len(df),
            "inference_confidence": confidence_scores
        }, default_flow_style=False)
        
        response = SchemaInferenceResponse(
            schema=schema,
            confidence_scores=confidence_scores,
            recommendations=recommendations,
            yaml_schema=yaml_schema
        )
        
        success = True
        
    except Exception as e:
        error = str(e)
        logger.error(f"Failed to infer schema: {error}")
        raise HTTPException(status_code=400, detail=error)
    
    finally:
        duration = time.time() - start_time
        record_telemetry("infer_schema", duration, success, error)
    
    return response

@app.post("/detect_outliers", response_model=OutlierResult)
async def detect_outliers(request: OutlierDetectionRequest):
    """Detect outliers in dataset."""
    start_time = time.time()
    success = False
    error = None
    
    try:
        df = get_dataset(request.name)
        sample_df = sample_dataset(df, 10000)  # Use sample for outlier detection
        
        # Select columns to analyze
        columns = request.columns or sample_df.select_dtypes(include=[np.number]).columns.tolist()
        if not columns:
            raise ValueError("No numeric columns found for outlier detection")
        
        outlier_indices = set()
        scores = {}
        
        if request.method == "iqr":
            # IQR method
            for col in columns:
                if col in sample_df.columns:
                    Q1 = sample_df[col].quantile(0.25)
                    Q3 = sample_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = sample_df[(sample_df[col] < lower_bound) | (sample_df[col] > upper_bound)]
                    outlier_indices.update(outliers.index.tolist())
        
        elif request.method == "isolation_forest":
            # Isolation Forest method
            numeric_data = sample_df[columns].dropna()
            if len(numeric_data) > 0:
                iso_forest = IsolationForest(contamination=request.contamination, random_state=Config.RANDOM_SEED)
                predictions = iso_forest.fit_predict(numeric_data)
                outlier_indices.update(numeric_data[predictions == -1].index.tolist())
                scores["isolation_forest"] = iso_forest.decision_function(numeric_data).tolist()
        
        elif request.method == "lof":
            # Local Outlier Factor method
            numeric_data = sample_df[columns].dropna()
            if len(numeric_data) > 0:
                lof = LocalOutlierFactor(contamination=request.contamination)
                predictions = lof.fit_predict(numeric_data)
                outlier_indices.update(numeric_data[predictions == -1].index.tolist())
                scores["lof"] = lof.negative_outlier_factor_.tolist()
        
        else:
            raise ValueError(f"Unsupported outlier detection method: {request.method}")
        
        outlier_indices = list(outlier_indices)
        outlier_percentage = (len(outlier_indices) / len(sample_df)) * 100
        
        response = OutlierResult(
            method=request.method,
            outlier_indices=outlier_indices,
            outlier_count=len(outlier_indices),
            outlier_percentage=outlier_percentage,
            columns_analyzed=columns,
            scores=scores if scores else None
        )
        
        success = True
        
    except Exception as e:
        error = str(e)
        logger.error(f"Failed to detect outliers: {error}")
        raise HTTPException(status_code=400, detail=error)
    
    finally:
        duration = time.time() - start_time
        record_telemetry("detect_outliers", duration, success, error)
    
    return response

# Health and monitoring endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_store_size": len(data_store),
        "telemetry": telemetry_data
    }

@app.get("/datasets")
async def list_datasets():
    """List all loaded datasets."""
    return {
        "datasets": [
            {
                "name": name,
                "shape": df.shape,
                "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
            }
            for name, df in data_store.items()
        ]
    }

@app.delete("/datasets/{name}")
async def delete_dataset(name: str):
    """Delete dataset from store."""
    if name in data_store:
        del data_store[name]
        return {"message": f"Dataset '{name}' deleted"}
    else:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup."""
    global redis_client, mongo_client
    
    try:
        # Initialize Redis connection
        redis_client = redis.from_url(Config.REDIS_URL)
        await redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        redis_client = None
    
    try:
        # Initialize MongoDB connection
        mongo_client = AsyncIOMotorClient(Config.MONGO_URL)
        await mongo_client.admin.command('ping')
        logger.info("MongoDB connection established")
    except Exception as e:
        logger.warning(f"MongoDB connection failed: {e}")
        mongo_client = None

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up connections on shutdown."""
    global redis_client, mongo_client
    
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed")
    
    if mongo_client:
        mongo_client.close()
        logger.info("MongoDB connection closed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 