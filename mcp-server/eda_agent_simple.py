#!/usr/bin/env python3
"""
Simplified EDA (Exploratory Data Analysis) Agent Service for Testing
A FastAPI-based microservice that provides data analysis tools without external dependencies.
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="EDA Agent Service (Simplified)",
    description="Exploratory Data Analysis microservice for testing",
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

# Global state (in-memory for testing)
data_store: Dict[str, pd.DataFrame] = {}
telemetry_data: Dict[str, Any] = {}

# Configuration
class Config:
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

# Utility functions
def get_dataset(name: str) -> pd.DataFrame:
    """Get dataset from store."""
    if name not in data_store:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
    return data_store[name]

def sample_dataset(df: pd.DataFrame, sample_size: int = Config.SAMPLE_SIZE) -> pd.DataFrame:
    """Sample dataset if it's too large."""
    if len(df) > sample_size:
        return df.sample(n=sample_size, random_state=Config.RANDOM_SEED)
    return df

def create_visualization_base64(fig) -> str:
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return img_str

def record_telemetry(operation: str, duration: float, success: bool, error: Optional[str] = None):
    """Record telemetry data."""
    if operation not in telemetry_data:
        telemetry_data[operation] = []
    
    telemetry_data[operation].append({
        "timestamp": datetime.now().isoformat(),
        "duration": duration,
        "success": success,
        "error": error
    })

# API Endpoints
@app.post("/load_data", response_model=LoadDataResponse)
async def load_data(request: LoadDataRequest):
    """Load data file into memory."""
    start_time = time.time()
    success = False
    error = None
    
    try:
        # Load data based on file type
        if request.file_type == "csv" or request.path.endswith('.csv'):
            df = pd.read_csv(request.path)
        elif request.file_type == "xlsx" or request.path.endswith('.xlsx'):
            df = pd.read_excel(request.path)
        elif request.file_type == "json" or request.path.endswith('.json'):
            df = pd.read_json(request.path)
        else:
            # Try to infer file type
            try:
                df = pd.read_csv(request.path)
            except:
                try:
                    df = pd.read_excel(request.path)
                except:
                    df = pd.read_json(request.path)
        
        # Store in memory
        data_store[request.name] = df
        
        # Prepare response
        response = LoadDataResponse(
            name=request.name,
            rows=len(df),
            cols=len(df.columns),
            dtypes=df.dtypes.astype(str).to_dict(),
            memory_usage=f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
            sample_preview=df.head().to_dict('records')
        )
        
        success = True
        
    except Exception as e:
        error = str(e)
        logger.error(f"Failed to load data: {error}")
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
            dtypes=df.dtypes.astype(str).to_dict(),
            memory_usage=f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
            preview=df.head().to_dict('records'),
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
    """Get statistical summary of dataset."""
    start_time = time.time()
    success = False
    error = None
    
    try:
        df = get_dataset(request.name)
        sample_df = sample_dataset(df, request.sample_size)
        
        # Calculate statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        descriptive_stats = sample_df[numeric_cols].describe().to_dict()
        
        # Correlation matrix
        correlation_matrix = sample_df[numeric_cols].corr().to_dict()
        
        # Skewness and kurtosis
        skewness = sample_df[numeric_cols].skew().to_dict()
        kurtosis = sample_df[numeric_cols].kurtosis().to_dict()
        
        response = StatisticalSummaryResponse(
            descriptive_stats=descriptive_stats,
            correlation_matrix=correlation_matrix,
            skewness=skewness,
            kurtosis=kurtosis
        )
        
        success = True
        
    except Exception as e:
        error = str(e)
        logger.error(f"Failed to get statistical summary: {error}")
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
        
        # Calculate missing data
        missing_counts = df.isnull().sum().to_dict()
        missing_percentages = (df.isnull().sum() / len(df) * 100).to_dict()
        
        # Find columns with missing data
        columns_with_missing = [col for col, count in missing_counts.items() if count > 0]
        
        # Generate recommendations
        recommendations = []
        if columns_with_missing:
            recommendations.append(f"Found missing data in {len(columns_with_missing)} columns")
            for col in columns_with_missing:
                percentage = missing_percentages[col]
                if percentage > 50:
                    recommendations.append(f"Column '{col}' has {percentage:.1f}% missing data - consider dropping")
                elif percentage > 10:
                    recommendations.append(f"Column '{col}' has {percentage:.1f}% missing data - consider imputation")
                else:
                    recommendations.append(f"Column '{col}' has {percentage:.1f}% missing data - safe to impute")
        else:
            recommendations.append("No missing data found in the dataset")
        
        # Create visualization
        if columns_with_missing:
            fig, ax = plt.subplots(figsize=(10, 6))
            missing_data = pd.DataFrame({
                'column': list(missing_percentages.keys()),
                'missing_percentage': list(missing_percentages.values())
            })
            missing_data.plot(x='column', y='missing_percentage', kind='bar', ax=ax)
            ax.set_title('Missing Data Percentage by Column')
            ax.set_ylabel('Missing Percentage (%)')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            
            visualization_base64 = create_visualization_base64(fig)
        else:
            visualization_base64 = None
        
        response = MissingDataResult(
            missing_counts=missing_counts,
            missing_percentages=missing_percentages,
            missing_patterns=[],  # Simplified for testing
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
    """Create visualization of dataset."""
    start_time = time.time()
    success = False
    error = None
    
    try:
        df = get_dataset(request.name)
        sample_df = sample_dataset(df, request.sample_size)
        
        # Select columns to visualize
        if request.columns:
            columns = [col for col in request.columns if col in sample_df.columns]
        else:
            columns = sample_df.select_dtypes(include=[np.number]).columns.tolist()[:5]  # First 5 numeric columns
        
        if not columns:
            raise ValueError("No suitable columns found for visualization")
        
        # Create visualization based on type
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if request.chart_type == "histogram":
            sample_df[columns[0]].hist(ax=ax, bins=30)
            ax.set_title(f'Histogram of {columns[0]}')
            ax.set_xlabel(columns[0])
            ax.set_ylabel('Frequency')
        
        elif request.chart_type == "boxplot":
            sample_df[columns].boxplot(ax=ax)
            ax.set_title('Box Plot of Numeric Columns')
            ax.set_ylabel('Values')
            ax.tick_params(axis='x', rotation=45)
        
        elif request.chart_type == "correlation":
            corr_matrix = sample_df[columns].corr()
            im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks(range(len(columns)))
            ax.set_yticks(range(len(columns)))
            ax.set_xticklabels(columns, rotation=45)
            ax.set_yticklabels(columns)
            ax.set_title('Correlation Matrix')
            plt.colorbar(im, ax=ax)
        
        else:
            # Default to histogram
            sample_df[columns[0]].hist(ax=ax, bins=30)
            ax.set_title(f'Histogram of {columns[0]}')
            ax.set_xlabel(columns[0])
            ax.set_ylabel('Frequency')
        
        plt.tight_layout()
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
    """Infer schema from dataset."""
    start_time = time.time()
    success = False
    error = None
    
    try:
        df = get_dataset(request.name)
        
        schema = {}
        confidence_scores = {}
        recommendations = []
        
        for column in df.columns:
            col_data = df[column]
            
            # Determine data type
            if col_data.dtype == 'object':
                # Check if it's datetime
                try:
                    pd.to_datetime(col_data.dropna())
                    inferred_type = "datetime"
                    confidence = 0.9
                except:
                    # Check if it's categorical
                    unique_ratio = col_data.nunique() / len(col_data)
                    if unique_ratio < 0.1:
                        inferred_type = "categorical"
                        confidence = 0.8
                    else:
                        inferred_type = "string"
                        confidence = 0.7
            elif col_data.dtype in ['int64', 'int32']:
                inferred_type = "integer"
                confidence = 0.95
            elif col_data.dtype in ['float64', 'float32']:
                inferred_type = "float"
                confidence = 0.95
            elif col_data.dtype == 'bool':
                inferred_type = "boolean"
                confidence = 0.95
            else:
                inferred_type = "unknown"
                confidence = 0.5
            
            schema[column] = {
                "type": inferred_type,
                "nullable": col_data.isnull().any(),
                "unique_count": col_data.nunique(),
                "sample_values": col_data.dropna().head(3).tolist()
            }
            
            confidence_scores[column] = confidence
            
            # Generate recommendations
            if confidence < request.confidence_threshold:
                recommendations.append(f"Low confidence ({confidence:.2f}) for column '{column}' type inference")
        
        # Create YAML schema
        yaml_schema = f"""
dataset: {request.name}
columns:
"""
        for col, info in schema.items():
            yaml_schema += f"  {col}:\n"
            yaml_schema += f"    type: {info['type']}\n"
            yaml_schema += f"    nullable: {info['nullable']}\n"
            yaml_schema += f"    unique_count: {info['unique_count']}\n"
        
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
        sample_df = sample_dataset(df, Config.SAMPLE_SIZE)
        
        # Select numeric columns
        if request.columns:
            columns = [col for col in request.columns if col in sample_df.select_dtypes(include=[np.number]).columns]
        else:
            columns = sample_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not columns:
            raise ValueError("No numeric columns found for outlier detection")
        
        outlier_indices = set()
        scores = {}
        
        if request.method == "iqr":
            # IQR method
            for col in columns:
                Q1 = sample_df[col].quantile(0.25)
                Q3 = sample_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = sample_df[(sample_df[col] < lower_bound) | (sample_df[col] > upper_bound)].index
                outlier_indices.update(outliers)
                
        elif request.method == "isolation_forest":
            # Isolation Forest
            iso_forest = IsolationForest(contamination=request.contamination, random_state=Config.RANDOM_SEED)
            predictions = iso_forest.fit_predict(sample_df[columns])
            outlier_indices.update(sample_df[predictions == -1].index)
            
        elif request.method == "lof":
            # Local Outlier Factor
            lof = LocalOutlierFactor(contamination=request.contamination)
            predictions = lof.fit_predict(sample_df[columns])
            outlier_indices.update(sample_df[predictions == -1].index)
        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 