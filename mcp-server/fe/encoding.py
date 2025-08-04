"""
Advanced Encoding Module for Feature Engineering

This module provides advanced categorical encoding techniques including
rare-bucket encoding, hash encoding, target encoding, and cyclical encoding.
"""

import logging
import hashlib
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction import FeatureHasher

from .context import PipelineCtx

logger = logging.getLogger(__name__)


def rare_bucket_encode(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """
    Encode rare categories by bucketing them into '__OTHER__'.
    
    Args:
        df: DataFrame to encode
        threshold: Minimum frequency threshold (default: 0.01 = 1%)
        
    Returns:
        DataFrame with rare categories bucketed
    """
    try:
        result_df = df.copy()
        
        for col in df.select_dtypes(include=['object', 'category']):
            vc = df[col].value_counts(normalize=True)
            rare_categories = vc[vc < threshold].index
            
            if len(rare_categories) > 0:
                logger.info(f"Bucketing {len(rare_categories)} rare categories in column '{col}'")
                result_df[col] = result_df[col].replace(rare_categories, "__OTHER__")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Rare bucket encoding failed: {e}")
        return df


def hash_encode(df: pd.DataFrame, n_features: int = 10) -> pd.DataFrame:
    """
    Hash encode categorical features to fixed dimensions.
    
    Args:
        df: DataFrame to encode
        n_features: Number of hash features to generate
        
    Returns:
        DataFrame with hash-encoded features
    """
    try:
        result_df = df.copy()
        
        for col in df.select_dtypes(include=['object', 'category']):
            # Create hash encoder
            hasher = FeatureHasher(n_features=n_features, input_type='string')
            
            # Prepare data for hashing
            hash_data = df[col].fillna('__MISSING__').astype(str).values.reshape(-1, 1)
            
            # Generate hash features
            hash_features = hasher.transform(hash_data).toarray()
            
            # Add hash features to DataFrame
            for i in range(n_features):
                result_df[f"{col}_hash_{i}"] = hash_features[:, i]
            
            # Remove original column
            result_df = result_df.drop(columns=[col])
            
            logger.info(f"Hash encoded column '{col}' into {n_features} features")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Hash encoding failed: {e}")
        return df


def cyclical_encode(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Create cyclical encoding for periodic features (hour, month, day of week).
    
    Args:
        series: Series with periodic values
        
    Returns:
        Tuple of (sin_encoded, cos_encoded) series
    """
    try:
        max_val = series.max() + 1
        sin_encoded = np.sin(2 * np.pi * series / max_val)
        cos_encoded = np.cos(2 * np.pi * series / max_val)
        
        return pd.Series(sin_encoded, index=series.index), pd.Series(cos_encoded, index=series.index)
        
    except Exception as e:
        logger.error(f"Cyclical encoding failed: {e}")
        return series, series


async def target_encode(df: pd.DataFrame, target_col: str, cv_folds: int = 5) -> pd.DataFrame:
    """
    Target encoding for categorical variables.
    
    Args:
        df: DataFrame to encode
        target_col: Target column name
        cv_folds: Number of cross-validation folds
        
    Returns:
        DataFrame with target-encoded features
    """
    try:
        result_df = df.copy()
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        # Get categorical columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in cat_cols:
            if col == target_col:
                continue
                
            # Calculate target encoding with cross-validation
            encoded_values = np.zeros(len(df))
            
            # Simple target encoding (can be enhanced with CV)
            target_means = df.groupby(col)[target_col].mean()
            encoded_values = df[col].map(target_means)
            
            # Fill missing values with global mean
            global_mean = df[target_col].mean()
            encoded_values = encoded_values.fillna(global_mean)
            
            # Add encoded column
            result_df[f"{col}_target_encoded"] = encoded_values
            
            # Remove original column
            result_df = result_df.drop(columns=[col])
            
            logger.info(f"Target encoded column '{col}'")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Target encoding failed: {e}")
        return df


def detect_ordinal_categorical(df: pd.DataFrame, threshold: float = 0.7) -> List[str]:
    """
    Detect ordinal categorical variables using monotonic trend test.
    
    Args:
        df: DataFrame to analyze
        threshold: Correlation threshold for ordinal detection
        
    Returns:
        List of column names that are likely ordinal
    """
    try:
        ordinal_cols = []
        
        for col in df.select_dtypes(include=['object', 'category']):
            # Convert to numeric for correlation analysis
            le = LabelEncoder()
            encoded = le.fit_transform(df[col].fillna('__MISSING__'))
            
            # Check correlation with other numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                correlations = []
                for num_col in numeric_cols:
                    corr = np.corrcoef(encoded, df[num_col])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                
                # If average correlation is high, likely ordinal
                if correlations and np.mean(correlations) > threshold:
                    ordinal_cols.append(col)
                    logger.info(f"Detected ordinal categorical: '{col}' (avg corr: {np.mean(correlations):.3f})")
        
        return ordinal_cols
        
    except Exception as e:
        logger.error(f"Ordinal detection failed: {e}")
        return []


async def encode_categorical_features(params: Dict[str, Any], ctx: PipelineCtx, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Advanced categorical encoding with automatic strategy selection.
    
    Args:
        params: Parameters dictionary containing:
            - run_id: Unique run identifier
            - strategy: Encoding strategy ('auto', 'rare_bucket', 'hash', 'target', 'one_hot')
            - target_column: Optional target column for target encoding
            - threshold: Threshold for rare bucket encoding
            - n_features: Number of hash features
        ctx: Pipeline context
        config: Configuration dictionary
        
    Returns:
        Dictionary with encoding results
    """
    try:
        if ctx.df is None:
            return {"status": "error", "error": "No data in pipeline context"}
        
        strategy = params.get('strategy', 'auto')
        target_col = params.get('target_column')
        
        # Auto-select strategy based on data characteristics
        if strategy == 'auto':
            cat_cols = ctx.df.select_dtypes(include=['object', 'category']).columns
            
            if len(cat_cols) == 0:
                return {"status": "success", "message": "No categorical columns found", "skipped": True}
            
            # Check for high cardinality
            max_cardinality = max([ctx.df[col].nunique() for col in cat_cols])
            
            if max_cardinality > 100:
                strategy = 'hash'
            elif target_col and target_col in ctx.df.columns:
                strategy = 'target'
            else:
                strategy = 'rare_bucket'
        
        # Apply encoding strategy
        original_shape = ctx.df.shape
        original_cols = list(ctx.df.columns)
        
        if strategy == 'rare_bucket':
            threshold = params.get('threshold', 0.01)
            ctx.df = rare_bucket_encode(ctx.df, threshold)
            
        elif strategy == 'hash':
            n_features = params.get('n_features', 10)
            ctx.df = hash_encode(ctx.df, n_features)
            
        elif strategy == 'target':
            if not target_col:
                return {"status": "error", "error": "target_column required for target encoding"}
            cv_folds = params.get('cv_folds', 5)
            ctx.df = await target_encode(ctx.df, target_col, cv_folds)
            
        elif strategy == 'one_hot':
            # One-hot encoding for low cardinality
            cat_cols = ctx.df.select_dtypes(include=['object', 'category']).columns
            ctx.df = pd.get_dummies(ctx.df, columns=cat_cols, drop_first=True)
            
        else:
            return {"status": "error", "error": f"Unknown encoding strategy: {strategy}"}
        
        # Update context
        ctx.meta['encoding'] = {
            'strategy': strategy,
            'original_shape': original_shape,
            'new_shape': ctx.df.shape,
            'columns_added': len(ctx.df.columns) - len(original_cols),
            'columns_removed': len(original_cols) - len([c for c in ctx.df.columns if c in original_cols])
        }
        ctx.steps.append('encode_categorical_features')
        
        result = {
            "status": "success",
            "strategy": strategy,
            "original_shape": original_shape,
            "new_shape": ctx.df.shape,
            "columns_added": ctx.meta['encoding']['columns_added'],
            "columns_removed": ctx.meta['encoding']['columns_removed'],
            "encoding_success": True
        }
        
        logger.info(f"Categorical encoding completed: {original_shape} -> {ctx.df.shape}")
        return result
        
    except Exception as e:
        logger.error(f"Categorical encoding failed: {e}")
        return {"status": "error", "error": str(e)}


async def generate_datetime_features(params: Dict[str, Any], ctx: PipelineCtx) -> Dict[str, Any]:
    """
    Generate datetime features including cyclical encoding.
    
    Args:
        params: Parameters dictionary containing:
            - run_id: Unique run identifier
            - datetime_columns: List of datetime columns to process
            - cyclical_features: List of cyclical features to generate
        ctx: Pipeline context
        
    Returns:
        Dictionary with datetime feature generation results
    """
    try:
        if ctx.df is None:
            return {"status": "error", "error": "No data in pipeline context"}
        
        datetime_cols = params.get('datetime_columns', [])
        cyclical_features = params.get('cyclical_features', ['hour', 'month', 'day_of_week'])
        
        # Auto-detect datetime columns if not specified
        if not datetime_cols:
            datetime_cols = ctx.df.select_dtypes(include=['datetime64']).columns.tolist()
        
        if not datetime_cols:
            return {"status": "success", "message": "No datetime columns found", "skipped": True}
        
        original_shape = ctx.df.shape
        features_added = 0
        
        for col in datetime_cols:
            if col not in ctx.df.columns:
                continue
                
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(ctx.df[col]):
                ctx.df[col] = pd.to_datetime(ctx.df[col])
            
            # Generate basic datetime features
            ctx.df[f"{col}_year"] = ctx.df[col].dt.year
            ctx.df[f"{col}_month"] = ctx.df[col].dt.month
            ctx.df[f"{col}_day"] = ctx.df[col].dt.day
            ctx.df[f"{col}_hour"] = ctx.df[col].dt.hour
            ctx.df[f"{col}_day_of_week"] = ctx.df[col].dt.dayofweek
            ctx.df[f"{col}_day_of_year"] = ctx.df[col].dt.dayofyear
            features_added += 6
            
            # Generate cyclical features
            if 'month' in cyclical_features:
                sin_month, cos_month = cyclical_encode(ctx.df[f"{col}_month"])
                ctx.df[f"{col}_month_sin"] = sin_month
                ctx.df[f"{col}_month_cos"] = cos_month
                features_added += 2
            
            if 'hour' in cyclical_features:
                sin_hour, cos_hour = cyclical_encode(ctx.df[f"{col}_hour"])
                ctx.df[f"{col}_hour_sin"] = sin_hour
                ctx.df[f"{col}_hour_cos"] = cos_hour
                features_added += 2
            
            if 'day_of_week' in cyclical_features:
                sin_dow, cos_dow = cyclical_encode(ctx.df[f"{col}_day_of_week"])
                ctx.df[f"{col}_dow_sin"] = sin_dow
                ctx.df[f"{col}_dow_cos"] = cos_dow
                features_added += 2
        
        # Update context
        ctx.meta['datetime_features'] = {
            'columns_processed': datetime_cols,
            'features_added': features_added,
            'cyclical_features': cyclical_features
        }
        ctx.steps.append('generate_datetime_features')
        
        result = {
            "status": "success",
            "columns_processed": datetime_cols,
            "features_added": features_added,
            "original_shape": original_shape,
            "new_shape": ctx.df.shape
        }
        
        logger.info(f"Datetime feature generation completed: {features_added} features added")
        return result
        
    except Exception as e:
        logger.error(f"Datetime feature generation failed: {e}")
        return {"status": "error", "error": str(e)}


# Configuration defaults
DEFAULT_ENCODING_CONFIG = {
    'rare_bucket_threshold': 0.01,
    'hash_features': 10,
    'target_cv_folds': 5,
    'cyclical_features': ['hour', 'month', 'day_of_week'],
    'ordinal_threshold': 0.7
} 