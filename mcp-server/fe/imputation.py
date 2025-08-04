"""
Advanced Imputation Module for Feature Engineering

This module provides advanced imputation strategies including KNN, MICE,
and automatic pattern detection for missing data.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from .context import PipelineCtx

logger = logging.getLogger(__name__)


def detect_missing_pattern(df: pd.DataFrame) -> str:
    """
    Detect missing data pattern (MCAR, MAR, MNAR).
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Pattern type: 'MCAR', 'MAR', or 'MNAR'
    """
    try:
        # Simple heuristic-based detection
        missing_data = df.isnull()
        
        if missing_data.sum().sum() == 0:
            return "NO_MISSING"
        
        # Calculate missing percentages per column
        missing_pct = missing_data.sum() / len(df)
        
        # Check for MCAR (Missing Completely At Random)
        # MCAR: missing values are randomly distributed
        if missing_pct.std() < 0.1:  # Low variance in missing rates
            return "MCAR"
        
        # Check for MAR (Missing At Random)
        # MAR: missingness depends on observed variables
        # Look for correlations between missing patterns and other variables
        missing_correlations = []
        for col in df.columns:
            if missing_data[col].sum() > 0:
                # Check correlation with other numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    other_cols = [c for c in numeric_cols if c != col]
                    if other_cols:
                        corr = missing_data[col].corr(df[other_cols].mean(axis=1))
                        missing_correlations.append(abs(corr))
        
        # If there are correlations with observed variables, likely MAR
        if missing_correlations and np.mean(missing_correlations) > 0.1:
            return "MAR"
        
        # Default to MNAR (Missing Not At Random)
        return "MNAR"
        
    except Exception as e:
        logger.warning(f"Error detecting missing pattern: {e}")
        return "MCAR"  # Default fallback


def choose_imputer(pattern: str, config: Dict[str, Any]) -> Any:
    """
    Choose appropriate imputer based on missing data pattern.
    
    Args:
        pattern: Missing data pattern ('MCAR', 'MAR', 'MNAR')
        config: Configuration dictionary
        
    Returns:
        Fitted imputer object
    """
    try:
        if pattern == "MCAR":
            # For MCAR, simple imputation is often sufficient
            strategy = config.get('mcar_strategy', 'median')
            return SimpleImputer(strategy=strategy)
        
        elif pattern == "MAR":
            # For MAR, KNN or MICE works well
            method = config.get('mar_method', 'knn')
            if method == 'knn':
                k = config.get('knn_k', 5)
                return KNNImputer(n_neighbors=k)
            else:
                max_iter = config.get('mice_max_iter', 10)
                return IterativeImputer(max_iter=max_iter, random_state=42)
        
        else:  # MNAR
            # For MNAR, MICE with more iterations
            max_iter = config.get('mnar_max_iter', 20)
            return IterativeImputer(max_iter=max_iter, random_state=42)
            
    except Exception as e:
        logger.error(f"Error choosing imputer: {e}")
        return SimpleImputer(strategy='median')  # Fallback


async def aio_run(func, *args, **kwargs):
    """
    Run CPU-intensive functions in thread pool.
    
    Args:
        func: Function to run
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args, **kwargs)


async def impute_missing_values(params: Dict[str, Any], ctx: PipelineCtx, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Advanced imputation with automatic pattern detection.
    
    Args:
        params: Parameters dictionary containing:
            - run_id: Unique run identifier
            - strategy: Imputation strategy ('auto', 'knn', 'mice', 'simple')
            - target_column: Optional target column name
        ctx: Pipeline context
        config: Configuration dictionary
        
    Returns:
        Dictionary with imputation results
    """
    try:
        if ctx.df is None:
            return {"status": "error", "error": "No data in pipeline context"}
        
        # Detect missing pattern
        pattern = detect_missing_pattern(ctx.df)
        logger.info(f"Detected missing pattern: {pattern}")
        
        # Choose imputation strategy
        strategy = params.get('strategy', 'auto')
        
        if strategy == 'auto':
            imputer = choose_imputer(pattern, config)
        elif strategy == 'knn':
            k = config.get('knn_k', 5)
            imputer = KNNImputer(n_neighbors=k)
        elif strategy == 'mice':
            max_iter = config.get('mice_max_iter', 10)
            imputer = IterativeImputer(max_iter=max_iter, random_state=42)
        elif strategy == 'simple':
            method = params.get('method', 'median')
            imputer = SimpleImputer(strategy=method)
        else:
            return {"status": "error", "error": f"Unknown strategy: {strategy}"}
        
        # Prepare data for imputation
        target_col = params.get('target_column')
        if target_col and target_col in ctx.df.columns:
            # Temporarily remove target column for imputation
            impute_data = ctx.df.drop(columns=[target_col])
            target_data = ctx.df[target_col]
        else:
            impute_data = ctx.df.copy()
            target_data = None
        
        # Count missing values before imputation
        missing_before = impute_data.isnull().sum().sum()
        
        # Run imputation
        logger.info(f"Running imputation with {imputer.__class__.__name__}")
        imputed_data = await aio_run(imputer.fit_transform, impute_data)
        
        # Convert back to DataFrame
        imputed_df = pd.DataFrame(imputed_data, columns=impute_data.columns, index=impute_data.index)
        
        # Restore target column if it was removed
        if target_data is not None:
            imputed_df[target_col] = target_data
        
        # Update context
        ctx.df = imputed_df
        ctx.meta['imputation'] = {
            'method': imputer.__class__.__name__,
            'pattern': pattern,
            'strategy': strategy,
            'missing_before': int(missing_before),
            'missing_after': int(imputed_df.isnull().sum().sum()),
            'params': imputer.get_params() if hasattr(imputer, 'get_params') else {}
        }
        ctx.steps.append('impute_missing_values')
        
        # Calculate remaining nulls
        remaining_nulls = int(imputed_df.isnull().sum().sum())
        
        result = {
            "status": "success",
            "pattern": pattern,
            "method": imputer.__class__.__name__,
            "missing_before": int(missing_before),
            "missing_after": remaining_nulls,
            "remaining_nulls": remaining_nulls,
            "imputation_success": remaining_nulls == 0
        }
        
        logger.info(f"Imputation completed: {missing_before} -> {remaining_nulls} missing values")
        return result
        
    except Exception as e:
        logger.error(f"Imputation failed: {e}")
        return {"status": "error", "error": str(e)}


def validate_imputation_result(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate imputation results.
    
    Args:
        df: DataFrame after imputation
        target_col: Optional target column name
        
    Returns:
        Validation results dictionary
    """
    try:
        validation = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': int(df.isnull().sum().sum()),
            'missing_percentage': float(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
        }
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            infinite_count = np.isinf(df[numeric_cols]).sum().sum()
            validation['infinite_values'] = int(infinite_count)
        
        # Check target column if specified
        if target_col and target_col in df.columns:
            target_missing = df[target_col].isnull().sum()
            validation['target_missing'] = int(target_missing)
            validation['target_missing_percentage'] = float(target_missing / len(df) * 100)
        
        return validation
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {"error": str(e)}


# Configuration defaults
DEFAULT_IMPUTATION_CONFIG = {
    'mcar_strategy': 'median',
    'mar_method': 'knn',
    'knn_k': 5,
    'mice_max_iter': 10,
    'mnar_max_iter': 20,
    'validation_enabled': True
} 