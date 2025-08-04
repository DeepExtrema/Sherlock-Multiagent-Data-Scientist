"""
Feature Pruning Module for Feature Engineering

This module provides feature pruning techniques including VIF (Variance Inflation Factor)
pruning for multicollinear features and other feature selection methods.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from .context import PipelineCtx

logger = logging.getLogger(__name__)


def compute_vif(df: pd.DataFrame, threshold: float = 5.0) -> List[str]:
    """
    Compute Variance Inflation Factor (VIF) for numeric features.
    
    Args:
        df: DataFrame with numeric features
        threshold: VIF threshold above which features are considered multicollinear
        
    Returns:
        List of column names with high VIF
    """
    try:
        # Select only numeric features
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return []
        
        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_df.columns)
        
        high_vif_cols = []
        
        for col in scaled_df.columns:
            # Use other features to predict this feature
            X = scaled_df.drop(columns=[col])
            y = scaled_df[col]
            
            # Fit linear regression
            lr = LinearRegression()
            lr.fit(X, y)
            
            # Calculate R-squared
            r_squared = lr.score(X, y)
            
            # Calculate VIF
            if r_squared == 1.0:
                vif = float('inf')
            else:
                vif = 1 / (1 - r_squared)
            
            if vif > threshold:
                high_vif_cols.append(col)
                logger.info(f"High VIF detected for '{col}': {vif:.2f}")
        
        return high_vif_cols
        
    except Exception as e:
        logger.error(f"VIF computation failed: {e}")
        return []


def remove_low_variance_features(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """
    Remove features with low variance.
    
    Args:
        df: DataFrame to process
        threshold: Variance threshold (features with variance below this are removed)
        
    Returns:
        DataFrame with low variance features removed
    """
    try:
        # Select numeric features
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) == 0:
            return df
        
        # Apply variance threshold
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(numeric_df)
        
        # Get selected features
        selected_features = numeric_df.columns[selector.get_support()].tolist()
        removed_features = [col for col in numeric_df.columns if col not in selected_features]
        
        if removed_features:
            logger.info(f"Removed {len(removed_features)} low variance features: {removed_features}")
        
        # Return DataFrame with only selected numeric features and all non-numeric features
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        result_df = df[selected_features + list(non_numeric_cols)]
        
        return result_df
        
    except Exception as e:
        logger.error(f"Low variance feature removal failed: {e}")
        return df


def select_features_by_mutual_info(df: pd.DataFrame, target_col: str, k: int = 10) -> List[str]:
    """
    Select features using mutual information.
    
    Args:
        df: DataFrame with features
        target_col: Target column name
        k: Number of top features to select
        
    Returns:
        List of selected feature names
    """
    try:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df[target_col]
        
        if len(X.columns) == 0:
            return []
        
        # Determine if target is categorical or continuous
        if y.dtype == 'object' or y.dtype.name == 'category':
            # Classification problem
            mi_scores = mutual_info_classif(X, y, random_state=42)
        else:
            # Regression problem
            mi_scores = mutual_info_regression(X, y, random_state=42)
        
        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'mutual_info': mi_scores
        }).sort_values('mutual_info', ascending=False)
        
        # Select top k features
        selected_features = feature_importance.head(k)['feature'].tolist()
        
        logger.info(f"Selected {len(selected_features)} features using mutual information")
        return selected_features
        
    except Exception as e:
        logger.error(f"Mutual information feature selection failed: {e}")
        return []


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


async def prune_multicollinear_features(params: Dict[str, Any], ctx: PipelineCtx, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove multicollinear features using VIF.
    
    Args:
        params: Parameters dictionary containing:
            - run_id: Unique run identifier
            - vif_threshold: VIF threshold for multicollinearity detection
            - method: Pruning method ('vif', 'variance', 'mutual_info', 'all')
            - target_column: Optional target column for mutual info selection
            - k_features: Number of features to select (for mutual info)
        ctx: Pipeline context
        config: Configuration dictionary
        
    Returns:
        Dictionary with pruning results
    """
    try:
        if ctx.df is None:
            return {"status": "error", "error": "No data in pipeline context"}
        
        method = params.get('method', 'vif')
        original_shape = ctx.df.shape
        original_cols = list(ctx.df.columns)
        removed_cols = []
        
        if method in ['vif', 'all']:
            # VIF pruning
            vif_threshold = params.get('vif_threshold', 5.0)
            high_vif_cols = await aio_run(compute_vif, ctx.df, vif_threshold)
            
            if high_vif_cols:
                ctx.df = ctx.df.drop(columns=high_vif_cols)
                removed_cols.extend(high_vif_cols)
                logger.info(f"Removed {len(high_vif_cols)} high VIF features")
        
        if method in ['variance', 'all']:
            # Variance threshold pruning
            variance_threshold = params.get('variance_threshold', 0.01)
            ctx.df = await aio_run(remove_low_variance_features, ctx.df, variance_threshold)
            
            # Calculate removed columns
            current_cols = list(ctx.df.columns)
            variance_removed = [col for col in original_cols if col not in current_cols and col not in removed_cols]
            removed_cols.extend(variance_removed)
        
        if method in ['mutual_info', 'all']:
            # Mutual information feature selection
            target_col = params.get('target_column')
            if target_col and target_col in ctx.df.columns:
                k_features = params.get('k_features', 10)
                selected_features = await aio_run(select_features_by_mutual_info, ctx.df, target_col, k_features)
                
                if selected_features:
                    # Keep selected features plus target and non-numeric features
                    non_numeric_cols = ctx.df.select_dtypes(exclude=[np.number]).columns
                    keep_cols = selected_features + list(non_numeric_cols)
                    keep_cols = [col for col in keep_cols if col in ctx.df.columns]
                    
                    mi_removed = [col for col in ctx.df.columns if col not in keep_cols]
                    ctx.df = ctx.df[keep_cols]
                    removed_cols.extend(mi_removed)
        
        # Update context
        ctx.meta['pruning'] = {
            'method': method,
            'original_shape': original_shape,
            'new_shape': ctx.df.shape,
            'removed_columns': removed_cols,
            'columns_removed': len(removed_cols)
        }
        ctx.steps.append('prune_multicollinear_features')
        
        result = {
            "status": "success",
            "method": method,
            "original_shape": original_shape,
            "new_shape": ctx.df.shape,
            "columns_removed": len(removed_cols),
            "removed_columns": removed_cols,
            "pruning_success": True
        }
        
        logger.info(f"Feature pruning completed: {original_shape} -> {ctx.df.shape}")
        return result
        
    except Exception as e:
        logger.error(f"Feature pruning failed: {e}")
        return {"status": "error", "error": str(e)}


def generate_interaction_features(df: pd.DataFrame, degree: int = 2, max_interactions: int = 100) -> pd.DataFrame:
    """
    Generate polynomial interaction features.
    
    Args:
        df: DataFrame with numeric features
        degree: Polynomial degree (default: 2 for pairwise interactions)
        max_interactions: Maximum number of interaction features to generate
        
    Returns:
        DataFrame with interaction features added
    """
    try:
        # Select numeric features
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return df
        
        result_df = df.copy()
        interaction_count = 0
        
        if degree == 2:
            # Generate pairwise interactions
            for i, col1 in enumerate(numeric_df.columns):
                for col2 in numeric_df.columns[i+1:]:
                    if interaction_count >= max_interactions:
                        break
                    
                    interaction_name = f"{col1}_x_{col2}"
                    result_df[interaction_name] = numeric_df[col1] * numeric_df[col2]
                    interaction_count += 1
                    
                    logger.debug(f"Generated interaction: {interaction_name}")
                
                if interaction_count >= max_interactions:
                    break
        
        logger.info(f"Generated {interaction_count} interaction features")
        return result_df
        
    except Exception as e:
        logger.error(f"Interaction feature generation failed: {e}")
        return df


async def generate_interactions(params: Dict[str, Any], ctx: PipelineCtx) -> Dict[str, Any]:
    """
    Generate interaction features and optionally prune multicollinear ones.
    
    Args:
        params: Parameters dictionary containing:
            - run_id: Unique run identifier
            - degree: Polynomial degree for interactions
            - max_interactions: Maximum number of interactions to generate
            - prune_after: Whether to prune multicollinear features after generation
            - vif_threshold: VIF threshold for pruning
        ctx: Pipeline context
        
    Returns:
        Dictionary with interaction generation results
    """
    try:
        if ctx.df is None:
            return {"status": "error", "error": "No data in pipeline context"}
        
        degree = params.get('degree', 2)
        max_interactions = params.get('max_interactions', 100)
        prune_after = params.get('prune_after', True)
        vif_threshold = params.get('vif_threshold', 5.0)
        
        original_shape = ctx.df.shape
        
        # Generate interaction features
        ctx.df = await aio_run(generate_interaction_features, ctx.df, degree, max_interactions)
        
        # Prune multicollinear features if requested
        if prune_after:
            high_vif_cols = await aio_run(compute_vif, ctx.df, vif_threshold)
            if high_vif_cols:
                ctx.df = ctx.df.drop(columns=high_vif_cols)
                logger.info(f"Removed {len(high_vif_cols)} high VIF features after interaction generation")
        
        # Update context
        ctx.meta['interactions'] = {
            'degree': degree,
            'max_interactions': max_interactions,
            'prune_after': prune_after,
            'original_shape': original_shape,
            'new_shape': ctx.df.shape,
            'features_added': ctx.df.shape[1] - original_shape[1]
        }
        ctx.steps.append('generate_interactions')
        
        result = {
            "status": "success",
            "degree": degree,
            "original_shape": original_shape,
            "new_shape": ctx.df.shape,
            "features_added": ctx.df.shape[1] - original_shape[1],
            "prune_after": prune_after
        }
        
        logger.info(f"Interaction generation completed: {original_shape} -> {ctx.df.shape}")
        return result
        
    except Exception as e:
        logger.error(f"Interaction generation failed: {e}")
        return {"status": "error", "error": str(e)}


# Configuration defaults
DEFAULT_PRUNING_CONFIG = {
    'vif_threshold': 5.0,
    'variance_threshold': 0.01,
    'k_features': 10,
    'max_interactions': 100,
    'prune_after_interactions': True
} 