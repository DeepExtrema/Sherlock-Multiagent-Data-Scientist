"""
Enhanced Feature Engineering Module

This module provides advanced feature engineering capabilities including
imputation, encoding, pruning, and pipeline context management.
"""

from .context import PipelineCtx, PipelineContextManager, get_ctx, save_ctx, delete_ctx
from .imputation import impute_missing_values, detect_missing_pattern, choose_imputer
from .encoding import (
    encode_categorical_features, generate_datetime_features,
    rare_bucket_encode, hash_encode, target_encode, cyclical_encode,
    detect_ordinal_categorical
)
from .pruning import (
    prune_multicollinear_features, generate_interactions,
    compute_vif, remove_low_variance_features, select_features_by_mutual_info
)
from .decorators import skip_if, track_metrics, validate_context

__all__ = [
    # Context management
    'PipelineCtx', 'PipelineContextManager', 'get_ctx', 'save_ctx', 'delete_ctx',
    
    # Imputation
    'impute_missing_values', 'detect_missing_pattern', 'choose_imputer',
    
    # Encoding
    'encode_categorical_features', 'generate_datetime_features',
    'rare_bucket_encode', 'hash_encode', 'target_encode', 'cyclical_encode',
    'detect_ordinal_categorical',
    
    # Pruning
    'prune_multicollinear_features', 'generate_interactions',
    'compute_vif', 'remove_low_variance_features', 'select_features_by_mutual_info',
    
    # Decorators
    'skip_if', 'track_metrics', 'validate_context'
] 