"""
Baseline Registry for Data Quality Comparisons

This module provides functionality to save and load baseline datasets
for data quality drift detection and schema consistency checks.
"""

import hashlib
import json
import pickle
from typing import Optional
import pandas as pd
import aioredis
from evidently.core import ColumnMapping


class BaselineRegistry:
    """Registry for managing baseline datasets for data quality comparisons."""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.baseline_prefix = "refinery:baseline:"
        self.mapping_prefix = "refinery:mapping:"
    
    async def save(self, df: pd.DataFrame, target_col: Optional[str] = None) -> str:
        """
        Save a baseline dataset and return its unique identifier.
        
        Args:
            df: DataFrame to save as baseline
            target_col: Optional target column name
            
        Returns:
            baseline_id: Unique hash identifier for the baseline
        """
        # Generate baseline ID from data hash including size and schema
        import os
        import time
        
        # Create a more robust hash including file size and modification time
        data_content = df.to_parquet() if hasattr(df, 'to_parquet') else df.to_csv(index=False).encode()
        file_size = len(data_content)
        mod_time = int(time.time())
        schema_hash = hashlib.md5(str(df.dtypes.to_dict()).encode()).hexdigest()[:8]
        
        combined_hash = f"{file_size}_{mod_time}_{schema_hash}"
        data_hash = hashlib.sha256(combined_hash.encode()).hexdigest()[:16]
        
        baseline_id = f"baseline_{data_hash}"
        
        # Save DataFrame to Redis
        df_bytes = pickle.dumps(df)
        await self.redis.setex(
            f"{self.baseline_prefix}{baseline_id}",
            86400 * 30,  # 30 days TTL
            df_bytes
        )
        
        # Save column mapping
        mapping = self._build_column_mapping(df, target_col)
        mapping_dict = {
            'target': mapping.target,
            'numerical_features': mapping.numerical_features,
            'categorical_features': mapping.categorical_features,
            'datetime_features': mapping.datetime_features,
            'text_features': mapping.text_features
        }
        
        await self.redis.setex(
            f"{self.mapping_prefix}{baseline_id}",
            86400 * 30,  # 30 days TTL
            json.dumps(mapping_dict)
        )
        
        return baseline_id
    
    async def load(self, baseline_id: str) -> Optional[pd.DataFrame]:
        """
        Load a baseline dataset by its ID.
        
        Args:
            baseline_id: Unique identifier for the baseline
            
        Returns:
            DataFrame if found, None otherwise
        """
        baseline_key = f"{self.baseline_prefix}{baseline_id}"
        df_bytes = await self.redis.get(baseline_key)
        
        if df_bytes is None:
            return None
            
        return pickle.loads(df_bytes)
    
    async def load_mapping(self, baseline_id: str) -> Optional[ColumnMapping]:
        """
        Load column mapping for a baseline dataset.
        
        Args:
            baseline_id: Unique identifier for the baseline
            
        Returns:
            ColumnMapping if found, None otherwise
        """
        mapping_key = f"{self.mapping_prefix}{baseline_id}"
        mapping_json = await self.redis.get(mapping_key)
        
        if mapping_json is None:
            return None
            
        mapping_dict = json.loads(mapping_json)
        return ColumnMapping(**mapping_dict)
    
    def _build_column_mapping(self, df: pd.DataFrame, target: Optional[str] = None) -> ColumnMapping:
        """
        Build column mapping for a DataFrame.
        
        Args:
            df: DataFrame to analyze
            target: Optional target column name
            
        Returns:
            ColumnMapping object
        """
        numerical_features = df.select_dtypes(include=['number']).columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_features = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Remove target from feature lists if specified
        if target:
            for feature_list in [numerical_features, categorical_features, datetime_features]:
                if target in feature_list:
                    feature_list.remove(target)
        
        return ColumnMapping(
            target=target,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            datetime_features=datetime_features,
            text_features=[]  # Will be populated separately if needed
        )
    
    async def list_baselines(self) -> list[str]:
        """
        List all available baseline IDs.
        
        Returns:
            List of baseline IDs
        """
        keys = await self.redis.keys(f"{self.baseline_prefix}*")
        return [key.decode().replace(self.baseline_prefix, "") for key in keys]
    
    async def delete(self, baseline_id: str) -> bool:
        """
        Delete a baseline dataset.
        
        Args:
            baseline_id: Unique identifier for the baseline
            
        Returns:
            True if deleted, False if not found
        """
        baseline_key = f"{self.baseline_prefix}{baseline_id}"
        mapping_key = f"{self.mapping_prefix}{baseline_id}"
        
        deleted_baseline = await self.redis.delete(baseline_key)
        deleted_mapping = await self.redis.delete(mapping_key)
        
        return deleted_baseline > 0 or deleted_mapping > 0 