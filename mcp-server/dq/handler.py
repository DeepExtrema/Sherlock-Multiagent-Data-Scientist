"""
Enhanced Data Quality Handler with Evidently Integration

This module provides enhanced data quality checks using Evidently's
statistical tests and advanced monitoring capabilities.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from evidently import ColumnMapping
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestColumnPresence, TestColumnType, TestShareOfMissingValues,
    TestNumMissingValues, TestColumnValueRange, TestColumnQuantile,
    TestShareOfOutliers, TestTargetShare, TestDatetimeColumnOutOfRange,
    TestColumnFeatureCorrelation, TestSimpleTargetLeakage
)
from evidently.test_preset import DataQualityPreset, DataDriftPreset, TargetDriftPreset
from evidently.metric_preset import DataQualityPreset as DataQualityMetricPreset
from evidently.metric_preset import DataDriftPreset as DataDriftMetricPreset

from .baseline import BaselineRegistry

logger = logging.getLogger(__name__)


class EnhancedDataQualityHandler:
    """Enhanced data quality handler with Evidently integration."""
    
    def __init__(self, baseline_registry: BaselineRegistry, config: Dict[str, Any]):
        self.baseline_registry = baseline_registry
        self.config = config.get('dq', {})
        self.sample_rows = self.config.get('sample_rows', 10000)
        self.outlier_threshold = self.config.get('outlier_threshold', 0.05)
        self.drift_threshold = self.config.get('drift_threshold', 0.05)
        self.critical_columns = self.config.get('critical_columns', [])
    
    async def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load data with sampling for large datasets."""
        try:
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path, nrows=self.sample_rows)
            elif data_path.endswith('.parquet'):
                df = pd.read_parquet(data_path)
                if len(df) > self.sample_rows:
                    df = df.sample(n=self.sample_rows, random_state=42)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
            
            logger.info(f"Loaded {len(df)} rows from {data_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load data from {data_path}: {e}")
            raise
    
    def _build_column_mapping(self, df: pd.DataFrame, target: Optional[str] = None) -> ColumnMapping:
        """Build column mapping for Evidently."""
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
            text_features=[]
        )
    
    async def _run_evidently_suite(self, suite: TestSuite, current: pd.DataFrame, 
                                 reference: Optional[pd.DataFrame] = None,
                                 mapping: Optional[ColumnMapping] = None) -> Dict[str, Any]:
        """Run Evidently test suite and return results."""
        try:
            suite.run(
                reference_data=reference,
                current_data=current,
                column_mapping=mapping
            )
            return suite.as_dict()
        except Exception as e:
            logger.error(f"Evidently suite execution failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _shrink_big_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Remove large data chunks from result to keep response size manageable."""
        if isinstance(result, dict):
            # Remove sample data and large chunks
            keys_to_remove = ['current_data', 'reference_data', 'data', 'details']
            for key in keys_to_remove:
                result.pop(key, None)
            
            # Recursively clean nested dictionaries
            for key, value in result.items():
                if isinstance(value, dict):
                    result[key] = self._shrink_big_dict(value)
                elif isinstance(value, list) and len(value) > 10:
                    result[key] = value[:10] + [f"... and {len(value) - 10} more items"]
        
        return result
    
    async def check_schema_consistency(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced schema consistency check with baseline comparison.
        
        Args:
            params: Dictionary containing:
                - data_path: Path to current data
                - baseline_id: Optional baseline ID for comparison
                - run_id: Unique run identifier
                
        Returns:
            Dictionary with schema consistency results
        """
        try:
            current_df = await self._load_data(params['data_path'])
            baseline_id = params.get('baseline_id')
            
            # Load baseline if provided
            reference_df = None
            if baseline_id:
                reference_df = await self.baseline_registry.load(baseline_id)
                if reference_df is None:
                    logger.warning(f"Baseline {baseline_id} not found")
            
            # Build column mapping
            mapping = self._build_column_mapping(current_df)
            
            # Define tests
            tests = [
                TestColumnPresence(columns=current_df.columns.tolist())
            ]
            
            # Add column type tests
            for col in current_df.columns:
                expected_type = str(current_df[col].dtype)
                tests.append(TestColumnType(column=col, expected_type=expected_type))
            
            # Add baseline comparison tests if available
            if reference_df is not None:
                tests.extend([
                    TestColumnPresence(columns=reference_df.columns.tolist()),
                    TestColumnType(columns=reference_df.columns.tolist())
                ])
            
            # Run test suite
            suite = TestSuite(tests)
            result = await self._run_evidently_suite(suite, current_df, reference_df, mapping)
            
            # Add custom metrics
            result['schema_metrics'] = {
                'total_columns': len(current_df.columns),
                'numeric_columns': len(mapping.numerical_features),
                'categorical_columns': len(mapping.categorical_features),
                'datetime_columns': len(mapping.datetime_features),
                'baseline_comparison': baseline_id is not None
            }
            
            return self._shrink_big_dict(result)
            
        except Exception as e:
            logger.error(f"Schema consistency check failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def check_missing_values(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced missing values check with row-level analysis.
        
        Args:
            params: Dictionary containing:
                - data_path: Path to current data
                - baseline_id: Optional baseline ID for comparison
                - run_id: Unique run identifier
                - critical_columns: List of critical columns for alerts
                
        Returns:
            Dictionary with missing values analysis
        """
        try:
            current_df = await self._load_data(params['data_path'])
            baseline_id = params.get('baseline_id')
            critical_columns = params.get('critical_columns', self.critical_columns)
            
            # Load baseline if provided
            reference_df = None
            if baseline_id:
                reference_df = await self.baseline_registry.load(baseline_id)
            
            # Build column mapping
            mapping = self._build_column_mapping(current_df)
            
            # Define tests
            tests = [
                TestShareOfMissingValues(columns=current_df.columns.tolist()),
                TestNumMissingValues(columns=current_df.columns.tolist())
            ]
            
            # Add critical column tests
            for col in critical_columns:
                if col in current_df.columns:
                    tests.append(TestShareOfMissingValues(column=col, lt=0.3))  # Alert if >30% missing
            
            # Run test suite
            suite = TestSuite(tests)
            result = await self._run_evidently_suite(suite, current_df, reference_df, mapping)
            
            # Add row-level analysis
            row_null_density = current_df.isnull().sum(axis=1) / len(current_df.columns)
            high_null_rows = (row_null_density > 0.3).sum()
            
            result['missing_analysis'] = {
                'total_rows': len(current_df),
                'high_null_rows': int(high_null_rows),
                'high_null_percentage': float(high_null_rows / len(current_df) * 100),
                'row_null_density_stats': {
                    'mean': float(row_null_density.mean()),
                    'std': float(row_null_density.std()),
                    'max': float(row_null_density.max())
                },
                'critical_columns': critical_columns
            }
            
            return self._shrink_big_dict(result)
            
        except Exception as e:
            logger.error(f"Missing values check failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def check_distributions(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced distribution analysis with drift detection.
        
        Args:
            params: Dictionary containing:
                - data_path: Path to current data
                - baseline_id: Baseline ID for comparison (required)
                - run_id: Unique run identifier
                
        Returns:
            Dictionary with distribution and drift analysis
        """
        try:
            current_df = await self._load_data(params['data_path'])
            baseline_id = params.get('baseline_id')
            
            if not baseline_id:
                return {"status": "error", "error": "baseline_id is required for distribution analysis"}
            
            reference_df = await self.baseline_registry.load(baseline_id)
            if reference_df is None:
                return {"status": "error", "error": f"Baseline {baseline_id} not found"}
            
            # Build column mapping
            mapping = self._build_column_mapping(current_df)
            
            # 1. Data drift detection
            drift_suite = DataDriftPreset()
            drift_result = await self._run_evidently_suite(
                drift_suite, current_df, reference_df, mapping
            )
            
            # 2. Outlier detection
            outlier_tests = []
            for col in mapping.numerical_features:
                outlier_tests.append(TestShareOfOutliers(
                    column=col, 
                    lt=self.outlier_threshold
                ))
            
            outlier_suite = TestSuite(outlier_tests)
            outlier_result = await self._run_evidently_suite(
                outlier_suite, current_df, None, mapping
            )
            
            # 3. Calculate drift rate
            drift_rate = 0.0
            if drift_result.get('summary', {}).get('all_passed') is False:
                failed_tests = drift_result.get('summary', {}).get('failed_tests', 0)
                total_tests = drift_result.get('summary', {}).get('total_tests', 1)
                drift_rate = failed_tests / total_tests
            
            result = {
                "status": "success",
                "drift_analysis": self._shrink_big_dict(drift_result),
                "outlier_analysis": self._shrink_big_dict(outlier_result),
                "drift_rate": drift_rate,
                "baseline_id": baseline_id,
                "run_id": params.get('run_id')
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Distribution check failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def check_duplicates(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced duplicate detection with correlation analysis moved to FE.
        
        Args:
            params: Dictionary containing:
                - data_path: Path to current data
                - run_id: Unique run identifier
                
        Returns:
            Dictionary with duplicate analysis
        """
        try:
            current_df = await self._load_data(params['data_path'])
            
            # Check for exact duplicates
            exact_duplicates = current_df.duplicated().sum()
            duplicate_percentage = exact_duplicates / len(current_df) * 100
            
            # Check for duplicates by key columns (if specified)
            key_columns = params.get('key_columns', [])
            key_duplicates = 0
            if key_columns:
                key_duplicates = current_df.duplicated(subset=key_columns).sum()
            
            result = {
                "status": "success",
                "duplicate_analysis": {
                    "total_rows": len(current_df),
                    "exact_duplicates": int(exact_duplicates),
                    "exact_duplicate_percentage": float(duplicate_percentage),
                    "key_columns": key_columns,
                    "key_duplicates": int(key_duplicates) if key_columns else None,
                    "recommendation": "remove_duplicates" if exact_duplicates > 0 else "no_action"
                },
                "run_id": params.get('run_id')
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Duplicate check failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def check_leakage(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced leakage detection with target correlation and simple leakage tests.
        
        Args:
            params: Dictionary containing:
                - data_path: Path to current data
                - target_column: Target column name
                - run_id: Unique run identifier
                
        Returns:
            Dictionary with leakage analysis
        """
        try:
            current_df = await self._load_data(params['data_path'])
            target_column = params.get('target_column')
            
            if not target_column:
                return {"status": "error", "error": "target_column is required for leakage detection"}
            
            if target_column not in current_df.columns:
                return {"status": "error", "error": f"Target column {target_column} not found in data"}
            
            # Build column mapping
            mapping = self._build_column_mapping(current_df, target_column)
            
            # Define tests
            tests = [
                TestColumnFeatureCorrelation(column=target_column),
                TestSimpleTargetLeakage(column=target_column)
            ]
            
            # Run test suite
            suite = TestSuite(tests)
            result = await self._run_evidently_suite(suite, current_df, None, mapping)
            
            # Add custom leakage analysis
            target_correlations = {}
            for col in mapping.numerical_features:
                if col != target_column:
                    correlation = current_df[col].corr(current_df[target_column])
                    if abs(correlation) > 0.8:  # High correlation threshold
                        target_correlations[col] = float(correlation)
            
            result['leakage_analysis'] = {
                'target_column': target_column,
                'high_correlation_features': target_correlations,
                'suspicious_features': list(target_correlations.keys()),
                'recommendation': 'investigate_leakage' if target_correlations else 'no_leakage_detected'
            }
            
            return self._shrink_big_dict(result)
            
        except Exception as e:
            logger.error(f"Leakage check failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def check_drift(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced drift detection with time-windowed history.
        
        Args:
            params: Dictionary containing:
                - data_path: Path to current data
                - baseline_id: Baseline ID for comparison
                - run_id: Unique run identifier
                - drift_type: Type of drift to check ('data', 'target', 'both')
                
        Returns:
            Dictionary with drift analysis
        """
        try:
            current_df = await self._load_data(params['data_path'])
            baseline_id = params.get('baseline_id')
            drift_type = params.get('drift_type', 'both')
            
            if not baseline_id:
                return {"status": "error", "error": "baseline_id is required for drift detection"}
            
            reference_df = await self.baseline_registry.load(baseline_id)
            if reference_df is None:
                return {"status": "error", "error": f"Baseline {baseline_id} not found"}
            
            # Build column mapping
            mapping = await self.baseline_registry.load_mapping(baseline_id)
            if mapping is None:
                mapping = self._build_column_mapping(current_df)
            
            results = {}
            
            # Data drift detection
            if drift_type in ['data', 'both']:
                data_drift_suite = DataDriftPreset()
                data_drift_result = await self._run_evidently_suite(
                    data_drift_suite, current_df, reference_df, mapping
                )
                results['data_drift'] = self._shrink_big_dict(data_drift_result)
            
            # Target drift detection
            if drift_type in ['target', 'both'] and mapping.target:
                target_drift_suite = TargetDriftPreset()
                target_drift_result = await self._run_evidently_suite(
                    target_drift_suite, current_df, reference_df, mapping
                )
                results['target_drift'] = self._shrink_big_dict(target_drift_result)
            
            # Calculate overall drift metrics
            drift_metrics = {
                'baseline_id': baseline_id,
                'drift_type': drift_type,
                'current_data_shape': current_df.shape,
                'reference_data_shape': reference_df.shape,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            result = {
                "status": "success",
                "drift_results": results,
                "drift_metrics": drift_metrics,
                "run_id": params.get('run_id')
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Drift check failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def check_freshness(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check data freshness and latency.
        
        Args:
            params: Dictionary containing:
                - data_path: Path to current data
                - datetime_column: Column containing timestamps
                - max_age_hours: Maximum acceptable age in hours
                - run_id: Unique run identifier
                
        Returns:
            Dictionary with freshness analysis
        """
        try:
            current_df = await self._load_data(params['data_path'])
            datetime_column = params.get('datetime_column')
            max_age_hours = params.get('max_age_hours', 24)
            
            if not datetime_column:
                return {"status": "error", "error": "datetime_column is required for freshness check"}
            
            if datetime_column not in current_df.columns:
                return {"status": "error", "error": f"Datetime column {datetime_column} not found"}
            
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(current_df[datetime_column]):
                current_df[datetime_column] = pd.to_datetime(current_df[datetime_column])
            
            # Calculate freshness metrics
            latest_timestamp = current_df[datetime_column].max()
            earliest_timestamp = current_df[datetime_column].min()
            current_time = pd.Timestamp.now()
            
            age_hours = (current_time - latest_timestamp).total_seconds() / 3600
            data_span_hours = (latest_timestamp - earliest_timestamp).total_seconds() / 3600
            
            # Check for out-of-range timestamps
            tests = [
                TestDatetimeColumnOutOfRange(
                    column=datetime_column,
                    max_age_hours=max_age_hours
                )
            ]
            
            suite = TestSuite(tests)
            result = await self._run_evidently_suite(suite, current_df, None, None)
            
            result['freshness_analysis'] = {
                'datetime_column': datetime_column,
                'latest_timestamp': latest_timestamp.isoformat(),
                'earliest_timestamp': earliest_timestamp.isoformat(),
                'current_time': current_time.isoformat(),
                'age_hours': float(age_hours),
                'data_span_hours': float(data_span_hours),
                'max_age_hours': max_age_hours,
                'is_fresh': age_hours <= max_age_hours,
                'recommendation': 'data_fresh' if age_hours <= max_age_hours else 'data_stale'
            }
            
            return self._shrink_big_dict(result)
            
        except Exception as e:
            logger.error(f"Freshness check failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def check_target_balance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check target class balance for classification tasks.
        
        Args:
            params: Dictionary containing:
                - data_path: Path to current data
                - target_column: Target column name
                - run_id: Unique run identifier
                
        Returns:
            Dictionary with target balance analysis
        """
        try:
            current_df = await self._load_data(params['data_path'])
            target_column = params.get('target_column')
            
            if not target_column:
                return {"status": "error", "error": "target_column is required for target balance check"}
            
            if target_column not in current_df.columns:
                return {"status": "error", "error": f"Target column {target_column} not found"}
            
            # Build column mapping
            mapping = self._build_column_mapping(current_df, target_column)
            
            # Run target share test
            tests = [TestTargetShare(column=target_column)]
            suite = TestSuite(tests)
            result = await self._run_evidently_suite(suite, current_df, None, mapping)
            
            # Add custom balance analysis
            target_counts = current_df[target_column].value_counts()
            total_samples = len(current_df)
            
            balance_analysis = {
                'target_column': target_column,
                'total_samples': total_samples,
                'class_counts': target_counts.to_dict(),
                'class_percentages': (target_counts / total_samples * 100).to_dict(),
                'min_class_percentage': float((target_counts / total_samples * 100).min()),
                'max_class_percentage': float((target_counts / total_samples * 100).max()),
                'is_balanced': len(target_counts) > 1 and (target_counts / total_samples * 100).min() > 5
            }
            
            result['balance_analysis'] = balance_analysis
            
            return self._shrink_big_dict(result)
            
        except Exception as e:
            logger.error(f"Target balance check failed: {e}")
            return {"status": "error", "error": str(e)} 