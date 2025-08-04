#!/usr/bin/env python3
"""
End-to-End Workflow Test for Refinery Agent
Tests complete DQ and FE pipeline workflows.
"""

import asyncio
import json
import logging
import time
import tempfile
import os
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np
import httpx
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RefineryE2ETest:
    """End-to-end test suite for refinery agent."""
    
    def __init__(self, agent_url: str = "http://localhost:8005"):
        self.agent_url = agent_url.rstrip('/')
        self.http_client = httpx.AsyncClient(timeout=300)
        self.test_data_dir = Path(tempfile.mkdtemp())
        
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.aclose()
        # Cleanup test data
        import shutil
        shutil.rmtree(self.test_data_dir, ignore_errors=True)
    
    def create_test_dataset(self, rows: int = 1000) -> pd.DataFrame:
        """Create a synthetic test dataset."""
        np.random.seed(42)
        
        # Generate synthetic data
        data = {
            'id': range(1, rows + 1),
            'numeric_feature_1': np.random.normal(100, 15, rows),
            'numeric_feature_2': np.random.exponential(50, rows),
            'categorical_feature': np.random.choice(['A', 'B', 'C', 'D'], rows),
            'text_feature': [f"Sample text {i} with some content" for i in range(rows)],
            'datetime_feature': pd.date_range('2023-01-01', periods=rows, freq='H'),
            'target': np.random.binomial(1, 0.3, rows)
        }
        
        df = pd.DataFrame(data)
        
        # Add some missing values
        df.loc[df.sample(frac=0.1).index, 'numeric_feature_1'] = np.nan
        df.loc[df.sample(frac=0.05).index, 'categorical_feature'] = np.nan
        
        # Add some duplicates
        df = pd.concat([df, df.sample(frac=0.02)], ignore_index=True)
        
        return df
    
    def create_reference_dataset(self, rows: int = 500) -> pd.DataFrame:
        """Create a reference dataset for drift detection."""
        np.random.seed(123)
        
        data = {
            'id': range(1, rows + 1),
            'numeric_feature_1': np.random.normal(95, 12, rows),  # Slightly different distribution
            'numeric_feature_2': np.random.exponential(45, rows),
            'categorical_feature': np.random.choice(['A', 'B', 'C', 'D'], rows),
            'text_feature': [f"Reference text {i}" for i in range(rows)],
            'datetime_feature': pd.date_range('2023-06-01', periods=rows, freq='H'),
            'target': np.random.binomial(1, 0.25, rows)
        }
        
        return pd.DataFrame(data)
    
    async def execute_task(self, task_id: str, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a refinery agent task."""
        request_data = {
            "task_id": task_id,
            "action": action,
            "params": params
        }
        
        try:
            response = await self.http_client.post(
                f"{self.agent_url}/execute",
                json=request_data
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            raise
    
    async def test_data_quality_workflow(self) -> bool:
        """Test complete data quality workflow."""
        logger.info("🧪 Testing Data Quality Workflow")
        
        # Create test data
        df = self.create_test_dataset(1000)
        data_path = self.test_data_dir / "test_data.parquet"
        df.to_parquet(data_path)
        
        # 1. Check schema consistency
        logger.info("  → Checking schema consistency")
        result = await self.execute_task(
            "dq_schema_check",
            "check_schema_consistency",
            {"data_path": str(data_path)}
        )
        
        if not result.get("success"):
            logger.error(f"Schema check failed: {result.get('error')}")
            return False
        
        schema_result = result.get("result", {})
        assert schema_result.get("status") == "pass", "Schema check should pass"
        logger.info(f"  ✅ Schema check passed: {len(schema_result.get('actual_schema', {}).get('columns', []))} columns")
        
        # 2. Check missing values
        logger.info("  → Checking missing values")
        result = await self.execute_task(
            "dq_missing_check",
            "check_missing_values",
            {"data_path": str(data_path), "threshold_pct": 0.5}
        )
        
        if not result.get("success"):
            logger.error(f"Missing values check failed: {result.get('error')}")
            return False
        
        missing_result = result.get("result", {})
        logger.info(f"  ✅ Missing values check: {missing_result.get('total_missing', 0)} total missing values")
        
        # 3. Check distributions
        logger.info("  → Checking distributions")
        result = await self.execute_task(
            "dq_distributions",
            "check_distributions",
            {"data_path": str(data_path)}
        )
        
        if not result.get("success"):
            logger.error(f"Distributions check failed: {result.get('error')}")
            return False
        
        dist_result = result.get("result", {})
        logger.info(f"  ✅ Distributions check: {len(dist_result.get('numeric_distributions', {}))} numeric features analyzed")
        
        # 4. Check duplicates
        logger.info("  → Checking duplicates")
        result = await self.execute_task(
            "dq_duplicates",
            "check_duplicates",
            {"data_path": str(data_path), "id_cols": ["id"]}
        )
        
        if not result.get("success"):
            logger.error(f"Duplicates check failed: {result.get('error')}")
            return False
        
        dup_result = result.get("result", {})
        logger.info(f"  ✅ Duplicates check: {dup_result.get('duplicate_rows', 0)} duplicate rows found")
        
        # 5. Check leakage
        logger.info("  → Checking data leakage")
        result = await self.execute_task(
            "dq_leakage",
            "check_leakage",
            {"data_path": str(data_path), "target_col": "target"}
        )
        
        if not result.get("success"):
            logger.error(f"Leakage check failed: {result.get('error')}")
            return False
        
        leakage_result = result.get("result", {})
        logger.info(f"  ✅ Leakage check: {len(leakage_result.get('suspicious_cols', []))} suspicious columns")
        
        # 6. Check drift
        logger.info("  → Checking data drift")
        ref_df = self.create_reference_dataset(500)
        ref_path = self.test_data_dir / "reference_data.parquet"
        ref_df.to_parquet(ref_path)
        
        result = await self.execute_task(
            "dq_drift",
            "check_drift",
            {"reference_path": str(ref_path), "current_path": str(data_path)}
        )
        
        if not result.get("success"):
            logger.error(f"Drift check failed: {result.get('error')}")
            return False
        
        drift_result = result.get("result", {})
        logger.info(f"  ✅ Drift check: {len(drift_result.get('columns_analyzed', []))} columns analyzed for drift")
        
        logger.info("🎉 Data Quality workflow completed successfully!")
        return True
    
    async def test_feature_engineering_workflow(self) -> bool:
        """Test complete feature engineering workflow."""
        logger.info("🧪 Testing Feature Engineering Workflow")
        
        # Create test data
        df = self.create_test_dataset(1000)
        data_path = self.test_data_dir / "fe_data.parquet"
        df.to_parquet(data_path)
        
        run_id = f"test_run_{int(time.time())}"
        
        # 1. Assign feature roles
        logger.info("  → Assigning feature roles")
        result = await self.execute_task(
            "fe_roles",
            "assign_feature_roles",
            {"input_path": str(data_path), "run_id": run_id}
        )
        
        if not result.get("success"):
            logger.error(f"Feature roles assignment failed: {result.get('error')}")
            return False
        
        roles_result = result.get("result", {})
        logger.info(f"  ✅ Feature roles assigned: {roles_result.get('feature_count', 0)} features")
        
        # 2. Impute missing values
        logger.info("  → Imputing missing values")
        result = await self.execute_task(
            "fe_impute",
            "impute_missing_values",
            {"input_path": str(data_path), "run_id": run_id, "strategy": "auto"}
        )
        
        if not result.get("success"):
            logger.error(f"Missing value imputation failed: {result.get('error')}")
            return False
        
        impute_result = result.get("result", {})
        logger.info(f"  ✅ Missing values imputed: {len(impute_result.get('imputed_columns', []))} columns")
        
        # 3. Scale numeric features
        logger.info("  → Scaling numeric features")
        result = await self.execute_task(
            "fe_scale",
            "scale_numeric_features",
            {"run_id": run_id, "method": "standard"}
        )
        
        if not result.get("success"):
            logger.error(f"Feature scaling failed: {result.get('error')}")
            return False
        
        scale_result = result.get("result", {})
        logger.info(f"  ✅ Numeric features scaled using {scale_result.get('scaling_method')}")
        
        # 4. Encode categorical features
        logger.info("  → Encoding categorical features")
        result = await self.execute_task(
            "fe_encode",
            "encode_categorical_features",
            {"run_id": run_id, "strategy": "auto"}
        )
        
        if not result.get("success"):
            logger.error(f"Categorical encoding failed: {result.get('error')}")
            return False
        
        encode_result = result.get("result", {})
        logger.info(f"  ✅ Categorical features encoded using {encode_result.get('encoding_strategy')}")
        
        # 5. Generate datetime features
        logger.info("  → Generating datetime features")
        result = await self.execute_task(
            "fe_datetime",
            "generate_datetime_features",
            {"run_id": run_id, "country": "US"}
        )
        
        if not result.get("success"):
            logger.error(f"Datetime feature generation failed: {result.get('error')}")
            return False
        
        datetime_result = result.get("result", {})
        logger.info(f"  ✅ Datetime features generated for {datetime_result.get('country')}")
        
        # 6. Vectorize text features
        logger.info("  → Vectorizing text features")
        result = await self.execute_task(
            "fe_text",
            "vectorise_text_features",
            {"run_id": run_id, "model": "mini-lm", "max_features": 1000}
        )
        
        if not result.get("success"):
            logger.error(f"Text vectorization failed: {result.get('error')}")
            return False
        
        text_result = result.get("result", {})
        logger.info(f"  ✅ Text features vectorized using {text_result.get('text_model')}")
        
        # 7. Generate interactions
        logger.info("  → Generating feature interactions")
        result = await self.execute_task(
            "fe_interactions",
            "generate_interactions",
            {"run_id": run_id, "max_degree": 2}
        )
        
        if not result.get("success"):
            logger.error(f"Feature interaction generation failed: {result.get('error')}")
            return False
        
        interaction_result = result.get("result", {})
        logger.info(f"  ✅ Feature interactions generated (degree {interaction_result.get('interaction_degree')})")
        
        # 8. Select features
        logger.info("  → Selecting features")
        result = await self.execute_task(
            "fe_select",
            "select_features",
            {"run_id": run_id, "method": "shap_top_k", "k": 50}
        )
        
        if not result.get("success"):
            logger.error(f"Feature selection failed: {result.get('error')}")
            return False
        
        select_result = result.get("result", {})
        logger.info(f"  ✅ Features selected: {select_result.get('selected_count')} features")
        
        # 9. Save pipeline
        logger.info("  → Saving feature engineering pipeline")
        export_data_path = self.test_data_dir / "fe_ready.parquet"
        export_pipeline_path = self.test_data_dir / "fe_pipeline.json"
        
        result = await self.execute_task(
            "fe_save",
            "save_fe_pipeline",
            {
                "input_path": str(data_path),
                "export_data_path": str(export_data_path),
                "export_pipeline_path": str(export_pipeline_path),
                "run_id": run_id
            }
        )
        
        if not result.get("success"):
            logger.error(f"Pipeline save failed: {result.get('error')}")
            return False
        
        save_result = result.get("result", {})
        logger.info(f"  ✅ Pipeline saved: {save_result.get('final_shape')}")
        
        logger.info("🎉 Feature Engineering workflow completed successfully!")
        return True
    
    async def test_load_scenario(self) -> bool:
        """Test high-load scenario with concurrent workflows."""
        logger.info("🧪 Testing Load Scenario")
        
        # Create multiple datasets
        datasets = []
        for i in range(5):
            df = self.create_test_dataset(500)
            data_path = self.test_data_dir / f"load_test_{i}.parquet"
            df.to_parquet(data_path)
            datasets.append(data_path)
        
        # Run concurrent DQ checks
        tasks = []
        for i, data_path in enumerate(datasets):
            task = self.execute_task(
                f"load_dq_{i}",
                "check_distributions",
                {"data_path": str(data_path)}
            )
            tasks.append(task)
        
        logger.info(f"  → Running {len(tasks)} concurrent DQ checks")
        start_time = time.time()
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        execution_time = time.time() - start_time
        successful_tasks = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        
        logger.info(f"  ✅ Load test completed: {successful_tasks}/{len(tasks)} tasks successful in {execution_time:.2f}s")
        
        return successful_tasks == len(tasks)
    
    async def test_fail_fast_scenario(self) -> bool:
        """Test fail-fast scenario with invalid data."""
        logger.info("🧪 Testing Fail-Fast Scenario")
        
        # Create dataset with schema issues
        df = self.create_test_dataset(100)
        # Add an unexpected column
        df['unexpected_column'] = np.random.randn(100)
        data_path = self.test_data_dir / "invalid_data.parquet"
        df.to_parquet(data_path)
        
        # Define expected schema (missing the unexpected column)
        expected_schema = {
            "columns": ["id", "numeric_feature_1", "numeric_feature_2", "categorical_feature", "text_feature", "datetime_feature", "target"]
        }
        
        # Test schema consistency check
        result = await self.execute_task(
            "fail_fast_schema",
            "check_schema_consistency",
            {"data_path": str(data_path), "expected_schema": expected_schema}
        )
        
        if not result.get("success"):
            logger.error(f"Schema check failed unexpectedly: {result.get('error')}")
            return False
        
        schema_result = result.get("result", {})
        
        # Should detect the schema mismatch
        if schema_result.get("status") == "pass":
            logger.error("Schema check should have failed due to unexpected column")
            return False
        
        extra_cols = schema_result.get("extra_columns", [])
        if "unexpected_column" not in extra_cols:
            logger.error("Schema check should have detected unexpected_column")
            return False
        
        logger.info(f"  ✅ Fail-fast test passed: detected {len(extra_cols)} extra columns")
        return True
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all end-to-end tests."""
        logger.info("🚀 Starting Refinery Agent End-to-End Tests")
        
        # Check if agent is available
        try:
            response = await self.http_client.get(f"{self.agent_url}/health")
            if response.status_code != 200:
                logger.error("Refinery agent is not healthy")
                return {"health_check": False}
        except Exception as e:
            logger.error(f"Cannot connect to refinery agent: {e}")
            return {"health_check": False}
        
        results = {}
        
        # Run individual test suites
        test_suites = [
            ("data_quality", self.test_data_quality_workflow),
            ("feature_engineering", self.test_feature_engineering_workflow),
            ("load_test", self.test_load_scenario),
            ("fail_fast", self.test_fail_fast_scenario)
        ]
        
        for test_name, test_func in test_suites:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Running {test_name} test suite")
                logger.info(f"{'='*50}")
                
                result = await test_func()
                results[test_name] = result
                
                if result:
                    logger.info(f"✅ {test_name} test suite PASSED")
                else:
                    logger.error(f"❌ {test_name} test suite FAILED")
                    
            except Exception as e:
                logger.error(f"❌ {test_name} test suite ERROR: {e}")
                results[test_name] = False
        
        # Summary
        logger.info(f"\n{'='*50}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*50}")
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name:20} {status}")
        
        logger.info(f"\nOverall: {passed}/{total} test suites passed")
        
        if passed == total:
            logger.info("🎉 All tests passed! Refinery agent is ready for production.")
        else:
            logger.error("⚠️  Some tests failed. Please review the implementation.")
        
        return results

async def main():
    """Main test runner."""
    async with RefineryE2ETest() as tester:
        results = await tester.run_all_tests()
        
        # Exit with appropriate code
        if all(results.values()):
            return 0
        else:
            return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code) 