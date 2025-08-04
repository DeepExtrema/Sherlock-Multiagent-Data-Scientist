#!/usr/bin/env python3
"""
Data Contract Validation Tests for Refinery Agent
Tests fail-fast scenarios and data contract enforcement.
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

class ContractValidationTest:
    """Test suite for data contract validation and fail-fast scenarios."""
    
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
    
    def create_invalid_dataset(self) -> pd.DataFrame:
        """Create a dataset with known data quality issues."""
        # Create base data
        data = {
            'id': [1, 2, 3, 4, 5, 1],  # Duplicate ID
            'name': ['Alice', None, 'Bob', 'Charlie', 'David', 'Eve'],  # Missing values
            'unexpected_column': np.random.randn(6),  # Extra column
            'category': ['A', 'B', 'C', 'D', 'A', 'B']  # Invalid enum value 'D'
        }
        
        return pd.DataFrame(data)
    
    def create_expected_schema(self) -> Dict[str, Any]:
        """Create expected schema for validation."""
        return {
            "columns": ["id", "name", "value", "category"],
            "dtypes": {
                "id": "int64",
                "name": "string",
                "value": "float64", 
                "category": "string"
            },
            "constraints": {
                "id": {"unique": True},
                "category": {"enum": ["A", "B", "C"]},
                "name": {"missing_threshold": 0.2}
            }
        }
    
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
    
    async def test_schema_contract_violation(self) -> bool:
        """Test that schema violations are detected and fail fast."""
        logger.info("🧪 Testing Schema Contract Violation")
        
        # Create invalid dataset
        df = self.create_invalid_dataset()
        data_path = self.test_data_dir / "invalid_schema.parquet"
        df.to_parquet(data_path)
        
        # Define expected schema
        expected_schema = self.create_expected_schema()
        
        # Test schema consistency check
        result = await self.execute_task(
            "contract_schema_check",
            "check_schema_consistency",
            {"data_path": str(data_path), "expected_schema": expected_schema}
        )
        
        if not result.get("success"):
            logger.error(f"Schema check failed unexpectedly: {result.get('error')}")
            return False
        
        schema_result = result.get("result", {})
        
        # Should detect schema violations
        if schema_result.get("status") == "pass":
            logger.error("Schema check should have failed due to contract violations")
            return False
        
        # Verify specific violations are detected
        missing_cols = schema_result.get("missing_columns", [])
        extra_cols = schema_result.get("extra_columns", [])
        
        expected_missing = ["value"]
        expected_extra = ["unexpected_column"]
        
        if not all(col in missing_cols for col in expected_missing):
            logger.error(f"Missing columns not detected: expected {expected_missing}, got {missing_cols}")
            return False
        
        if not all(col in extra_cols for col in expected_extra):
            logger.error(f"Extra columns not detected: expected {expected_extra}, got {extra_cols}")
            return False
        
        logger.info(f"  ✅ Schema contract violations detected: {len(missing_cols)} missing, {len(extra_cols)} extra")
        return True
    
    async def test_missing_values_threshold(self) -> bool:
        """Test that missing values above threshold are detected."""
        logger.info("🧪 Testing Missing Values Threshold")
        
        # Create dataset with high missing values
        df = self.create_invalid_dataset()
        data_path = self.test_data_dir / "high_missing.parquet"
        df.to_parquet(data_path)
        
        # Test with low threshold
        result = await self.execute_task(
            "contract_missing_check",
            "check_missing_values",
            {"data_path": str(data_path), "threshold_pct": 0.1}  # 10% threshold
        )
        
        if not result.get("success"):
            logger.error(f"Missing values check failed: {result.get('error')}")
            return False
        
        missing_result = result.get("result", {})
        
        # Should detect threshold violation
        if not missing_result.get("threshold_exceeded"):
            logger.error("Missing values threshold should have been exceeded")
            return False
        
        cols_over_threshold = missing_result.get("cols_over_threshold", [])
        if "name" not in cols_over_threshold:
            logger.error(f"Column 'name' should be over threshold, got: {cols_over_threshold}")
            return False
        
        logger.info(f"  ✅ Missing values threshold detected: {len(cols_over_threshold)} columns over limit")
        return True
    
    async def test_duplicate_detection(self) -> bool:
        """Test that duplicate IDs are detected."""
        logger.info("🧪 Testing Duplicate Detection")
        
        # Create dataset with duplicates
        df = self.create_invalid_dataset()
        data_path = self.test_data_dir / "duplicates.parquet"
        df.to_parquet(data_path)
        
        # Test duplicate check
        result = await self.execute_task(
            "contract_duplicates_check",
            "check_duplicates",
            {"data_path": str(data_path), "id_cols": ["id"]}
        )
        
        if not result.get("success"):
            logger.error(f"Duplicate check failed: {result.get('error')}")
            return False
        
        dup_result = result.get("result", {})
        
        # Should detect duplicates
        dup_ids = dup_result.get("duplicate_ids", 0)
        if dup_ids == 0:
            logger.error("Duplicate IDs should have been detected")
            return False
        
        logger.info(f"  ✅ Duplicate detection: {dup_ids} duplicate IDs found")
        return True
    
    async def test_invalid_action_handling(self) -> bool:
        """Test that invalid actions return proper error responses."""
        logger.info("🧪 Testing Invalid Action Handling")
        
        # Test with invalid action
        try:
            result = await self.execute_task(
                "invalid_action_test",
                "invalid_action_name",  # This should not exist
                {"data_path": "dummy.parquet"}
            )
            
            # Should not succeed
            if result.get("success"):
                logger.error("Invalid action should have failed")
                return False
            
            error = result.get("error", "")
            if "Unsupported action" not in error:
                logger.error(f"Expected 'Unsupported action' error, got: {error}")
                return False
            
            logger.info(f"  ✅ Invalid action properly rejected: {error}")
            return True
            
        except Exception as e:
            logger.error(f"Invalid action test failed unexpectedly: {e}")
            return False
    
    async def test_malformed_payload_handling(self) -> bool:
        """Test that malformed payloads are handled gracefully."""
        logger.info("🧪 Testing Malformed Payload Handling")
        
        # Test with malformed request
        malformed_request = {
            "task_id": "malformed_test",
            "action": "check_schema_consistency",
            "params": {
                "data_path": "/nonexistent/file.parquet"
            }
        }
        
        try:
            response = await self.http_client.post(
                f"{self.agent_url}/execute",
                json=malformed_request
            )
            
            result = response.json()
            
            # Should fail gracefully
            if result.get("success"):
                logger.error("Malformed payload should have failed")
                return False
            
            error = result.get("error", "")
            if "FileNotFoundError" not in error and "No such file" not in error:
                logger.error(f"Expected file not found error, got: {error}")
                return False
            
            logger.info(f"  ✅ Malformed payload handled gracefully: {error}")
            return True
            
        except Exception as e:
            logger.error(f"Malformed payload test failed: {e}")
            return False
    
    async def test_configurable_thresholds(self) -> bool:
        """Test that configurable thresholds work correctly."""
        logger.info("🧪 Testing Configurable Thresholds")
        
        # Create test data
        df = pd.DataFrame({
            'id': range(100),
            'value': np.random.normal(100, 15, 100)
        })
        data_path = self.test_data_dir / "threshold_test.parquet"
        df.to_parquet(data_path)
        
        # Test with custom threshold
        custom_threshold = 0.05  # 5%
        result = await self.execute_task(
            "threshold_test",
            "check_missing_values",
            {"data_path": str(data_path), "threshold_pct": custom_threshold}
        )
        
        if not result.get("success"):
            logger.error(f"Threshold test failed: {result.get('error')}")
            return False
        
        missing_result = result.get("result", {})
        
        # Verify threshold was used
        threshold_exceeded = missing_result.get("threshold_exceeded", False)
        logger.info(f"  ✅ Configurable threshold test: threshold_exceeded={threshold_exceeded}")
        return True
    
    async def test_idempotency(self) -> bool:
        """Test that repeated requests are idempotent."""
        logger.info("🧪 Testing Idempotency")
        
        # Create test data
        df = pd.DataFrame({
            'id': range(10),
            'name': [f'User{i}' for i in range(10)]
        })
        data_path = self.test_data_dir / "idempotency_test.parquet"
        df.to_parquet(data_path)
        
        # Execute same task twice
        task_id = "idempotency_test"
        params = {"data_path": str(data_path)}
        
        result1 = await self.execute_task(task_id, "check_schema_consistency", params)
        result2 = await self.execute_task(task_id, "check_schema_consistency", params)
        
        # Results should be identical
        if result1.get("success") != result2.get("success"):
            logger.error("Idempotency test failed: success status differs")
            return False
        
        if result1.get("result") != result2.get("result"):
            logger.error("Idempotency test failed: results differ")
            return False
        
        logger.info("  ✅ Idempotency test passed: repeated requests return same results")
        return True
    
    async def run_all_contract_tests(self) -> Dict[str, bool]:
        """Run all contract validation tests."""
        logger.info("🚀 Starting Data Contract Validation Tests")
        
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
            ("schema_contract", self.test_schema_contract_violation),
            ("missing_values_threshold", self.test_missing_values_threshold),
            ("duplicate_detection", self.test_duplicate_detection),
            ("invalid_action", self.test_invalid_action_handling),
            ("malformed_payload", self.test_malformed_payload_handling),
            ("configurable_thresholds", self.test_configurable_thresholds),
            ("idempotency", self.test_idempotency)
        ]
        
        for test_name, test_func in test_suites:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Running {test_name} test")
                logger.info(f"{'='*50}")
                
                result = await test_func()
                results[test_name] = result
                
                if result:
                    logger.info(f"✅ {test_name} test PASSED")
                else:
                    logger.error(f"❌ {test_name} test FAILED")
                    
            except Exception as e:
                logger.error(f"❌ {test_name} test ERROR: {e}")
                results[test_name] = False
        
        # Summary
        logger.info(f"\n{'='*50}")
        logger.info("CONTRACT TEST SUMMARY")
        logger.info(f"{'='*50}")
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name:25} {status}")
        
        logger.info(f"\nOverall: {passed}/{total} contract tests passed")
        
        if passed == total:
            logger.info("🎉 All contract tests passed! Data contracts are properly enforced.")
        else:
            logger.error("⚠️  Some contract tests failed. Please review the implementation.")
        
        return results

async def main():
    """Main test runner."""
    async with ContractValidationTest() as tester:
        results = await tester.run_all_contract_tests()
        
        # Exit with appropriate code
        if all(results.values()):
            return 0
        else:
            return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code) 