#!/usr/bin/env python3
"""
ML Agent Test Script
Tests the ML agent functionality to identify issues and verify capabilities.
"""

import asyncio
import json
import time
import httpx
import pandas as pd
from pathlib import Path

class MLAgentTester:
    """Test the ML Agent functionality."""
    
    def __init__(self):
        self.base_url = "http://localhost:8002"
        self.http_client = httpx.AsyncClient(timeout=30)
        
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.aclose()
    
    async def test_health_endpoint(self):
        """Test the health endpoint."""
        print("🔍 Testing health endpoint...")
        try:
            response = await self.http_client.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    async def test_metrics_endpoint(self):
        """Test the metrics endpoint."""
        print("🔍 Testing metrics endpoint...")
        try:
            response = await self.http_client.get(f"{self.base_url}/metrics")
            if response.status_code == 200:
                print("✅ Metrics endpoint working")
                return True
            else:
                print(f"❌ Metrics endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Metrics endpoint error: {e}")
            return False
    
    async def test_experiments_endpoint(self):
        """Test the experiments endpoint."""
        print("🔍 Testing experiments endpoint...")
        try:
            response = await self.http_client.get(f"{self.base_url}/experiments")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Experiments endpoint working: {data}")
                return True
            else:
                print(f"❌ Experiments endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Experiments endpoint error: {e}")
            return False
    
    async def test_class_imbalance_endpoint(self):
        """Test the class imbalance endpoint with iris dataset."""
        print("🔍 Testing class imbalance endpoint...")
        
        # Check if iris.csv exists
        iris_path = Path("iris.csv")
        if not iris_path.exists():
            print("❌ iris.csv not found")
            return False
        
        try:
            # Test class imbalance analysis
            request_data = {
                "task_id": "test_task_001",
                "data_path": str(iris_path.absolute()),
                "target_column": "species",
                "sampling_strategy": "none",
                "random_state": 42,
                "test_size": 0.2,
                "cv_folds": 5
            }
            
            response = await self.http_client.post(
                f"{self.base_url}/class_imbalance",
                json=request_data
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Class imbalance analysis successful")
                print(f"   - Experiment ID: {data['result']['experiment_id']}")
                print(f"   - Imbalance ratio: {data['result']['imbalance_metrics']['imbalance_ratio']}")
                print(f"   - Severity: {data['result']['imbalance_metrics']['severity']}")
                return True
            else:
                print(f"❌ Class imbalance failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Class imbalance error: {e}")
            return False
    
    async def test_training_endpoint(self):
        """Test the training endpoint."""
        print("🔍 Testing training endpoint...")
        
        try:
            # Test model training
            request_data = {
                "task_id": "test_task_002",
                "experiment_id": "test_exp_001",
                "model_type": "random_forest",
                "hyperparameters": {
                    "n_estimators": 100,
                    "max_depth": 10
                },
                "split_strategy": "stratified",
                "cv_folds": 5,
                "random_state": 42,
                "early_stopping": False,
                "max_iterations": 1000
            }
            
            response = await self.http_client.post(
                f"{self.base_url}/train_validation_test",
                json=request_data
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Training successful")
                print(f"   - Model type: {data['result']['model_type']}")
                print(f"   - Accuracy: {data['result']['metrics']['accuracy']:.4f}")
                return True
            else:
                print(f"❌ Training failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Training error: {e}")
            return False
    
    async def test_baseline_endpoint(self):
        """Test the baseline sanity endpoint."""
        print("🔍 Testing baseline sanity endpoint...")
        
        try:
            request_data = {
                "task_id": "test_task_003",
                "experiment_id": "test_exp_001",
                "baseline_models": ["baseline_random", "baseline_majority", "naive_bayes"],
                "leakage_test": True,
                "association_analysis": True,
                "max_rules": 5,
                "min_confidence": 0.5
            }
            
            response = await self.http_client.post(
                f"{self.base_url}/baseline_sanity",
                json=request_data
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Baseline sanity check successful")
                print(f"   - Models tested: {len(data['result']['baseline_results'])}")
                return True
            else:
                print(f"❌ Baseline sanity failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Baseline sanity error: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all tests."""
        print("🚀 Starting ML Agent Tests")
        print("=" * 50)
        
        tests = [
            ("Health Endpoint", self.test_health_endpoint),
            ("Metrics Endpoint", self.test_metrics_endpoint),
            ("Experiments Endpoint", self.test_experiments_endpoint),
            ("Class Imbalance", self.test_class_imbalance_endpoint),
            ("Training", self.test_training_endpoint),
            ("Baseline Sanity", self.test_baseline_endpoint),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            print(f"\n📋 Running: {test_name}")
            try:
                result = await test_func()
                results[test_name] = result
                print(f"   Status: {'✅ PASSED' if result else '❌ FAILED'}")
            except Exception as e:
                print(f"   Status: ❌ ERROR - {e}")
                results[test_name] = False
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name}: {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 All tests passed! ML Agent is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the issues above.")
        
        return results

async def main():
    """Main function."""
    async with MLAgentTester() as tester:
        await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main()) 