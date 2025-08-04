#!/usr/bin/env python3
"""
End-to-End Test: Iris Dataset Prediction Model
Tests complete workflow from data upload to model creation using the orchestrator.
"""

import asyncio
import json
import logging
import time
import tempfile
import os
import subprocess
import signal
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import shutil

import pandas as pd
import numpy as np
import httpx
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IrisE2ETest:
    """End-to-end test for iris dataset prediction model."""
    
    def __init__(self):
        self.orchestrator_url = "http://localhost:8000"
        self.eda_agent_url = "http://localhost:8001"
        self.ml_agent_url = "http://localhost:8002"
        self.refinery_agent_url = "http://localhost:8005"
        
        self.http_client = httpx.AsyncClient(timeout=300)
        self.test_data_dir = Path(tempfile.mkdtemp())
        self.processes = []
        
        # Test results
        self.results = {}
        
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.aclose()
        self.cleanup()
    
    def cleanup(self):
        """Clean up test resources."""
        # Stop all processes
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
        
        # Cleanup test data
        shutil.rmtree(self.test_data_dir, ignore_errors=True)
    
    async def start_service(self, service_name: str, command: list, port: int) -> subprocess.Popen:
        """Start a service and wait for it to be ready."""
        logger.info(f"Starting {service_name}...")
        
        # Set environment variables
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path.cwd())
        
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        self.processes.append(process)
        
        # Wait for service to be ready
        max_retries = 30
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = httpx.get(f"http://localhost:{port}/health", timeout=5)
                if response.status_code == 200:
                    logger.info(f"✅ {service_name} is ready")
                    return process
            except:
                pass
            
            await asyncio.sleep(2)
            retry_count += 1
        
        raise RuntimeError(f"Failed to start {service_name}")
    
    async def start_services(self):
        """Start all required services."""
        logger.info("🚀 Starting services...")
        
        # Start Master Orchestrator
        orchestrator_process = await self.start_service(
            "Master Orchestrator",
            [sys.executable, "master_orchestrator_api.py"],
            8000
        )
        
        # Start EDA Agent
        eda_process = await self.start_service(
            "EDA Agent",
            [sys.executable, "eda_agent.py"],
            8001
        )
        
        # Start ML Agent
        ml_process = await self.start_service(
            "ML Agent",
            [sys.executable, "ml_agent.py"],
            8002
        )
        
        # Start Refinery Agent
        refinery_process = await self.start_service(
            "Refinery Agent",
            [sys.executable, "refinery_agent.py"],
            8005
        )
        
        logger.info("✅ All services started successfully")
    
    async def upload_iris_dataset(self) -> str:
        """Upload the iris dataset to the orchestrator."""
        logger.info("📤 Uploading iris dataset...")
        
        # Copy iris.csv to test directory
        iris_source = Path("iris.csv")
        iris_dest = self.test_data_dir / "iris.csv"
        shutil.copy2(iris_source, iris_dest)
        
        # Upload to orchestrator
        with open(iris_dest, "rb") as f:
            files = {"file": ("iris.csv", f, "text/csv")}
            data = {"name": "iris_dataset"}
            
            response = await self.http_client.post(
                f"{self.orchestrator_url}/datasets/upload",
                files=files,
                data=data
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Failed to upload dataset: {response.text}")
            
            result = response.json()
            dataset_id = result.get("dataset_id")
            logger.info(f"✅ Dataset uploaded with ID: {dataset_id}")
            return dataset_id
    
    async def run_eda_analysis(self, dataset_name: str) -> Dict[str, Any]:
        """Run comprehensive EDA analysis."""
        logger.info("🔍 Running EDA analysis...")
        
        # Load data into EDA agent
        load_response = await self.http_client.post(
            f"{self.eda_agent_url}/load_data",
            json={
                "path": str(self.test_data_dir / "iris.csv"),
                "name": dataset_name,
                "file_type": "csv"
            }
        )
        
        if load_response.status_code != 200:
            raise RuntimeError(f"Failed to load data: {load_response.text}")
        
        # Get basic info
        basic_info = await self.http_client.post(
            f"{self.eda_agent_url}/basic_info",
            json={"name": dataset_name}
        )
        
        # Get statistical summary
        stats_summary = await self.http_client.post(
            f"{self.eda_agent_url}/statistical_summary",
            json={"name": dataset_name}
        )
        
        # Get missing data analysis
        missing_data = await self.http_client.post(
            f"{self.eda_agent_url}/missing_data_analysis",
            json={"name": dataset_name}
        )
        
        # Infer schema
        schema = await self.http_client.post(
            f"{self.eda_agent_url}/infer_schema",
            json={"name": dataset_name}
        )
        
        # Create visualizations
        viz_response = await self.http_client.post(
            f"{self.eda_agent_url}/create_visualization",
            json={
                "name": dataset_name,
                "chart_type": "correlation",
                "columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
            }
        )
        
        eda_results = {
            "basic_info": basic_info.json() if basic_info.status_code == 200 else None,
            "stats_summary": stats_summary.json() if stats_summary.status_code == 200 else None,
            "missing_data": missing_data.json() if missing_data.status_code == 200 else None,
            "schema": schema.json() if schema.status_code == 200 else None,
            "visualization": viz_response.json() if viz_response.status_code == 200 else None
        }
        
        logger.info("✅ EDA analysis completed")
        return eda_results
    
    async def run_data_quality_checks(self, dataset_path: str) -> Dict[str, Any]:
        """Run data quality checks using refinery agent."""
        logger.info("🔍 Running data quality checks...")
        
        # Check schema consistency
        schema_check = await self.http_client.post(
            f"{self.refinery_agent_url}/execute",
            json={
                "task_id": "dq_schema_iris",
                "action": "check_schema_consistency",
                "params": {"data_path": dataset_path}
            }
        )
        
        # Check missing values
        missing_check = await self.http_client.post(
            f"{self.refinery_agent_url}/execute",
            json={
                "task_id": "dq_missing_iris",
                "action": "check_missing_values",
                "params": {"data_path": dataset_path, "threshold_pct": 0.5}
            }
        )
        
        # Check distributions
        dist_check = await self.http_client.post(
            f"{self.refinery_agent_url}/execute",
            json={
                "task_id": "dq_dist_iris",
                "action": "check_distributions",
                "params": {"data_path": dataset_path}
            }
        )
        
        dq_results = {
            "schema_check": schema_check.json() if schema_check.status_code == 200 else None,
            "missing_check": missing_check.json() if missing_check.status_code == 200 else None,
            "dist_check": dist_check.json() if dist_check.status_code == 200 else None
        }
        
        logger.info("✅ Data quality checks completed")
        return dq_results
    
    async def run_feature_engineering(self, dataset_path: str) -> Dict[str, Any]:
        """Run feature engineering pipeline."""
        logger.info("⚙️ Running feature engineering...")
        
        run_id = f"iris_fe_{int(time.time())}"
        
        # Assign feature roles
        roles_result = await self.http_client.post(
            f"{self.refinery_agent_url}/execute",
            json={
                "task_id": "fe_roles_iris",
                "action": "assign_feature_roles",
                "params": {"input_path": dataset_path, "run_id": run_id}
            }
        )
        
        # Scale numeric features
        scale_result = await self.http_client.post(
            f"{self.refinery_agent_url}/execute",
            json={
                "task_id": "fe_scale_iris",
                "action": "scale_numeric_features",
                "params": {"run_id": run_id, "method": "standard"}
            }
        )
        
        # Encode categorical features (species)
        encode_result = await self.http_client.post(
            f"{self.refinery_agent_url}/execute",
            json={
                "task_id": "fe_encode_iris",
                "action": "encode_categorical_features",
                "params": {"run_id": run_id, "strategy": "label"}
            }
        )
        
        # Save pipeline
        export_path = str(self.test_data_dir / "iris_processed.parquet")
        pipeline_path = str(self.test_data_dir / "iris_pipeline.json")
        
        save_result = await self.http_client.post(
            f"{self.refinery_agent_url}/execute",
            json={
                "task_id": "fe_save_iris",
                "action": "save_fe_pipeline",
                "params": {
                    "input_path": dataset_path,
                    "export_data_path": export_path,
                    "export_pipeline_path": pipeline_path,
                    "run_id": run_id
                }
            }
        )
        
        fe_results = {
            "roles": roles_result.json() if roles_result.status_code == 200 else None,
            "scaling": scale_result.json() if scale_result.status_code == 200 else None,
            "encoding": encode_result.json() if encode_result.status_code == 200 else None,
            "save": save_result.json() if save_result.status_code == 200 else None,
            "processed_data_path": export_path,
            "pipeline_path": pipeline_path
        }
        
        logger.info("✅ Feature engineering completed")
        return fe_results
    
    async def create_prediction_model(self, processed_data_path: str) -> Dict[str, Any]:
        """Create and train prediction model."""
        logger.info("🤖 Creating prediction model...")
        
        task_id = f"iris_ml_{int(time.time())}"
        experiment_id = f"iris_exp_{int(time.time())}"
        
        # Step 1: Class imbalance analysis
        imbalance_result = await self.http_client.post(
            f"{self.ml_agent_url}/class_imbalance",
            json={
                "task_id": task_id,
                "data_path": processed_data_path,
                "target_column": "species",
                "sampling_strategy": "none",  # Iris is balanced
                "random_state": 42,
                "test_size": 0.2,
                "cv_folds": 5
            }
        )
        
        # Step 2: Train multiple models
        models_to_train = ["random_forest", "gradient_boosting", "logistic_regression", "svm"]
        training_results = {}
        
        for model_type in models_to_train:
            train_result = await self.http_client.post(
                f"{self.ml_agent_url}/train_validation_test",
                json={
                    "task_id": f"{task_id}_{model_type}",
                    "experiment_id": experiment_id,
                    "model_type": model_type,
                    "split_strategy": "stratified",
                    "cv_folds": 5,
                    "random_state": 42,
                    "early_stopping": True
                }
            )
            
            if train_result.status_code == 200:
                training_results[model_type] = train_result.json()
        
        # Step 3: Baseline sanity checks
        baseline_result = await self.http_client.post(
            f"{self.ml_agent_url}/baseline_sanity",
            json={
                "task_id": f"{task_id}_baseline",
                "experiment_id": experiment_id,
                "baseline_models": ["baseline_random", "baseline_majority", "naive_bayes"],
                "leakage_test": True,
                "association_analysis": True
            }
        )
        
        # Step 4: Experiment tracking
        tracking_result = await self.http_client.post(
            f"{self.ml_agent_url}/experiment_tracking",
            json={
                "task_id": f"{task_id}_tracking",
                "experiment_id": experiment_id,
                "experiment_name": "Iris Classification",
                "tags": {
                    "dataset": "iris",
                    "task": "classification",
                    "test_run": "true"
                },
                "artifact_path": str(self.test_data_dir),
                "model_registry": True
            }
        )
        
        ml_results = {
            "imbalance_analysis": imbalance_result.json() if imbalance_result.status_code == 200 else None,
            "training_results": training_results,
            "baseline_checks": baseline_result.json() if baseline_result.status_code == 200 else None,
            "experiment_tracking": tracking_result.json() if tracking_result.status_code == 200 else None
        }
        
        logger.info("✅ Prediction model created")
        return ml_results
    
    async def run_orchestrator_workflow(self, dataset_id: str) -> Dict[str, Any]:
        """Run complete workflow through orchestrator."""
        logger.info("🎯 Running orchestrator workflow...")
        
        # Define workflow tasks
        workflow_request = {
            "run_name": f"iris_classification_{int(time.time())}",
            "priority": 1,
            "tasks": [
                {
                    "agent": "eda_agent",
                    "action": "basic_info",
                    "args": {"name": "iris_dataset"}
                },
                {
                    "agent": "eda_agent",
                    "action": "statistical_summary",
                    "args": {"name": "iris_dataset"}
                },
                {
                    "agent": "refinery_agent",
                    "action": "check_schema_consistency",
                    "args": {"data_path": str(self.test_data_dir / "iris.csv")}
                },
                {
                    "agent": "ml_agent",
                    "action": "class_imbalance",
                    "args": {
                        "data_path": str(self.test_data_dir / "iris.csv"),
                        "target_column": "species"
                    }
                }
            ]
        }
        
        # Start workflow
        response = await self.http_client.post(
            f"{self.orchestrator_url}/workflows/start",
            json=workflow_request
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to start workflow: {response.text}")
        
        workflow_result = response.json()
        run_id = workflow_result["run_id"]
        
        # Monitor workflow progress
        max_wait = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status_response = await self.http_client.get(
                f"{self.orchestrator_url}/runs/{run_id}/status"
            )
            
            if status_response.status_code == 200:
                status = status_response.json()
                logger.info(f"Workflow status: {status['status']} - Progress: {status['progress']:.1f}%")
                
                if status['status'] == 'completed':
                    # Get artifacts
                    artifacts_response = await self.http_client.get(
                        f"{self.orchestrator_url}/runs/{run_id}/artifacts"
                    )
                    
                    if artifacts_response.status_code == 200:
                        artifacts = artifacts_response.json()
                        logger.info(f"✅ Workflow completed with {len(artifacts)} artifacts")
                        return {
                            "run_id": run_id,
                            "status": status,
                            "artifacts": artifacts
                        }
                
                elif status['status'] == 'failed':
                    raise RuntimeError(f"Workflow failed: {status.get('error_message', 'Unknown error')}")
            
            await asyncio.sleep(5)
        
        raise RuntimeError("Workflow timed out")
    
    def analyze_results(self, eda_results: Dict[str, Any], dq_results: Dict[str, Any], 
                       fe_results: Dict[str, Any], ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and summarize test results."""
        logger.info("📊 Analyzing results...")
        
        analysis = {
            "summary": {
                "eda_success": all(eda_results.values()),
                "dq_success": all(dq_results.values()),
                "fe_success": all(fe_results.values()),
                "ml_success": all(ml_results.values())
            },
            "key_findings": [],
            "recommendations": [],
            "model_performance": {}
        }
        
        # Analyze EDA results
        if eda_results.get("basic_info"):
            basic_info = eda_results["basic_info"]
            analysis["key_findings"].append(f"Dataset shape: {basic_info['shape']}")
            analysis["key_findings"].append(f"Memory usage: {basic_info['memory_usage']}")
        
        if eda_results.get("stats_summary"):
            stats = eda_results["stats_summary"]
            analysis["key_findings"].append(f"Features analyzed: {len(stats.get('descriptive_stats', {}))}")
        
        # Analyze ML results
        if ml_results.get("training_results"):
            training_results = ml_results["training_results"]
            
            for model_name, result in training_results.items():
                if result and result.get("success"):
                    metrics = result.get("result", {}).get("metrics", {})
                    if metrics:
                        analysis["model_performance"][model_name] = {
                            "accuracy": metrics.get("accuracy", 0),
                            "f1_score": metrics.get("f1_score", 0),
                            "precision": metrics.get("precision", 0),
                            "recall": metrics.get("recall", 0)
                        }
        
        # Generate recommendations
        if analysis["summary"]["eda_success"]:
            analysis["recommendations"].append("✅ EDA analysis completed successfully")
        
        if analysis["summary"]["dq_success"]:
            analysis["recommendations"].append("✅ Data quality checks passed")
        
        if analysis["summary"]["fe_success"]:
            analysis["recommendations"].append("✅ Feature engineering pipeline created")
        
        if analysis["summary"]["ml_success"]:
            best_model = max(analysis["model_performance"].items(), 
                           key=lambda x: x[1]["accuracy"])[0]
            analysis["recommendations"].append(f"✅ Best model: {best_model}")
        
        return analysis
    
    async def run_complete_test(self) -> Dict[str, Any]:
        """Run the complete end-to-end test."""
        logger.info("🚀 Starting Iris Dataset End-to-End Test")
        logger.info("=" * 60)
        
        try:
            # Step 1: Start services
            await self.start_services()
            
            # Step 2: Upload dataset
            dataset_id = await self.upload_iris_dataset()
            
            # Step 3: Run EDA analysis
            eda_results = await self.run_eda_analysis("iris_dataset")
            
            # Step 4: Run data quality checks
            dq_results = await self.run_data_quality_checks(str(self.test_data_dir / "iris.csv"))
            
            # Step 5: Run feature engineering
            fe_results = await self.run_feature_engineering(str(self.test_data_dir / "iris.csv"))
            
            # Step 6: Create prediction model
            processed_data_path = fe_results.get("processed_data_path")
            if processed_data_path and os.path.exists(processed_data_path):
                ml_results = await self.create_prediction_model(processed_data_path)
            else:
                # Fallback to original data
                ml_results = await self.create_prediction_model(str(self.test_data_dir / "iris.csv"))
            
            # Step 7: Run orchestrator workflow
            orchestrator_results = await self.run_orchestrator_workflow(dataset_id)
            
            # Step 8: Analyze results
            analysis = self.analyze_results(eda_results, dq_results, fe_results, ml_results)
            
            # Compile final results
            self.results = {
                "test_name": "Iris Dataset End-to-End Test",
                "timestamp": time.time(),
                "success": all([
                    analysis["summary"]["eda_success"],
                    analysis["summary"]["dq_success"],
                    analysis["summary"]["fe_success"],
                    analysis["summary"]["ml_success"]
                ]),
                "eda_results": eda_results,
                "dq_results": dq_results,
                "fe_results": fe_results,
                "ml_results": ml_results,
                "orchestrator_results": orchestrator_results,
                "analysis": analysis
            }
            
            logger.info("=" * 60)
            logger.info("🎉 End-to-End Test Completed Successfully!")
            logger.info("=" * 60)
            
            return self.results
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            self.results = {
                "test_name": "Iris Dataset End-to-End Test",
                "timestamp": time.time(),
                "success": False,
                "error": str(e)
            }
            return self.results
    
    def print_summary(self):
        """Print test summary."""
        if not self.results:
            logger.error("No results to summarize")
            return
        
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        logger.info(f"Test Name: {self.results['test_name']}")
        logger.info(f"Success: {'✅ PASSED' if self.results['success'] else '❌ FAILED'}")
        logger.info(f"Timestamp: {self.results['timestamp']}")
        
        if self.results.get('error'):
            logger.error(f"Error: {self.results['error']}")
        
        if self.results.get('analysis'):
            analysis = self.results['analysis']
            
            logger.info("\nComponent Status:")
            for component, status in analysis['summary'].items():
                status_icon = "✅" if status else "❌"
                logger.info(f"  {component}: {status_icon}")
            
            if analysis.get('key_findings'):
                logger.info("\nKey Findings:")
                for finding in analysis['key_findings']:
                    logger.info(f"  • {finding}")
            
            if analysis.get('recommendations'):
                logger.info("\nRecommendations:")
                for rec in analysis['recommendations']:
                    logger.info(f"  • {rec}")
            
            if analysis.get('model_performance'):
                logger.info("\nModel Performance:")
                for model, metrics in analysis['model_performance'].items():
                    logger.info(f"  {model}:")
                    for metric, value in metrics.items():
                        logger.info(f"    {metric}: {value:.4f}")

async def main():
    """Main test runner."""
    async with IrisE2ETest() as tester:
        results = await tester.run_complete_test()
        tester.print_summary()
        
        # Save results to file
        results_file = Path("iris_e2e_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {results_file}")
        
        # Exit with appropriate code
        return 0 if results['success'] else 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1) 