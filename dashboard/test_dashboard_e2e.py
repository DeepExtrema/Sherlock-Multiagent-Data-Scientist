#!/usr/bin/env python3
"""
End-to-End Dashboard Integration Test
Tests the complete Deepline system including:
- Infrastructure services (MongoDB, Kafka)
- MCP Server (EDA Agent)
- ML Agent
- Dashboard Backend
- Dashboard Frontend
- Real-time event streaming
- Workflow orchestration
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
from typing import Dict, Any, Optional, List
import shutil
import requests
import websockets
import httpx
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DashboardE2ETest:
    """End-to-end test for the complete Deepline dashboard system."""
    
    def __init__(self):
        # Service URLs
        self.dashboard_backend_url = "http://localhost:8000"
        self.dashboard_frontend_url = "http://localhost:3000"
        self.eda_agent_url = "http://localhost:8001"
        self.ml_agent_url = "http://localhost:8002"
        
        # Infrastructure
        self.mongodb_url = "mongodb://localhost:27017"
        self.kafka_url = "localhost:9092"
        
        self.http_client = httpx.AsyncClient(timeout=300)
        self.test_data_dir = Path(tempfile.mkdtemp())
        self.processes = []
        
        # Test results
        self.results = {
            'infrastructure': {},
            'services': {},
            'workflows': {},
            'dashboard': {},
            'events': {},
            'overall': {}
        }
        
        # Test data
        self.iris_data = None
        self.workflow_id = None
        self.run_id = None
        
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.aclose()
        self.cleanup()
    
    def cleanup(self):
        """Clean up test resources."""
        logger.info("Cleaning up test resources...")
        
        # Stop all processes
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                try:
                    process.kill()
                except:
                    pass
        
        # Cleanup test data
        shutil.rmtree(self.test_data_dir, ignore_errors=True)
    
    async def start_service(self, service_name: str, command: list, port: int, health_endpoint: str = "/health") -> subprocess.Popen:
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
                response = httpx.get(f"http://localhost:{port}{health_endpoint}", timeout=5)
                if response.status_code == 200:
                    logger.info(f"✅ {service_name} is ready")
                    return process
            except:
                pass
            
            await asyncio.sleep(2)
            retry_count += 1
        
        raise Exception(f"Failed to start {service_name} after {max_retries} retries")
    
    async def start_infrastructure(self):
        """Start MongoDB and Kafka using docker-compose."""
        logger.info("Starting infrastructure services...")
        
        try:
            # Start docker-compose services
            process = subprocess.Popen(
                ["docker-compose", "up", "-d"],
                cwd=Path.cwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            process.wait(timeout=60)
            
            # Wait for services to be ready
            await asyncio.sleep(10)
            
            # Test MongoDB connection
            try:
                import pymongo
                client = pymongo.MongoClient(self.mongodb_url)
                client.admin.command('ping')
                logger.info("✅ MongoDB is ready")
                self.results['infrastructure']['mongodb'] = 'PASS'
            except Exception as e:
                logger.error(f"❌ MongoDB connection failed: {e}")
                self.results['infrastructure']['mongodb'] = 'FAIL'
                raise
            
            # Test Kafka connection
            try:
                from confluent_kafka import Producer
                producer = Producer({'bootstrap.servers': self.kafka_url})
                producer.flush(timeout=10)
                logger.info("✅ Kafka is ready")
                self.results['infrastructure']['kafka'] = 'PASS'
            except Exception as e:
                logger.error(f"❌ Kafka connection failed: {e}")
                self.results['infrastructure']['kafka'] = 'FAIL'
                raise
                
        except Exception as e:
            logger.error(f"Failed to start infrastructure: {e}")
            self.results['infrastructure']['overall'] = 'FAIL'
            raise
    
    async def start_services(self):
        """Start all application services."""
        logger.info("Starting application services...")
        
        try:
            # Start EDA Agent (FastAPI service)
            eda_process = await self.start_service(
                "EDA Agent",
                ["python", "-m", "uvicorn", "eda_agent:app", "--host", "0.0.0.0", "--port", "8001"],
                8001,
                "/health"
            )
            self.results['services']['eda_agent'] = 'PASS'
            
            # Start ML Agent
            ml_process = await self.start_service(
                "ML Agent",
                ["python", "-m", "uvicorn", "ml_agent:app", "--host", "0.0.0.0", "--port", "8002"],
                8002,
                "/health"
            )
            self.results['services']['ml_agent'] = 'PASS'
            
            # Start Dashboard Backend
            dashboard_process = await self.start_service(
                "Dashboard Backend",
                ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"],
                8000,
                "/"
            )
            self.results['services']['dashboard_backend'] = 'PASS'
            
            # Start Dashboard Frontend (React)
            frontend_process = await self.start_service(
                "Dashboard Frontend",
                ["npm", "start"],
                3000,
                "/"
            )
            self.results['services']['dashboard_frontend'] = 'PASS'
            
            logger.info("✅ All services started successfully")
            self.results['services']['overall'] = 'PASS'
            
        except Exception as e:
            logger.error(f"Failed to start services: {e}")
            self.results['services']['overall'] = 'FAIL'
            raise
    
    def create_test_data(self):
        """Create test dataset (Iris)."""
        logger.info("Creating test dataset...")
        
        from sklearn.datasets import load_iris
        
        iris = load_iris()
        self.iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
        self.iris_data['target'] = iris.target
        
        # Save to test directory
        test_file = self.test_data_dir / "iris_test.csv"
        self.iris_data.to_csv(test_file, index=False)
        
        logger.info(f"✅ Test dataset created: {test_file}")
        return str(test_file)
    
    async def test_eda_agent(self, dataset_path: str):
        """Test EDA Agent functionality."""
        logger.info("Testing EDA Agent...")
        
        try:
            # Test basic info
            response = await self.http_client.post(
                f"{self.eda_agent_url}/basic_info",
                json={"name": "iris_test"}
            )
            assert response.status_code == 200
            logger.info("✅ EDA Agent basic_info endpoint working")
            
            # Test statistical summary
            response = await self.http_client.post(
                f"{self.eda_agent_url}/statistical_summary",
                json={"name": "iris_test"}
            )
            assert response.status_code == 200
            logger.info("✅ EDA Agent statistical_summary endpoint working")
            
            self.results['services']['eda_agent_tests'] = 'PASS'
            
        except Exception as e:
            logger.error(f"❌ EDA Agent test failed: {e}")
            self.results['services']['eda_agent_tests'] = 'FAIL'
            raise
    
    async def test_ml_agent(self):
        """Test ML Agent functionality."""
        logger.info("Testing ML Agent...")
        
        try:
            # Test health endpoint
            response = await self.http_client.get(f"{self.ml_agent_url}/health")
            assert response.status_code == 200
            logger.info("✅ ML Agent health endpoint working")
            
            # Test metrics endpoint
            response = await self.http_client.get(f"{self.ml_agent_url}/metrics")
            assert response.status_code == 200
            logger.info("✅ ML Agent metrics endpoint working")
            
            # Test experiments endpoint
            response = await self.http_client.get(f"{self.ml_agent_url}/experiments")
            assert response.status_code == 200
            logger.info("✅ ML Agent experiments endpoint working")
            
            self.results['services']['ml_agent_tests'] = 'PASS'
            
        except Exception as e:
            logger.error(f"❌ ML Agent test failed: {e}")
            self.results['services']['ml_agent_tests'] = 'FAIL'
            raise
    
    async def test_dashboard_backend(self):
        """Test Dashboard Backend functionality."""
        logger.info("Testing Dashboard Backend...")
        
        try:
            # Test root endpoint
            response = await self.http_client.get(f"{self.dashboard_backend_url}/")
            assert response.status_code == 200
            logger.info("✅ Dashboard Backend root endpoint working")
            
            # Test runs endpoint
            response = await self.http_client.get(f"{self.dashboard_backend_url}/runs")
            assert response.status_code == 200
            logger.info("✅ Dashboard Backend runs endpoint working")
            
            # Test WebSocket connection
            try:
                async with websockets.connect(f"ws://localhost:8000/ws/events") as websocket:
                    # Send a test message
                    await websocket.send(json.dumps({"type": "test", "message": "Hello"}))
                    
                    # Wait for response
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    logger.info("✅ Dashboard Backend WebSocket working")
                    
            except Exception as e:
                logger.error(f"❌ Dashboard Backend WebSocket test failed: {e}")
                raise
            
            self.results['dashboard']['backend_tests'] = 'PASS'
            
        except Exception as e:
            logger.error(f"❌ Dashboard Backend test failed: {e}")
            self.results['dashboard']['backend_tests'] = 'FAIL'
            raise
    
    async def test_dashboard_frontend(self):
        """Test Dashboard Frontend functionality."""
        logger.info("Testing Dashboard Frontend...")
        
        try:
            # Test if React app is accessible
            response = await self.http_client.get(f"{self.dashboard_frontend_url}/")
            assert response.status_code == 200
            logger.info("✅ Dashboard Frontend is accessible")
            
            # Check if it's a React app (should contain React-related content)
            content = response.text
            if "react" in content.lower() or "app" in content.lower():
                logger.info("✅ Dashboard Frontend appears to be a React application")
            else:
                logger.warning("⚠️ Dashboard Frontend might not be a React app")
            
            self.results['dashboard']['frontend_tests'] = 'PASS'
            
        except Exception as e:
            logger.error(f"❌ Dashboard Frontend test failed: {e}")
            self.results['dashboard']['frontend_tests'] = 'FAIL'
            raise
    
    async def create_workflow(self):
        """Create a test workflow."""
        logger.info("Creating test workflow...")
        
        try:
            workflow_data = {
                "name": "Test ML Workflow",
                "description": "End-to-end test workflow",
                "steps": [
                    {
                        "name": "data_analysis",
                        "agent": "eda",
                        "parameters": {"dataset": "iris_test"}
                    },
                    {
                        "name": "model_training",
                        "agent": "ml",
                        "parameters": {"target": "target", "algorithm": "random_forest"}
                    }
                ]
            }
            
            response = await self.http_client.post(
                f"{self.dashboard_backend_url}/runs",
                json=workflow_data
            )
            
            assert response.status_code == 200
            result = response.json()
            self.workflow_id = result.get('workflow_id')
            self.run_id = result.get('run_id')
            
            logger.info(f"✅ Workflow created: {self.workflow_id}")
            self.results['workflows']['creation'] = 'PASS'
            
        except Exception as e:
            logger.error(f"❌ Workflow creation failed: {e}")
            self.results['workflows']['creation'] = 'FAIL'
            raise
    
    async def monitor_workflow_events(self, duration: int = 30):
        """Monitor workflow events via WebSocket."""
        logger.info(f"Monitoring workflow events for {duration} seconds...")
        
        events_received = []
        
        try:
            async with websockets.connect(f"ws://localhost:8000/ws/events") as websocket:
                start_time = time.time()
                
                while time.time() - start_time < duration:
                    try:
                        # Wait for events with timeout
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        event = json.loads(message)
                        events_received.append(event)
                        logger.info(f"📡 Event received: {event.get('type', 'unknown')}")
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        logger.error(f"Error receiving event: {e}")
                        break
                
                logger.info(f"✅ Received {len(events_received)} events")
                self.results['events']['websocket'] = 'PASS'
                self.results['events']['count'] = len(events_received)
                
        except Exception as e:
            logger.error(f"❌ Event monitoring failed: {e}")
            self.results['events']['websocket'] = 'FAIL'
    
    async def test_workflow_execution(self):
        """Test workflow execution and monitoring."""
        logger.info("Testing workflow execution...")
        
        try:
            # Start event monitoring in background
            event_task = asyncio.create_task(self.monitor_workflow_events(30))
            
            # Wait a bit for events to start
            await asyncio.sleep(5)
            
            # Check workflow status
            if self.run_id:
                response = await self.http_client.get(f"{self.dashboard_backend_url}/runs/{self.run_id}")
                assert response.status_code == 200
                run_data = response.json()
                logger.info(f"✅ Workflow status: {run_data.get('status', 'unknown')}")
            
            # Wait for event monitoring to complete
            await event_task
            
            self.results['workflows']['execution'] = 'PASS'
            
        except Exception as e:
            logger.error(f"❌ Workflow execution test failed: {e}")
            self.results['workflows']['execution'] = 'FAIL'
            raise
    
    async def test_data_persistence(self):
        """Test data persistence across services."""
        logger.info("Testing data persistence...")
        
        try:
            # Check if runs are persisted in dashboard
            response = await self.http_client.get(f"{self.dashboard_backend_url}/runs")
            assert response.status_code == 200
            runs_data = response.json()
            
            if runs_data.get('runs'):
                logger.info(f"✅ Found {len(runs_data['runs'])} persisted runs")
                self.results['dashboard']['persistence'] = 'PASS'
            else:
                logger.warning("⚠️ No runs found in persistence")
                self.results['dashboard']['persistence'] = 'WARNING'
            
        except Exception as e:
            logger.error(f"❌ Data persistence test failed: {e}")
            self.results['dashboard']['persistence'] = 'FAIL'
            raise
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze test results and generate summary."""
        logger.info("Analyzing test results...")
        
        # Calculate overall success rate
        total_tests = 0
        passed_tests = 0
        
        for category, tests in self.results.items():
            if isinstance(tests, dict):
                for test_name, result in tests.items():
                    if test_name != 'overall':
                        total_tests += 1
                        if result == 'PASS':
                            passed_tests += 1
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Determine overall status
        if success_rate >= 90:
            overall_status = 'PASS'
        elif success_rate >= 70:
            overall_status = 'WARNING'
        else:
            overall_status = 'FAIL'
        
        self.results['overall'] = {
            'success_rate': success_rate,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'status': overall_status,
            'timestamp': datetime.now().isoformat()
        }
        
        return self.results
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*80)
        print("DEEPLINE DASHBOARD END-TO-END TEST SUMMARY")
        print("="*80)
        
        # Infrastructure
        print("\n🏗️  INFRASTRUCTURE:")
        for service, status in self.results.get('infrastructure', {}).items():
            if service != 'overall':
                print(f"  {service:20} : {status}")
        
        # Services
        print("\n🔧 SERVICES:")
        for service, status in self.results.get('services', {}).items():
            if service != 'overall':
                print(f"  {service:20} : {status}")
        
        # Dashboard
        print("\n📊 DASHBOARD:")
        for test, status in self.results.get('dashboard', {}).items():
            if test != 'overall':
                print(f"  {test:20} : {status}")
        
        # Workflows
        print("\n🔄 WORKFLOWS:")
        for test, status in self.results.get('workflows', {}).items():
            if test != 'overall':
                print(f"  {test:20} : {status}")
        
        # Events
        print("\n📡 EVENTS:")
        for test, status in self.results.get('events', {}).items():
            if test != 'overall':
                print(f"  {test:20} : {status}")
        
        # Overall
        overall = self.results.get('overall', {})
        print("\n📈 OVERALL RESULTS:")
        print(f"  Success Rate        : {overall.get('success_rate', 0):.1f}%")
        print(f"  Total Tests         : {overall.get('total_tests', 0)}")
        print(f"  Passed Tests        : {overall.get('passed_tests', 0)}")
        print(f"  Overall Status      : {overall.get('status', 'UNKNOWN')}")
        
        print("\n" + "="*80)
    
    async def run_complete_test(self) -> Dict[str, Any]:
        """Run the complete end-to-end test."""
        logger.info("Starting Deepline Dashboard End-to-End Test")
        logger.info("="*60)
        
        try:
            # Step 1: Start infrastructure
            await self.start_infrastructure()
            
            # Step 2: Start services
            await self.start_services()
            
            # Step 3: Create test data
            dataset_path = self.create_test_data()
            
            # Step 4: Test individual services
            await self.test_eda_agent(dataset_path)
            await self.test_ml_agent()
            await self.test_dashboard_backend()
            await self.test_dashboard_frontend()
            
            # Step 5: Test workflow creation and execution
            await self.create_workflow()
            await self.test_workflow_execution()
            
            # Step 6: Test data persistence
            await self.test_data_persistence()
            
            # Step 7: Analyze results
            results = self.analyze_results()
            
            # Step 8: Print summary
            self.print_summary()
            
            logger.info("✅ End-to-end test completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"❌ End-to-end test failed: {e}")
            self.analyze_results()
            self.print_summary()
            raise

async def main():
    """Main test runner."""
    async with DashboardE2ETest() as test:
        try:
            results = await test.run_complete_test()
            
            # Exit with appropriate code
            if results['overall']['status'] == 'PASS':
                sys.exit(0)
            elif results['overall']['status'] == 'WARNING':
                sys.exit(1)
            else:
                sys.exit(2)
                
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Test failed with exception: {e}")
            sys.exit(2)

if __name__ == "__main__":
    asyncio.run(main()) 