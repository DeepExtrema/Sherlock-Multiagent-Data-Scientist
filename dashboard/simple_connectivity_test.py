#!/usr/bin/env python3
"""
Simple Connectivity Test for Deepline Dashboard
Tests connectivity to existing services and basic functionality.
"""

import asyncio
import json
import logging
import time
import httpx
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleConnectivityTest:
    """Simple connectivity test for dashboard components."""
    
    def __init__(self):
        # Service URLs
        self.dashboard_backend_url = "http://localhost:8000"
        self.dashboard_frontend_url = "http://localhost:3000"
        self.eda_agent_url = "http://localhost:8001"
        self.ml_agent_url = "http://localhost:8002"
        
        self.http_client = httpx.AsyncClient(timeout=30)
        
        # Test results
        self.results = {
            'connectivity': {},
            'services': {},
            'overall': {}
        }
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.aclose()
    
    async def test_service_connectivity(self, service_name: str, url: str, endpoint: str = "/health"):
        """Test if a service is reachable."""
        logger.info(f"Testing connectivity to {service_name}...")
        
        try:
            response = await self.http_client.get(f"{url}{endpoint}")
            if response.status_code == 200:
                logger.info(f"✅ {service_name} is reachable")
                self.results['connectivity'][service_name] = 'PASS'
                return True
            else:
                logger.warning(f"⚠️ {service_name} returned status {response.status_code}")
                self.results['connectivity'][service_name] = 'WARNING'
                return False
        except Exception as e:
            logger.error(f"❌ {service_name} is not reachable: {e}")
            self.results['connectivity'][service_name] = 'FAIL'
            return False
    
    async def test_dashboard_backend(self):
        """Test dashboard backend functionality."""
        logger.info("Testing Dashboard Backend...")
        
        try:
            # Test root endpoint
            response = await self.http_client.get(f"{self.dashboard_backend_url}/")
            if response.status_code == 200:
                logger.info("✅ Dashboard Backend root endpoint working")
                
                # Test runs endpoint
                response = await self.http_client.get(f"{self.dashboard_backend_url}/runs")
                if response.status_code == 200:
                    logger.info("✅ Dashboard Backend runs endpoint working")
                    self.results['services']['dashboard_backend'] = 'PASS'
                else:
                    logger.warning(f"⚠️ Dashboard Backend runs endpoint returned status {response.status_code}")
                    self.results['services']['dashboard_backend'] = 'WARNING'
            else:
                logger.error(f"❌ Dashboard Backend root endpoint returned status {response.status_code}")
                self.results['services']['dashboard_backend'] = 'FAIL'
                
        except Exception as e:
            logger.error(f"❌ Dashboard Backend test failed: {e}")
            self.results['services']['dashboard_backend'] = 'FAIL'
    
    async def test_eda_agent(self):
        """Test EDA Agent functionality."""
        logger.info("Testing EDA Agent...")
        
        try:
            # Test health endpoint
            response = await self.http_client.get(f"{self.eda_agent_url}/health")
            if response.status_code == 200:
                logger.info("✅ EDA Agent health endpoint working")
                
                # Test datasets endpoint
                response = await self.http_client.get(f"{self.eda_agent_url}/datasets")
                if response.status_code == 200:
                    logger.info("✅ EDA Agent datasets endpoint working")
                    self.results['services']['eda_agent'] = 'PASS'
                else:
                    logger.warning(f"⚠️ EDA Agent datasets endpoint returned status {response.status_code}")
                    self.results['services']['eda_agent'] = 'WARNING'
            else:
                logger.error(f"❌ EDA Agent health endpoint returned status {response.status_code}")
                self.results['services']['eda_agent'] = 'FAIL'
                
        except Exception as e:
            logger.error(f"❌ EDA Agent test failed: {e}")
            self.results['services']['eda_agent'] = 'FAIL'
    
    async def test_ml_agent(self):
        """Test ML Agent functionality."""
        logger.info("Testing ML Agent...")
        
        try:
            # Test health endpoint
            response = await self.http_client.get(f"{self.ml_agent_url}/health")
            if response.status_code == 200:
                logger.info("✅ ML Agent health endpoint working")
                
                # Test metrics endpoint
                response = await self.http_client.get(f"{self.ml_agent_url}/metrics")
                if response.status_code == 200:
                    logger.info("✅ ML Agent metrics endpoint working")
                    self.results['services']['ml_agent'] = 'PASS'
                else:
                    logger.warning(f"⚠️ ML Agent metrics endpoint returned status {response.status_code}")
                    self.results['services']['ml_agent'] = 'WARNING'
            else:
                logger.error(f"❌ ML Agent health endpoint returned status {response.status_code}")
                self.results['services']['ml_agent'] = 'FAIL'
                
        except Exception as e:
            logger.error(f"❌ ML Agent test failed: {e}")
            self.results['services']['ml_agent'] = 'FAIL'
    
    async def test_dashboard_frontend(self):
        """Test dashboard frontend accessibility."""
        logger.info("Testing Dashboard Frontend...")
        
        try:
            response = await self.http_client.get(f"{self.dashboard_frontend_url}/")
            if response.status_code == 200:
                logger.info("✅ Dashboard Frontend is accessible")
                self.results['services']['dashboard_frontend'] = 'PASS'
            else:
                logger.warning(f"⚠️ Dashboard Frontend returned status {response.status_code}")
                self.results['services']['dashboard_frontend'] = 'WARNING'
                
        except Exception as e:
            logger.error(f"❌ Dashboard Frontend test failed: {e}")
            self.results['services']['dashboard_frontend'] = 'FAIL'
    
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
        print("DEEPLINE DASHBOARD CONNECTIVITY TEST SUMMARY")
        print("="*80)
        
        # Connectivity
        print("\n🔌 CONNECTIVITY:")
        for service, status in self.results.get('connectivity', {}).items():
            if service != 'overall':
                print(f"  {service:20} : {status}")
        
        # Services
        print("\n🔧 SERVICES:")
        for service, status in self.results.get('services', {}).items():
            if service != 'overall':
                print(f"  {service:20} : {status}")
        
        # Overall
        overall = self.results.get('overall', {})
        print("\n📈 OVERALL RESULTS:")
        print(f"  Success Rate        : {overall.get('success_rate', 0):.1f}%")
        print(f"  Total Tests         : {overall.get('total_tests', 0)}")
        print(f"  Passed Tests        : {overall.get('passed_tests', 0)}")
        print(f"  Overall Status      : {overall.get('status', 'UNKNOWN')}")
        
        print("\n" + "="*80)
    
    async def run_complete_test(self) -> Dict[str, Any]:
        """Run the complete connectivity test."""
        logger.info("Starting Deepline Dashboard Connectivity Test")
        logger.info("="*60)
        
        try:
            # Step 1: Test connectivity to all services
            await self.test_service_connectivity("Dashboard Backend", self.dashboard_backend_url, "/")
            await self.test_service_connectivity("Dashboard Frontend", self.dashboard_frontend_url, "/")
            await self.test_service_connectivity("EDA Agent", self.eda_agent_url, "/health")
            await self.test_service_connectivity("ML Agent", self.ml_agent_url, "/health")
            
            # Step 2: Test service functionality
            await self.test_dashboard_backend()
            await self.test_dashboard_frontend()
            await self.test_eda_agent()
            await self.test_ml_agent()
            
            # Step 3: Analyze results
            results = self.analyze_results()
            
            # Step 4: Print summary
            self.print_summary()
            
            logger.info("✅ Connectivity test completed!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Connectivity test failed: {e}")
            self.analyze_results()
            self.print_summary()
            raise

async def main():
    """Main test runner."""
    async with SimpleConnectivityTest() as test:
        try:
            results = await test.run_complete_test()
            
            # Exit with appropriate code
            if results['overall']['status'] == 'PASS':
                print("\n🎉 All services are working correctly!")
                return 0
            elif results['overall']['status'] == 'WARNING':
                print("\n⚠️ Some services have issues but the system is mostly functional.")
                return 1
            else:
                print("\n❌ Multiple services are not working correctly.")
                return 2
                
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
            return 1
        except Exception as e:
            logger.error(f"Test failed with exception: {e}")
            return 2

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code) 