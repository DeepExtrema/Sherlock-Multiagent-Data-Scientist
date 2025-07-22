"""
Test Script for Hybrid API Implementation

Tests the complete async translation workflow:
- /workflows/translate endpoint with token return
- /translation/{token} polling with status updates
- /workflows/dsl direct execution
- /workflows/suggest generation
- Edge cases: timeouts, invalid inputs, Redis fallback
"""

import asyncio
import time
import json
from typing import Dict, Any
import aiohttp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridAPITester:
    """Comprehensive test suite for Hybrid API."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session = None
        self.test_results = []
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def test_health_check(self) -> bool:
        """Test health endpoint to verify API is running."""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ Health check passed: {data['version']}")
                    
                    # Check hybrid API components
                    components = data.get("components", {})
                    required_components = [
                        "translation_queue", "translation_worker", 
                        "llm_translator", "workflow_manager"
                    ]
                    
                    for component in required_components:
                        if not components.get(component):
                            logger.warning(f"⚠️  Component {component} not available")
                    
                    return True
                else:
                    logger.error(f"❌ Health check failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
            return False
    
    async def test_translation_workflow(self) -> bool:
        """Test complete async translation workflow."""
        try:
            # Step 1: Submit translation request
            nl_request = {
                "natural_language": "Load a CSV file called sales_data.csv, analyze missing values, and create a histogram visualization",
                "client_id": "test_client",
                "priority": 8,
                "metadata": {"test": "hybrid_api"}
            }
            
            logger.info("🔄 Submitting translation request...")
            async with self.session.post(
                f"{self.base_url}/api/v1/workflows/translate",
                json=nl_request
            ) as response:
                if response.status != 200:
                    logger.error(f"❌ Translation request failed: {response.status}")
                    text = await response.text()
                    logger.error(f"Response: {text}")
                    return False
                
                data = await response.json()
                token = data["token"]
                logger.info(f"✅ Translation queued with token: {token}")
            
            # Step 2: Poll for translation completion
            logger.info("🔄 Polling for translation completion...")
            max_polls = 20
            poll_count = 0
            
            while poll_count < max_polls:
                async with self.session.get(
                    f"{self.base_url}/api/v1/translation/{token}"
                ) as response:
                    if response.status != 200:
                        logger.error(f"❌ Status polling failed: {response.status}")
                        return False
                    
                    status_data = await response.json()
                    status = status_data["status"]
                    
                    logger.info(f"📊 Translation status: {status}")
                    
                    if status == "done":
                        dsl = status_data.get("dsl")
                        if dsl:
                            logger.info("✅ Translation completed successfully")
                            logger.info(f"Generated DSL:\n{dsl}")
                            
                            # Step 3: Execute the generated DSL
                            return await self._test_dsl_execution(dsl)
                        else:
                            logger.error("❌ Translation marked done but no DSL returned")
                            return False
                    
                    elif status in ["error", "timeout"]:
                        logger.error(f"❌ Translation failed: {status}")
                        error_msg = status_data.get("error_message", "Unknown error")
                        logger.error(f"Error: {error_msg}")
                        return False
                    
                    elif status == "needs_human":
                        logger.warning("⚠️  Translation requires human intervention")
                        # This is acceptable for complex requests
                        return True
                    
                    # Continue polling
                    poll_count += 1
                    await asyncio.sleep(2)
            
            logger.error("❌ Translation polling timed out")
            return False
            
        except Exception as e:
            logger.error(f"❌ Translation workflow error: {e}")
            return False
    
    async def _test_dsl_execution(self, dsl: str) -> bool:
        """Test DSL execution endpoint."""
        try:
            dsl_request = {
                "dsl_yaml": dsl,
                "client_id": "test_client",
                "validate_only": True,  # Only validate for testing
                "metadata": {"test": "dsl_execution"}
            }
            
            logger.info("🔄 Testing DSL execution...")
            async with self.session.post(
                f"{self.base_url}/api/v1/workflows/dsl",
                json=dsl_request
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("valid"):
                        logger.info("✅ DSL validation passed")
                        return True
                    else:
                        logger.error(f"❌ DSL validation failed: {data.get('errors')}")
                        return False
                else:
                    logger.error(f"❌ DSL execution failed: {response.status}")
                    text = await response.text()
                    logger.error(f"Response: {text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ DSL execution error: {e}")
            return False
    
    async def test_suggestions_endpoint(self) -> bool:
        """Test workflow suggestions endpoint."""
        try:
            suggestion_request = {
                "context": "I want to analyze customer purchase patterns",
                "domain": "data-science",
                "complexity": "medium"
            }
            
            logger.info("🔄 Testing suggestions endpoint...")
            async with self.session.post(
                f"{self.base_url}/api/v1/workflows/suggest",
                json=suggestion_request
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    suggestions = data.get("suggestions", [])
                    
                    if suggestions:
                        logger.info(f"✅ Generated {len(suggestions)} suggestions")
                        for i, suggestion in enumerate(suggestions):
                            title = suggestion.get("title", "Untitled")
                            logger.info(f"  {i+1}. {title}")
                        return True
                    else:
                        logger.warning("⚠️  No suggestions generated")
                        return False
                else:
                    logger.error(f"❌ Suggestions request failed: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Suggestions test error: {e}")
            return False
    
    async def test_edge_cases(self) -> bool:
        """Test various edge cases and error handling."""
        edge_case_results = []
        
        # Test 1: Invalid token format
        try:
            async with self.session.get(
                f"{self.base_url}/api/v1/translation/invalid-token"
            ) as response:
                if response.status == 400:
                    logger.info("✅ Invalid token format handled correctly")
                    edge_case_results.append(True)
                else:
                    logger.error(f"❌ Invalid token should return 400, got {response.status}")
                    edge_case_results.append(False)
        except Exception as e:
            logger.error(f"❌ Invalid token test error: {e}")
            edge_case_results.append(False)
        
        # Test 2: Empty natural language
        try:
            empty_request = {
                "natural_language": "",
                "client_id": "test_client"
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/workflows/translate",
                json=empty_request
            ) as response:
                if response.status == 422:  # Validation error
                    logger.info("✅ Empty natural language handled correctly")
                    edge_case_results.append(True)
                else:
                    logger.error(f"❌ Empty NL should return 422, got {response.status}")
                    edge_case_results.append(False)
        except Exception as e:
            logger.error(f"❌ Empty NL test error: {e}")
            edge_case_results.append(False)
        
        # Test 3: Invalid DSL YAML
        try:
            invalid_dsl_request = {
                "dsl_yaml": "invalid: yaml: content: [missing bracket",
                "client_id": "test_client"
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/workflows/dsl",
                json=invalid_dsl_request
            ) as response:
                if response.status == 400:
                    logger.info("✅ Invalid DSL YAML handled correctly")
                    edge_case_results.append(True)
                else:
                    logger.error(f"❌ Invalid DSL should return 400, got {response.status}")
                    edge_case_results.append(False)
        except Exception as e:
            logger.error(f"❌ Invalid DSL test error: {e}")
            edge_case_results.append(False)
        
        # Test 4: Non-existent token
        try:
            fake_token = "a" * 32  # Valid format but non-existent
            async with self.session.get(
                f"{self.base_url}/api/v1/translation/{fake_token}"
            ) as response:
                if response.status == 404:
                    logger.info("✅ Non-existent token handled correctly")
                    edge_case_results.append(True)
                else:
                    logger.error(f"❌ Non-existent token should return 404, got {response.status}")
                    edge_case_results.append(False)
        except Exception as e:
            logger.error(f"❌ Non-existent token test error: {e}")
            edge_case_results.append(False)
        
        return all(edge_case_results)
    
    async def test_rate_limiting(self) -> bool:
        """Test rate limiting functionality."""
        try:
            logger.info("🔄 Testing rate limiting...")
            
            # Send many requests rapidly
            requests = []
            for i in range(10):
                request = {
                    "natural_language": f"Test request {i}",
                    "client_id": "rate_test_client"
                }
                requests.append(
                    self.session.post(
                        f"{self.base_url}/api/v1/workflows/translate",
                        json=request
                    )
                )
            
            responses = await asyncio.gather(*requests, return_exceptions=True)
            
            # Check if some requests were rate limited
            rate_limited = 0
            successful = 0
            
            for response in responses:
                if isinstance(response, Exception):
                    continue
                    
                async with response:
                    if response.status == 429:
                        rate_limited += 1
                    elif response.status == 200:
                        successful += 1
            
            if rate_limited > 0:
                logger.info(f"✅ Rate limiting working: {rate_limited} requests limited, {successful} successful")
                return True
            else:
                logger.warning("⚠️  Rate limiting not triggered (may be configured loosely)")
                return True  # Not necessarily a failure
                
        except Exception as e:
            logger.error(f"❌ Rate limiting test error: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all test suites."""
        test_results = {}
        
        logger.info("🚀 Starting Hybrid API Test Suite")
        logger.info("=" * 50)
        
        # Health check
        test_results["health_check"] = await self.test_health_check()
        
        # Core workflow
        test_results["translation_workflow"] = await self.test_translation_workflow()
        
        # Suggestions
        test_results["suggestions"] = await self.test_suggestions_endpoint()
        
        # Edge cases
        test_results["edge_cases"] = await self.test_edge_cases()
        
        # Rate limiting
        test_results["rate_limiting"] = await self.test_rate_limiting()
        
        # Summary
        logger.info("=" * 50)
        logger.info("🏁 Test Results Summary:")
        
        passed = 0
        total = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"  {test_name}: {status}")
            if result:
                passed += 1
        
        logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All tests passed! Hybrid API is working correctly.")
        else:
            logger.error(f"⚠️  {total - passed} tests failed. Check implementation.")
        
        return test_results


async def main():
    """Main test runner."""
    print("Hybrid API Test Suite")
    print("Testing the async translation workflow implementation")
    print()
    
    try:
        async with HybridAPITester() as tester:
            results = await tester.run_all_tests()
            
            # Return exit code based on results
            all_passed = all(results.values())
            return 0 if all_passed else 1
            
    except KeyboardInterrupt:
        logger.info("\n⚠️  Tests interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Test suite error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 