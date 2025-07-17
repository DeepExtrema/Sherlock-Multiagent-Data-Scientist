"""
Test script for Master Orchestrator

This script tests the basic functionality of the Master Orchestrator components.
"""

import asyncio
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_security_utils():
    """Test security utilities."""
    logger.info("Testing SecurityUtils...")
    
    try:
        from orchestrator.security import SecurityUtils
        
        security = SecurityUtils()
        
        # Test input sanitization
        dangerous_input = "<script>alert('xss')</script>Load data from users.csv"
        clean_input = security.sanitize_input(dangerous_input)
        assert "script" not in clean_input
        assert "users.csv" in clean_input
        
        # Test context minimization
        long_text = "Load the customer data. Analyze the sales patterns. Create visualizations. Train a model."
        minimal = security.minimize_context(long_text, max_sentences=2)
        assert len(minimal) < len(long_text)
        
        logger.info("✅ SecurityUtils test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ SecurityUtils test failed: {e}")
        return False

async def test_cache_client():
    """Test cache client."""
    logger.info("Testing CacheClient...")
    
    try:
        from orchestrator.cache_client import CacheClient
        
        cache = CacheClient(namespace="test")
        
        # Test set/get
        test_key = "test_key"
        test_value = {"data": "test_value", "timestamp": datetime.now().isoformat()}
        
        await cache.set(test_key, test_value, ttl=60)
        retrieved = await cache.get(test_key)
        
        assert retrieved == test_value
        
        # Test delete
        await cache.delete(test_key)
        retrieved_after_delete = await cache.get(test_key)
        assert retrieved_after_delete is None
        
        # Test health check
        health = await cache.health_check()
        assert isinstance(health, bool)
        
        logger.info("✅ CacheClient test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ CacheClient test failed: {e}")
        return False

async def test_guards():
    """Test concurrency guard and rate limiter."""
    logger.info("Testing Guards...")
    
    try:
        from orchestrator.guards import ConcurrencyGuard, TokenRateLimiter
        
        # Test concurrency guard
        guard = ConcurrencyGuard(max_concurrent=2)
        
        # Should allow first two
        assert await guard.acquire() == True
        assert await guard.acquire() == True
        
        # Should block third
        assert await guard.acquire() == False
        
        # Release one and should allow again
        await guard.release()
        assert await guard.acquire() == True
        
        # Test rate limiter
        rate_limiter = TokenRateLimiter({"test": 10})  # 10 requests per minute
        
        client_id = "test_client"
        assert rate_limiter.check(client_id) == True
        
        logger.info("✅ Guards test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Guards test failed: {e}")
        return False

async def test_rule_based_translator():
    """Test rule-based translator."""
    logger.info("Testing RuleBasedTranslator...")
    
    try:
        from orchestrator.translator import RuleBasedTranslator
        
        config = {
            "rule_mappings": {
                "load data": {
                    "id": "load_task",
                    "agent": "eda_agent",
                    "action": "load_data",
                    "params": {"file": "data.csv"}
                }
            }
        }
        
        translator = RuleBasedTranslator(config)
        
        # Test translation
        result = translator.translate("Please load data from customers.csv")
        
        assert result is not None
        assert "tasks" in result
        assert len(result["tasks"]) > 0
        assert result["tasks"][0]["agent"] == "eda_agent"
        assert result["tasks"][0]["action"] == "load_data"
        
        logger.info("✅ RuleBasedTranslator test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ RuleBasedTranslator test failed: {e}")
        return False

async def test_workflow_manager():
    """Test workflow manager (basic functionality without DB)."""
    logger.info("Testing WorkflowManager...")
    
    try:
        from orchestrator.workflow_manager import WorkflowManager
        
        config = {
            "mongo_url": "mongodb://localhost:27017",
            "db_name": "test_deepline"
        }
        
        manager = WorkflowManager(config)
        
        # Test workflow definition
        workflow_def = {
            "tasks": [
                {
                    "id": "task1",
                    "agent": "eda_agent",
                    "action": "load_data",
                    "params": {"file": "test.csv"},
                    "depends_on": []
                },
                {
                    "id": "task2", 
                    "agent": "eda_agent",
                    "action": "analyze_data",
                    "params": {},
                    "depends_on": ["task1"]
                }
            ]
        }
        
        # Test workflow initialization (will work even without DB)
        # The actual DB operations will be skipped gracefully
        run_id = await manager.init_workflow(workflow_def)
        assert run_id is not None
        assert run_id.startswith("run_")
        
        # Test statistics
        stats = manager.get_statistics()
        assert "workflows_created" in stats
        
        logger.info("✅ WorkflowManager test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ WorkflowManager test failed: {e}")
        return False

async def test_llm_translator():
    """Test LLM translator (basic functionality)."""
    logger.info("Testing LLMTranslator...")
    
    try:
        from orchestrator.translator import LLMTranslator
        
        config = {
            "model_version": "claude-3-sonnet-20240229",
            "max_input_length": 1000,
            "llm_max_tokens": 4000,
            "llm_max_retries": 3,
            "temperature": 0.0
        }
        
        translator = LLMTranslator(config)
        
        # Test basic initialization
        assert translator.model_version == "claude-3-sonnet-20240229"
        assert translator.max_tokens == 4000
        
        # Test prompt building
        prompt = translator._build_llm_prompt("load data from file")
        assert "load data from file" in prompt
        assert "JSON" in prompt
        
        # Test workflow validation
        valid_workflow = {
            "tasks": [
                {
                    "id": "task1",
                    "agent": "eda_agent", 
                    "action": "load_data",
                    "params": {"file": "test.csv"}
                }
            ]
        }
        
        assert translator._validate_workflow_structure(valid_workflow) == True
        
        # Test invalid workflow
        invalid_workflow = {"invalid": "structure"}
        assert translator._validate_workflow_structure(invalid_workflow) == False
        
        logger.info("✅ LLMTranslator test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ LLMTranslator test failed: {e}")
        return False

async def test_configuration():
    """Test configuration loading."""
    logger.info("Testing Configuration...")
    
    try:
        from config import load_config
        
        # Try to load config
        config = load_config()
        
        # Check basic structure
        assert hasattr(config, 'orchestrator')
        assert hasattr(config, 'master_orchestrator')
        
        # Check orchestrator config
        assert config.orchestrator.max_concurrent_workflows >= 1
        assert config.orchestrator.retry.max_retries >= 0
        
        # Check master orchestrator config
        assert hasattr(config.master_orchestrator, 'llm')
        assert hasattr(config.master_orchestrator, 'rules')
        assert hasattr(config.master_orchestrator, 'infrastructure')
        
        logger.info("✅ Configuration test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False

async def run_all_tests():
    """Run all tests."""
    logger.info("🚀 Starting Master Orchestrator Tests...")
    
    tests = [
        test_configuration,
        test_security_utils,
        test_cache_client,
        test_guards,
        test_rule_based_translator,
        test_llm_translator,
        test_workflow_manager,
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            logger.error(f"Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    logger.info(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Master Orchestrator is ready.")
    else:
        logger.warning("⚠️  Some tests failed. Check the logs above.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1) 