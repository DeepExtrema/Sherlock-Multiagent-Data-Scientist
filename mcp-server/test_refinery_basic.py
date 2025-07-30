#!/usr/bin/env python3
"""
Basic Refinery Agent Test

A lightweight test script that validates the refinery agent implementation
without requiring heavy ML dependencies.
"""

import json
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_file_structure():
    """Test that all expected files exist."""
    logger.info("Testing file structure...")
    
    expected_files = [
        'refinery_agent.py',
        'orchestrator/refinery_agent_integration.py',
        'config.yaml'
    ]
    
    all_exist = True
    for file_path in expected_files:
        if Path(file_path).exists():
            logger.info(f"✓ {file_path} exists")
        else:
            logger.error(f"✗ {file_path} missing")
            all_exist = False
    
    return all_exist

def test_refinery_agent_structure():
    """Test refinery agent Python file structure."""
    logger.info("Testing refinery agent structure...")
    
    try:
        with open('refinery_agent.py', 'r') as f:
            content = f.read()
        
        # Check for key components
        required_components = [
            'class TaskRequest',
            'class TaskResponse', 
            'async def execute',
            '/health',
            '/execute',
            'check_schema_consistency',
            'check_missing_values',
            'assign_feature_roles',
            'save_fe_pipeline'
        ]
        
        missing_components = []
        for component in required_components:
            if component not in content:
                missing_components.append(component)
        
        if missing_components:
            logger.error(f"Missing components in refinery_agent.py: {missing_components}")
            return False
        
        logger.info("✓ Refinery agent structure test passed")
        return True
        
    except FileNotFoundError:
        logger.error("refinery_agent.py not found")
        return False
    except Exception as e:
        logger.error(f"Refinery agent structure test failed: {e}")
        return False

def run_all_tests():
    """Run all validation tests."""
    logger.info("=" * 50)
    logger.info("REFINERY AGENT VALIDATION TESTS")
    logger.info("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Refinery Agent Structure", test_refinery_agent_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            if test_func():
                logger.info(f"✓ {test_name}: PASSED")
                passed += 1
            else:
                logger.error(f"✗ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name}: ERROR - {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Passed: {passed}/{total}")
    logger.info(f"Failed: {total - passed}/{total}")
    logger.info(f"Success Rate: {passed/total:.1%}")
    
    if passed == total:
        logger.info("🎉 All validation tests passed!")
        return True
    else:
        logger.error("❌ Some validation tests failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)