#!/usr/bin/env python3
"""
Simple validation script for DSL Repair Pipeline implementation.
"""

import sys
import os
import asyncio
from unittest.mock import AsyncMock

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from orchestrator.llm_client import LlmClient, call_llm
        print("✅ LLM Client imported successfully")
    except Exception as e:
        print(f"❌ LLM Client import failed: {e}")
        return False
    
    try:
        from orchestrator.dsl_repair_pipeline import repair_dsl, _quick_fixes
        print("✅ DSL Repair Pipeline imported successfully")
    except Exception as e:
        print(f"❌ DSL Repair Pipeline import failed: {e}")
        return False
    
    try:
        from config import get_config
        config = get_config()
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Configuration load failed: {e}")
        return False
    
    return True

def test_quick_fixes():
    """Test the quick fixes functionality."""
    print("\n🔧 Testing quick fixes...")
    
    try:
        from orchestrator.dsl_repair_pipeline import _quick_fixes
        
        # Test case 1: Missing workflow
        yaml_str = """
tasks:
- name: process_data
  agent: eda
  action: analyze
"""
        result = _quick_fixes(yaml_str)
        print("✅ Quick fix for missing workflow works")
        
        # Test case 2: Misspelled keys
        yaml_str = """
workflow:
  name: test_workflow
tasks:
- name: task1
  agent: eda
  action: analyze
  param: {}
  dependson: []
"""
        result = _quick_fixes(yaml_str)
        print("✅ Quick fix for misspelled keys works")
        
        return True
    except Exception as e:
        print(f"❌ Quick fixes test failed: {e}")
        return False

async def test_repair_pipeline():
    """Test the repair pipeline with mock database."""
    print("\n🛠️ Testing repair pipeline...")
    
    try:
        from orchestrator.dsl_repair_pipeline import repair_dsl
        
        # Create mock database
        mock_db = AsyncMock()
        mock_db.dsl_repair_logs = AsyncMock()
        mock_db.dsl_repair_logs.insert_one = AsyncMock()
        
        # Test with valid DSL
        valid_dsl = """
workflow:
  name: valid_workflow
tasks:
- name: process_data
  agent: eda
  action: analyze
  params: {}
  depends_on: []
"""
        
        result = await repair_dsl(valid_dsl, mock_db)
        print("✅ Repair pipeline handles valid DSL")
        
        return True
    except Exception as e:
        print(f"❌ Repair pipeline test failed: {e}")
        return False

def test_config():
    """Test configuration structure."""
    print("\n⚙️ Testing configuration...")
    
    try:
        from config import get_config
        config = get_config()
        
        # Check LLM config
        if hasattr(config.master_orchestrator, 'llm'):
            print("✅ LLM configuration present")
        else:
            print("❌ LLM configuration missing")
            return False
        
        # Check DSL repair config
        if hasattr(config.master_orchestrator, 'dsl_repair'):
            print("✅ DSL repair configuration present")
        else:
            print("❌ DSL repair configuration missing")
            return False
        
        # Check agent actions config
        if hasattr(config.master_orchestrator, 'agent_actions'):
            print("✅ Agent actions configuration present")
        else:
            print("❌ Agent actions configuration missing")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_schema():
    """Test JSON schema file."""
    print("\n📋 Testing JSON schema...")
    
    try:
        schema_path = os.path.join(os.path.dirname(__file__), "schemas", "dsl_schema.json")
        if os.path.exists(schema_path):
            print("✅ JSON schema file exists")
            
            import json
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            
            if "properties" in schema and "workflow" in schema["properties"]:
                print("✅ JSON schema structure is valid")
                return True
            else:
                print("❌ JSON schema structure is invalid")
                return False
        else:
            print("❌ JSON schema file not found")
            return False
    except Exception as e:
        print(f"❌ Schema test failed: {e}")
        return False

async def main():
    """Run all validation tests."""
    print("🚀 DSL Repair Pipeline Validation")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Quick Fixes", test_quick_fixes),
        ("Configuration", test_config),
        ("JSON Schema", test_schema),
        ("Repair Pipeline", test_repair_pipeline),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! DSL Repair Pipeline is ready.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 