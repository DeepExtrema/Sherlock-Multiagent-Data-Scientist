#!/usr/bin/env python3
"""
Validation script for Agent Routing implementation.
"""

import sys
import os
import asyncio

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_agent_registry():
    """Test the agent registry functionality."""
    print("🔍 Testing Agent Registry...")
    
    try:
        from orchestrator.agent_registry import (
            get_agent_matrix, get_agent_names, get_agent_actions,
            is_valid_agent, is_valid_action, is_valid,
            get_agent_stats, validate_workflow_tasks
        )
        print("✅ Agent registry imported successfully")
        
        # Test agent matrix
        matrix = get_agent_matrix()
        print(f"✅ Agent matrix loaded: {list(matrix.keys())}")
        
        # Test agent names
        names = get_agent_names()
        print(f"✅ Agent names: {list(names)}")
        
        # Test validation
        assert is_valid("eda", "analyze") == True
        assert is_valid("eda", "train") == False
        print("✅ Agent validation working")
        
        # Test workflow validation
        valid_tasks = [{"agent": "eda", "action": "analyze"}]
        errors = validate_workflow_tasks(valid_tasks)
        assert len(errors) == 0
        print("✅ Workflow validation working")
        
        return True
    except Exception as e:
        print(f"❌ Agent registry test failed: {e}")
        return False

def test_configuration():
    """Test agent routing configuration."""
    print("\n⚙️ Testing Configuration...")
    
    try:
        from config import get_config
        config = get_config()
        
        # Check agent routing config
        if hasattr(config.master_orchestrator, 'agent_routing'):
            print("✅ Agent routing configuration present")
            print(f"   Mode: {config.master_orchestrator.agent_routing.mode}")
            print(f"   Default topic: {config.master_orchestrator.agent_routing.default_topic}")
            print(f"   Topic prefix: {config.master_orchestrator.agent_routing.topic_prefix}")
        else:
            print("❌ Agent routing configuration missing")
            return False
        
        # Check agent actions config
        if hasattr(config.master_orchestrator, 'agent_actions'):
            print("✅ Agent actions configuration present")
            for agent, actions in config.master_orchestrator.agent_actions.__dict__.items():
                print(f"   {agent}: {len(actions)} actions")
        else:
            print("❌ Agent actions configuration missing")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_api_endpoints():
    """Test agent API endpoints."""
    print("\n🌐 Testing API Endpoints...")
    
    try:
        from api.agent_router import create_agent_router
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        
        app = FastAPI()
        router = create_agent_router()
        app.include_router(router)
        
        client = TestClient(app)
        
        # Test GET /agents/
        response = client.get("/agents/")
        if response.status_code == 200:
            print("✅ GET /agents/ endpoint working")
        else:
            print(f"❌ GET /agents/ failed: {response.status_code}")
            return False
        
        # Test GET /agents/eda
        response = client.get("/agents/eda")
        if response.status_code == 200:
            print("✅ GET /agents/{agent} endpoint working")
        else:
            print(f"❌ GET /agents/eda failed: {response.status_code}")
            return False
        
        # Test POST /agents/validate
        response = client.post("/agents/validate", json={
            "agent": "eda",
            "action": "analyze"
        })
        if response.status_code == 200:
            print("✅ POST /agents/validate endpoint working")
        else:
            print(f"❌ POST /agents/validate failed: {response.status_code}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ API endpoints test failed: {e}")
        return False

def test_integration():
    """Test integration with existing systems."""
    print("\n🔗 Testing Integration...")
    
    try:
        # Test DSL repair integration
        from orchestrator.dsl_repair_pipeline import _validate_agent_action
        
        assert _validate_agent_action("eda", "analyze", None) == True
        assert _validate_agent_action("eda", "train", None) == False
        print("✅ DSL repair integration working")
        
        # Test hybrid router integration
        from api.hybrid_router import validate_workflow_tasks
        
        valid_tasks = [{"agent": "eda", "action": "analyze"}]
        errors = validate_workflow_tasks(valid_tasks)
        assert len(errors) == 0
        print("✅ Hybrid router integration working")
        
        return True
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

async def main():
    """Run all validation tests."""
    print("🚀 Agent Routing Validation")
    print("=" * 50)
    
    tests = [
        ("Agent Registry", test_agent_registry),
        ("Configuration", test_configuration),
        ("API Endpoints", test_api_endpoints),
        ("Integration", test_integration),
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
        print("🎉 All tests passed! Agent Routing is ready.")
        print("\n📋 Next Steps:")
        print("1. ✅ Agent Registry - Complete")
        print("2. ✅ Configuration - Complete")
        print("3. ✅ API Endpoints - Complete")
        print("4. ✅ Integration - Complete")
        print("5. 🔄 Task Header Generator - Ready for implementation")
        print("6. 🔄 Worker Pool Filter - Ready for implementation")
        print("7. 🔄 Kafka Topic Routing - Ready for implementation")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 