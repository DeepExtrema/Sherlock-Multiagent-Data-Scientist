"""
Unit tests for Agent Routing functionality

Tests the agent registry, validation, and routing capabilities.
"""

import pytest
import sys
import os
from unittest.mock import AsyncMock, patch, MagicMock

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from mcp_server.orchestrator.agent_registry import (
    get_agent_matrix, get_agent_names, get_agent_actions,
    is_valid_agent, is_valid_action, is_valid,
    get_agent_stats, validate_workflow_tasks, refresh_agent_matrix
)

class TestAgentRegistry:
    """Test the agent registry functionality."""
    
    def test_get_agent_matrix(self):
        """Test getting the agent matrix."""
        matrix = get_agent_matrix()
        
        assert isinstance(matrix, dict)
        assert "eda" in matrix
        assert "fe" in matrix
        assert "model" in matrix
        assert "custom" in matrix
        
        # Check that each agent has actions
        for agent, actions in matrix.items():
            assert isinstance(actions, list)
            assert len(actions) > 0
    
    def test_get_agent_names(self):
        """Test getting agent names."""
        names = get_agent_names()
        
        assert isinstance(names, set)
        assert "eda" in names
        assert "fe" in names
        assert "model" in names
        assert "custom" in names
    
    def test_get_agent_actions(self):
        """Test getting actions for specific agents."""
        eda_actions = get_agent_actions("eda")
        fe_actions = get_agent_actions("fe")
        
        assert isinstance(eda_actions, list)
        assert isinstance(fe_actions, list)
        assert "analyze" in eda_actions
        assert "create_visualization" in fe_actions
    
    def test_is_valid_agent(self):
        """Test agent validation."""
        assert is_valid_agent("eda") == True
        assert is_valid_agent("fe") == True
        assert is_valid_agent("model") == True
        assert is_valid_agent("custom") == True
        assert is_valid_agent("unknown") == False
        assert is_valid_agent("") == False
    
    def test_is_valid_action(self):
        """Test action validation."""
        assert is_valid_action("eda", "analyze") == True
        assert is_valid_action("eda", "clean") == True
        assert is_valid_action("fe", "create_visualization") == True
        assert is_valid_action("model", "train") == True
        
        # Invalid combinations
        assert is_valid_action("eda", "train") == False
        assert is_valid_action("fe", "analyze") == False
        assert is_valid_action("unknown", "analyze") == False
    
    def test_is_valid(self):
        """Test agent-action combination validation."""
        assert is_valid("eda", "analyze") == True
        assert is_valid("fe", "create_visualization") == True
        assert is_valid("model", "train") == True
        assert is_valid("custom", "execute") == True
        
        # Invalid combinations
        assert is_valid("eda", "train") == False
        assert is_valid("unknown", "analyze") == False
    
    def test_get_agent_stats(self):
        """Test getting agent statistics."""
        stats = get_agent_stats()
        
        assert isinstance(stats, dict)
        assert "eda" in stats
        assert "fe" in stats
        assert "model" in stats
        assert "custom" in stats
        
        for agent, agent_stats in stats.items():
            assert "actions" in agent_stats
            assert "action_count" in agent_stats
            assert "status" in agent_stats
            assert isinstance(agent_stats["actions"], list)
            assert isinstance(agent_stats["action_count"], int)
            assert agent_stats["status"] == "active"
    
    def test_validate_workflow_tasks(self):
        """Test workflow task validation."""
        # Valid tasks
        valid_tasks = [
            {"agent": "eda", "action": "analyze"},
            {"agent": "fe", "action": "create_visualization"}
        ]
        errors = validate_workflow_tasks(valid_tasks)
        assert len(errors) == 0
        
        # Invalid agent
        invalid_agent_tasks = [
            {"agent": "unknown", "action": "analyze"}
        ]
        errors = validate_workflow_tasks(invalid_agent_tasks)
        assert len(errors) == 1
        assert "Invalid agent" in errors[0]
        
        # Invalid action
        invalid_action_tasks = [
            {"agent": "eda", "action": "train"}
        ]
        errors = validate_workflow_tasks(invalid_action_tasks)
        assert len(errors) == 1
        assert "Invalid action" in errors[0]
        
        # Missing fields
        missing_fields_tasks = [
            {"agent": "eda"},  # missing action
            {"action": "analyze"}  # missing agent
        ]
        errors = validate_workflow_tasks(missing_fields_tasks)
        assert len(errors) == 2
        assert any("Missing 'agent' field" in error for error in errors)
        assert any("Missing 'action' field" in error for error in errors)
    
    def test_refresh_agent_matrix(self):
        """Test refreshing the agent matrix."""
        # Get initial matrix
        initial_matrix = get_agent_matrix()
        
        # Refresh
        refresh_agent_matrix()
        
        # Get matrix again (should be the same)
        refreshed_matrix = get_agent_matrix()
        
        assert initial_matrix == refreshed_matrix

class TestAgentRoutingConfig:
    """Test agent routing configuration."""
    
    def test_config_structure(self):
        """Test that agent routing config is properly structured."""
        from mcp_server.config import get_config
        
        config = get_config()
        
        # Check that agent_routing config exists
        assert hasattr(config.master_orchestrator, 'agent_routing')
        
        # Check routing mode
        assert config.master_orchestrator.agent_routing.mode in ["header", "topic"]
        
        # Check topic configuration
        assert config.master_orchestrator.agent_routing.default_topic == "task.requests"
        assert config.master_orchestrator.agent_routing.topic_prefix == "task.requests."

class TestAgentAPI:
    """Test agent API endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_agent_matrix_endpoint(self):
        """Test GET /agents/ endpoint."""
        from mcp_server.api.agent_router import create_agent_router
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        
        app = FastAPI()
        router = create_agent_router()
        app.include_router(router)
        
        client = TestClient(app)
        response = client.get("/agents/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "agents" in data
        assert "total_agents" in data
        assert "routing_mode" in data
        assert data["total_agents"] == 4  # eda, fe, model, custom
    
    @pytest.mark.asyncio
    async def test_get_agent_info_endpoint(self):
        """Test GET /agents/{agent_name} endpoint."""
        from mcp_server.api.agent_router import create_agent_router
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        
        app = FastAPI()
        router = create_agent_router()
        app.include_router(router)
        
        client = TestClient(app)
        
        # Valid agent
        response = client.get("/agents/eda")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "eda"
        assert "actions" in data
        assert "action_count" in data
        assert "status" in data
        
        # Invalid agent
        response = client.get("/agents/unknown")
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_get_agent_actions_endpoint(self):
        """Test GET /agents/{agent_name}/actions endpoint."""
        from mcp_server.api.agent_router import create_agent_router
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        
        app = FastAPI()
        router = create_agent_router()
        app.include_router(router)
        
        client = TestClient(app)
        
        # Valid agent
        response = client.get("/agents/eda/actions")
        assert response.status_code == 200
        actions = response.json()
        assert isinstance(actions, list)
        assert "analyze" in actions
        
        # Invalid agent
        response = client.get("/agents/unknown/actions")
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_validate_agent_action_endpoint(self):
        """Test POST /agents/validate endpoint."""
        from mcp_server.api.agent_router import create_agent_router
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        
        app = FastAPI()
        router = create_agent_router()
        app.include_router(router)
        
        client = TestClient(app)
        
        # Valid combination
        response = client.post("/agents/validate", json={
            "agent": "eda",
            "action": "analyze"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] == True
        assert data["agent_valid"] == True
        assert data["action_valid"] == True
        
        # Invalid combination
        response = client.post("/agents/validate", json={
            "agent": "eda",
            "action": "train"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] == False
        assert data["agent_valid"] == True
        assert data["action_valid"] == False
    
    @pytest.mark.asyncio
    async def test_get_agent_names_endpoint(self):
        """Test GET /agents/names endpoint."""
        from mcp_server.api.agent_router import create_agent_router
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        
        app = FastAPI()
        router = create_agent_router()
        app.include_router(router)
        
        client = TestClient(app)
        response = client.get("/agents/names")
        
        assert response.status_code == 200
        names = response.json()
        assert isinstance(names, list)
        assert "eda" in names
        assert "fe" in names
        assert "model" in names
        assert "custom" in names

class TestIntegration:
    """Integration tests for agent routing."""
    
    def test_dsl_repair_with_agent_validation(self):
        """Test that DSL repair uses agent registry for validation."""
        from mcp_server.orchestrator.dsl_repair_pipeline import _validate_agent_action
        
        # This should now use the agent registry
        assert _validate_agent_action("eda", "analyze", None) == True
        assert _validate_agent_action("eda", "train", None) == False
    
    def test_workflow_validation_integration(self):
        """Test workflow validation integration."""
        # Test with valid workflow
        valid_tasks = [
            {"agent": "eda", "action": "analyze"},
            {"agent": "fe", "action": "create_visualization"}
        ]
        errors = validate_workflow_tasks(valid_tasks)
        assert len(errors) == 0
        
        # Test with invalid workflow
        invalid_tasks = [
            {"agent": "eda", "action": "train"}  # Invalid action for EDA
        ]
        errors = validate_workflow_tasks(invalid_tasks)
        assert len(errors) == 1
        assert "Invalid action" in errors[0]

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 