"""
Unit tests for DSL Repair Pipeline

Tests the automatic repair and validation of workflow DSL using Guardrails-AI
and LLM-based repair capabilities.
"""

import pytest
import yaml
import asyncio
import sys
import os
from unittest.mock import AsyncMock, patch, MagicMock

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from mcp_server.orchestrator.dsl_repair_pipeline import (
    repair_dsl, repair, _quick_fixes, _detect_circular_dependencies,
    _validate_agent_action, _llm_repair_step
)
from mcp_server.orchestrator.translator import NeedsHumanError

# Test cases for invalid DSL
INVALID_CASES = {
    "missing_workflow": """
tasks:
- name: process_data
  agent: eda
  action: analyze
""",
    
    "bad_indent_and_misspelled": """
workflow:
  name: data_workflow
tasks:
- name: process_data
  agent: eda
  action: analyze
  param: {}
  dependson: []
""",
    
    "circular_dependencies": """
workflow:
  name: circular_workflow
tasks:
- name: task_a
  agent: eda
  action: analyze
  depends_on: ["task_b"]
- name: task_b
  agent: fe
  action: create_visualization
  depends_on: ["task_a"]
""",
    
    "invalid_agent_action": """
workflow:
  name: invalid_workflow
tasks:
- name: invalid_task
  agent: eda
  action: train_model
""",
    
    "missing_required_fields": """
workflow:
  name: incomplete_workflow
tasks:
- agent: eda
  action: analyze
""",
    
    "empty_tasks": """
workflow:
  name: empty_workflow
tasks: []
""",
    
    "malformed_yaml": """
workflow:
  name: malformed_workflow
tasks:
- name: task1
  agent: eda
  action: analyze
  depends_on: [task2
- name: task2
  agent: fe
  action: create_visualization
""",
}

# Valid DSL for comparison
VALID_DSL = """
workflow:
  name: valid_workflow
  description: "A valid workflow"
  priority: 5
  sla_minutes: 60
tasks:
- name: process_data
  agent: eda
  action: analyze
  params:
    input_file: "data.csv"
  depends_on: []
- name: generate_report
  agent: fe
  action: create_visualization
  params:
    chart_type: "bar"
  depends_on: ["process_data"]
"""

@pytest.fixture
def mock_db():
    """Mock MongoDB database."""
    db = AsyncMock()
    db.dsl_repair_logs = AsyncMock()
    db.dsl_repair_logs.insert_one = AsyncMock()
    return db

@pytest.fixture
def mock_config():
    """Mock configuration."""
    config = MagicMock()
    config.master_orchestrator.dsl_repair.max_repair_attempts = 3
    config.master_orchestrator.dsl_repair.enable_auto_repair = True
    config.master_orchestrator.dsl_repair.log_repair_attempts = True
    config.master_orchestrator.dsl_repair.timeout_seconds = 30
    config.master_orchestrator.llm.max_tokens = 800
    config.master_orchestrator.orchestrator.deadlock.alert_webhook = ""
    
    # Agent-action matrix
    config.master_orchestrator.agent_actions = {
        "eda": ["analyze", "clean", "transform", "explore", "preprocess"],
        "fe": ["create_visualization", "build_dashboard", "generate_report", "create_chart", "export_data"],
        "model": ["train", "predict", "evaluate", "tune", "deploy"],
        "custom": ["execute", "process", "run_script", "call_api"]
    }
    
    return config

class TestQuickFixes:
    """Test the quick fixes functionality."""
    
    def test_rename_misspelled_keys(self):
        """Test renaming common misspellings."""
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
        parsed = yaml.safe_load(result)
        
        assert "params" in parsed["tasks"][0]
        assert "depends_on" in parsed["tasks"][0]
        assert "param" not in parsed["tasks"][0]
        assert "dependson" not in parsed["tasks"][0]
    
    def test_fill_defaults(self):
        """Test filling default values."""
        yaml_str = """
workflow:
  name: test_workflow
tasks:
- name: task1
  agent: eda
  action: analyze
"""
        result = _quick_fixes(yaml_str)
        parsed = yaml.safe_load(result)
        
        assert parsed["tasks"][0]["params"] == {}
        assert parsed["tasks"][0]["depends_on"] == []
        assert parsed["workflow"]["priority"] == 5
        assert parsed["workflow"]["sla_minutes"] == 60
    
    def test_add_missing_workflow(self):
        """Test adding workflow section if missing."""
        yaml_str = """
tasks:
- name: task1
  agent: eda
  action: analyze
"""
        result = _quick_fixes(yaml_str)
        parsed = yaml.safe_load(result)
        
        assert "workflow" in parsed
        assert parsed["workflow"]["name"] == "unnamed_workflow"
    
    def test_fix_tabs_to_spaces(self):
        """Test converting tabs to spaces."""
        yaml_str = """
workflow:
	name: test_workflow
tasks:
	- name: task1
		agent: eda
		action: analyze
"""
        result = _quick_fixes(yaml_str)
        # Should not raise YAMLError
        parsed = yaml.safe_load(result)
        assert parsed is not None

class TestCircularDependencyDetection:
    """Test circular dependency detection."""
    
    def test_no_circular_dependencies(self):
        """Test valid dependency chain."""
        tasks = [
            {"name": "task1", "depends_on": []},
            {"name": "task2", "depends_on": ["task1"]},
            {"name": "task3", "depends_on": ["task2"]}
        ]
        cycles = _detect_circular_dependencies(tasks)
        assert cycles == []
    
    def test_simple_circular_dependency(self):
        """Test simple circular dependency."""
        tasks = [
            {"name": "task1", "depends_on": ["task2"]},
            {"name": "task2", "depends_on": ["task1"]}
        ]
        cycles = _detect_circular_dependencies(tasks)
        assert len(cycles) > 0
        assert "task1" in cycles[0]
        assert "task2" in cycles[0]
    
    def test_complex_circular_dependency(self):
        """Test complex circular dependency."""
        tasks = [
            {"name": "task1", "depends_on": ["task2"]},
            {"name": "task2", "depends_on": ["task3"]},
            {"name": "task3", "depends_on": ["task1"]}
        ]
        cycles = _detect_circular_dependencies(tasks)
        assert len(cycles) > 0
    
    def test_self_dependency(self):
        """Test task depending on itself."""
        tasks = [
            {"name": "task1", "depends_on": ["task1"]}
        ]
        cycles = _detect_circular_dependencies(tasks)
        assert len(cycles) > 0

class TestAgentActionValidation:
    """Test agent-action validation."""
    
    def test_valid_agent_action_combinations(self, mock_config):
        """Test valid agent-action combinations."""
        valid_combinations = [
            ("eda", "analyze"),
            ("fe", "create_visualization"),
            ("model", "train"),
            ("custom", "execute")
        ]
        
        for agent, action in valid_combinations:
            assert _validate_agent_action(agent, action, mock_config)
    
    def test_invalid_agent_action_combinations(self, mock_config):
        """Test invalid agent-action combinations."""
        invalid_combinations = [
            ("eda", "train"),  # EDA can't train
            ("fe", "analyze"),  # FE can't analyze
            ("model", "create_visualization"),  # Model can't create viz
            ("custom", "analyze")  # Custom can't analyze
        ]
        
        for agent, action in invalid_combinations:
            assert not _validate_agent_action(agent, action, mock_config)
    
    def test_unknown_agent(self, mock_config):
        """Test unknown agent."""
        assert not _validate_agent_action("unknown_agent", "analyze", mock_config)

class TestDSLRepair:
    """Test the main DSL repair functionality."""
    
    @pytest.mark.asyncio
    async def test_repair_missing_workflow(self, mock_db, mock_config):
        """Test repairing DSL with missing workflow section."""
        with patch('mcp_server.orchestrator.dsl_repair_pipeline.get_config', return_value=mock_config):
            result = await repair_dsl(INVALID_CASES["missing_workflow"], mock_db)
            
            assert "workflow" in result
            assert "tasks" in result
            assert result["workflow"]["name"] == "unnamed_workflow"
            assert len(result["tasks"]) == 1
    
    @pytest.mark.asyncio
    async def test_repair_bad_indent_and_misspelled(self, mock_db, mock_config):
        """Test repairing DSL with bad indentation and misspelled keys."""
        with patch('mcp_server.orchestrator.dsl_repair_pipeline.get_config', return_value=mock_config):
            result = await repair_dsl(INVALID_CASES["bad_indent_and_misspelled"], mock_db)
            
            assert "workflow" in result
            assert "tasks" in result
            task = result["tasks"][0]
            assert "params" in task
            assert "depends_on" in task
            assert "param" not in task
            assert "dependson" not in task
    
    @pytest.mark.asyncio
    async def test_repair_circular_dependencies_fails(self, mock_db, mock_config):
        """Test that circular dependencies cause repair to fail."""
        with patch('mcp_server.orchestrator.dsl_repair_pipeline.get_config', return_value=mock_config):
            with pytest.raises(NeedsHumanError) as exc_info:
                await repair_dsl(INVALID_CASES["circular_dependencies"], mock_db)
            
            assert "circular dependencies" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_repair_invalid_agent_action_fails(self, mock_db, mock_config):
        """Test that invalid agent-action combinations cause repair to fail."""
        with patch('mcp_server.orchestrator.dsl_repair_pipeline.get_config', return_value=mock_config):
            with pytest.raises(NeedsHumanError) as exc_info:
                await repair_dsl(INVALID_CASES["invalid_agent_action"], mock_db)
            
            assert "invalid action" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_repair_empty_tasks_fails(self, mock_db, mock_config):
        """Test that empty tasks cause repair to fail."""
        with patch('mcp_server.orchestrator.dsl_repair_pipeline.get_config', return_value=mock_config):
            with pytest.raises(NeedsHumanError) as exc_info:
                await repair_dsl(INVALID_CASES["empty_tasks"], mock_db)
            
            assert "empty" in str(exc_info.value).lower() or "missing" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_repair_malformed_yaml_fails(self, mock_db, mock_config):
        """Test that malformed YAML causes repair to fail."""
        with patch('mcp_server.orchestrator.dsl_repair_pipeline.get_config', return_value=mock_config):
            with pytest.raises(NeedsHumanError):
                await repair_dsl(INVALID_CASES["malformed_yaml"], mock_db)
    
    @pytest.mark.asyncio
    async def test_valid_dsl_passes_through(self, mock_db, mock_config):
        """Test that valid DSL passes through without repair."""
        with patch('mcp_server.orchestrator.dsl_repair_pipeline.get_config', return_value=mock_config):
            result = await repair_dsl(VALID_DSL, mock_db)
            
            assert "workflow" in result
            assert "tasks" in result
            assert result["workflow"]["name"] == "valid_workflow"
            assert len(result["tasks"]) == 2
    
    @pytest.mark.asyncio
    async def test_repair_with_llm_fallback(self, mock_db, mock_config):
        """Test repair with LLM fallback."""
        with patch('mcp_server.orchestrator.dsl_repair_pipeline.get_config', return_value=mock_config):
            with patch('mcp_server.orchestrator.dsl_repair_pipeline._llm_repair_step') as mock_llm:
                mock_llm.return_value = VALID_DSL
                
                result = await repair_dsl(INVALID_CASES["missing_workflow"], mock_db)
                
                assert mock_llm.called
                assert "workflow" in result
    
    @pytest.mark.asyncio
    async def test_repair_timeout(self, mock_db, mock_config):
        """Test repair timeout."""
        mock_config.master_orchestrator.dsl_repair.timeout_seconds = 0.1
        
        with patch('mcp_server.orchestrator.dsl_repair_pipeline.get_config', return_value=mock_config):
            with patch('mcp_server.orchestrator.dsl_repair_pipeline._llm_repair_step') as mock_llm:
                mock_llm.side_effect = asyncio.sleep(1)  # Simulate slow LLM
                
                with pytest.raises(asyncio.TimeoutError):
                    await repair(INVALID_CASES["missing_workflow"], mock_db)

class TestLLMRepairStep:
    """Test LLM repair step functionality."""
    
    @pytest.mark.asyncio
    async def test_llm_repair_step(self, mock_config):
        """Test LLM repair step."""
        with patch('mcp_server.orchestrator.dsl_repair_pipeline.call_llm') as mock_call_llm:
            mock_call_llm.return_value = VALID_DSL
            
            result = await _llm_repair_step("invalid yaml", ValueError("test error"), mock_config)
            
            assert mock_call_llm.called
            assert result == VALID_DSL
    
    @pytest.mark.asyncio
    async def test_llm_repair_step_with_markdown(self, mock_config):
        """Test LLM repair step with markdown code blocks."""
        with patch('mcp_server.orchestrator.dsl_repair_pipeline.call_llm') as mock_call_llm:
            mock_call_llm.return_value = f"```yaml\n{VALID_DSL}\n```"
            
            result = await _llm_repair_step("invalid yaml", ValueError("test error"), mock_config)
            
            assert result == VALID_DSL
    
    @pytest.mark.asyncio
    async def test_llm_repair_step_failure(self, mock_config):
        """Test LLM repair step failure."""
        with patch('mcp_server.orchestrator.dsl_repair_pipeline.call_llm') as mock_call_llm:
            mock_call_llm.side_effect = Exception("LLM error")
            
            with pytest.raises(Exception):
                await _llm_repair_step("invalid yaml", ValueError("test error"), mock_config)

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 