"""
Unit tests for LLM Client

Tests the thin wrapper around Ollama and OpenAI APIs for LLM calls.
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import AsyncMock, patch, MagicMock

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from mcp_server.orchestrator.llm_client import (
    LlmClient, LlmRequest, LlmResponse, call_llm, get_llm_client
)

class TestLlmRequest:
    """Test LLM request model."""
    
    def test_valid_request(self):
        """Test creating a valid LLM request."""
        request = LlmRequest(
            prompt="Test prompt",
            model="llama2-13b",
            temperature=0.0,
            max_tokens=800
        )
        
        assert request.prompt == "Test prompt"
        assert request.model == "llama2-13b"
        assert request.temperature == 0.0
        assert request.max_tokens == 800
    
    def test_default_values(self):
        """Test default values for LLM request."""
        request = LlmRequest(prompt="Test prompt")
        
        assert request.model == "llama2-13b"
        assert request.temperature == 0.0
        assert request.max_tokens == 800
    
    def test_invalid_temperature(self):
        """Test invalid temperature values."""
        with pytest.raises(ValueError):
            LlmRequest(prompt="Test", temperature=-1.0)
        
        with pytest.raises(ValueError):
            LlmRequest(prompt="Test", temperature=3.0)
    
    def test_invalid_max_tokens(self):
        """Test invalid max_tokens values."""
        with pytest.raises(ValueError):
            LlmRequest(prompt="Test", max_tokens=0)
        
        with pytest.raises(ValueError):
            LlmRequest(prompt="Test", max_tokens=5000)

class TestLlmResponse:
    """Test LLM response model."""
    
    def test_valid_response(self):
        """Test creating a valid LLM response."""
        response = LlmResponse(
            text="Generated text",
            model="llama2-13b",
            provider="ollama"
        )
        
        assert response.text == "Generated text"
        assert response.model == "llama2-13b"
        assert response.provider == "ollama"

class TestLlmClient:
    """Test LLM client functionality."""
    
    @pytest.fixture
    def client(self):
        """Create LLM client instance."""
        return LlmClient(
            endpoint="http://localhost:11434/api/generate",
            fallback_provider="openai"
        )
    
    @pytest.mark.asyncio
    async def test_call_ollama_success(self, client):
        """Test successful Ollama API call."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Generated text"}
        mock_response.raise_for_status.return_value = None
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            
            result = await client.call_llm("Test prompt")
            
            assert result.text == "Generated text"
            assert result.model == "llama2-13b"
            assert result.provider == "ollama"
    
    @pytest.mark.asyncio
    async def test_call_ollama_failure_with_openai_fallback(self, client):
        """Test Ollama failure with OpenAI fallback."""
        # Mock Ollama failure
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.side_effect = Exception("Ollama error")
            
            # Mock OpenAI success
            with patch('mcp_server.orchestrator.llm_client.OPENAI_AVAILABLE', True):
                with patch('mcp_server.orchestrator.llm_client.openai') as mock_openai:
                    with patch('os.getenv', return_value="test-api-key"):
                        with patch('asyncio.to_thread') as mock_to_thread:
                            mock_to_thread.return_value = MagicMock(
                                choices=[MagicMock(message=MagicMock(content="OpenAI response"))]
                            )
                            
                            result = await client.call_llm("Test prompt")
                            
                            assert result.text == "OpenAI response"
                            assert result.provider == "openai"
    
    @pytest.mark.asyncio
    async def test_call_ollama_failure_no_fallback(self, client):
        """Test Ollama failure without fallback."""
        client.fallback_provider = "nonexistent"
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.side_effect = Exception("Ollama error")
            
            with pytest.raises(Exception) as exc_info:
                await client.call_llm("Test prompt")
            
            assert "Ollama error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_call_openai_no_api_key(self, client):
        """Test OpenAI call without API key."""
        with patch('mcp_server.orchestrator.llm_client.OPENAI_AVAILABLE', True):
            with patch('os.getenv', return_value=None):
                with pytest.raises(ValueError) as exc_info:
                    await client._call_openai(MagicMock())
                
                assert "OPENAI_API_KEY" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_call_openai_model_mapping(self, client):
        """Test OpenAI model mapping."""
        with patch('mcp_server.orchestrator.llm_client.OPENAI_AVAILABLE', True):
            with patch('os.getenv', return_value="test-api-key"):
                with patch('asyncio.to_thread') as mock_to_thread:
                    mock_to_thread.return_value = MagicMock(
                        choices=[MagicMock(message=MagicMock(content="Response"))]
                    )
                    
                    # Test different model mappings
                    request = LlmRequest(model="llama2-7b")
                    result = await client._call_openai(request)
                    assert result.model == "gpt-4o-mini"
                    
                    request = LlmRequest(model="unknown-model")
                    result = await client._call_openai(request)
                    assert result.model == "gpt-4o-mini"
    
    @pytest.mark.asyncio
    async def test_call_openai_openai_not_available(self, client):
        """Test OpenAI call when library not available."""
        with patch('mcp_server.orchestrator.llm_client.OPENAI_AVAILABLE', False):
            with pytest.raises(ImportError):
                await client._call_openai(MagicMock())

class TestGlobalClient:
    """Test global client functionality."""
    
    @pytest.mark.asyncio
    async def test_get_llm_client(self):
        """Test getting global LLM client."""
        with patch('mcp_server.orchestrator.llm_client.get_config') as mock_get_config:
            mock_config = MagicMock()
            mock_config.master_orchestrator.llm.endpoint = "http://test:11434/api/generate"
            mock_config.master_orchestrator.llm.fallback_provider = "openai"
            mock_get_config.return_value = mock_config
            
            client = get_llm_client()
            
            assert isinstance(client, LlmClient)
            assert client.endpoint == "http://test:11434/api/generate"
            assert client.fallback_provider == "openai"
    
    @pytest.mark.asyncio
    async def test_call_llm_convenience(self):
        """Test convenience call_llm function."""
        with patch('mcp_server.orchestrator.llm_client.get_llm_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.call_llm.return_value = LlmResponse(
                text="Test response",
                model="llama2-13b",
                provider="ollama"
            )
            mock_get_client.return_value = mock_client
            
            result = await call_llm("Test prompt", temperature=0.5)
            
            assert result == "Test response"
            mock_client.call_llm.assert_called_once_with(
                "Test prompt",
                temperature=0.5
            )

class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_http_error_handling(self):
        """Test HTTP error handling."""
        client = LlmClient("http://localhost:11434/api/generate")
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            # Test HTTP error
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = Exception("HTTP 500")
            mock_client.post.return_value = mock_response
            
            with pytest.raises(Exception):
                await client.call_llm("Test prompt")
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling."""
        client = LlmClient("http://localhost:11434/api/generate")
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.side_effect = asyncio.TimeoutError("Request timeout")
            
            with pytest.raises(Exception):
                await client.call_llm("Test prompt")

class TestIntegration:
    """Integration tests."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test full LLM call workflow."""
        with patch('mcp_server.orchestrator.llm_client.get_config') as mock_get_config:
            mock_config = MagicMock()
            mock_config.master_orchestrator.llm.endpoint = "http://localhost:11434/api/generate"
            mock_config.master_orchestrator.llm.fallback_provider = "openai"
            mock_get_config.return_value = mock_config
            
            with patch('httpx.AsyncClient') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value.__aenter__.return_value = mock_client
                
                mock_response = MagicMock()
                mock_response.json.return_value = {"response": "Integration test response"}
                mock_response.raise_for_status.return_value = None
                mock_client.post.return_value = mock_response
                
                result = await call_llm("Integration test prompt")
                
                assert result == "Integration test response"

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 