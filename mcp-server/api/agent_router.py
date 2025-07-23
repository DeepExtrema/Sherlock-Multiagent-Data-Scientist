"""
Agent API Router

Provides REST endpoints for agent capabilities and status.
Exposes the agent registry for dashboard and client consumption.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ..orchestrator.agent_registry import (
    get_agent_matrix, get_agent_names, get_agent_actions,
    get_agent_stats, is_valid_agent, is_valid_action
)

logger = logging.getLogger(__name__)

# Response Models
class AgentInfo(BaseModel):
    """Information about a single agent."""
    name: str = Field(..., description="Agent name")
    actions: List[str] = Field(..., description="List of valid actions")
    action_count: int = Field(..., description="Number of valid actions")
    status: str = Field(..., description="Agent status")

class AgentMatrixResponse(BaseModel):
    """Response model for agent matrix."""
    agents: Dict[str, AgentInfo] = Field(..., description="Agent information")
    total_agents: int = Field(..., description="Total number of agents")
    routing_mode: str = Field(..., description="Current routing mode")

class AgentValidationRequest(BaseModel):
    """Request model for agent validation."""
    agent: str = Field(..., description="Agent name to validate")
    action: str = Field(..., description="Action name to validate")

class AgentValidationResponse(BaseModel):
    """Response model for agent validation."""
    valid: bool = Field(..., description="Whether the combination is valid")
    agent_valid: bool = Field(..., description="Whether the agent is valid")
    action_valid: bool = Field(..., description="Whether the action is valid for the agent")
    valid_actions: List[str] = Field(..., description="Valid actions for the agent (if agent is valid)")

def create_agent_router() -> APIRouter:
    """Create and configure the agent API router."""
    router = APIRouter(prefix="/agents", tags=["agents"])

    @router.get("/", response_model=AgentMatrixResponse)
    async def get_agent_matrix():
        """
        Get the complete agent matrix with capabilities.
        
        Returns:
            Complete agent matrix with status information
        """
        try:
            matrix = get_agent_matrix()
            stats = get_agent_stats()
            
            # Convert to response format
            agents = {}
            for agent_name, agent_stats in stats.items():
                agents[agent_name] = AgentInfo(
                    name=agent_name,
                    actions=agent_stats["actions"],
                    action_count=agent_stats["action_count"],
                    status=agent_stats["status"]
                )
            
            return AgentMatrixResponse(
                agents=agents,
                total_agents=len(agents),
                routing_mode="header"  # TODO: Get from config
            )
            
        except Exception as e:
            logger.error(f"Failed to get agent matrix: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve agent matrix"
            )

    @router.get("/{agent_name}", response_model=AgentInfo)
    async def get_agent_info(agent_name: str):
        """
        Get information about a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Agent information
            
        Raises:
            404: Agent not found
        """
        try:
            if not is_valid_agent(agent_name):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Agent '{agent_name}' not found"
                )
            
            actions = get_agent_actions(agent_name)
            stats = get_agent_stats()
            
            return AgentInfo(
                name=agent_name,
                actions=actions,
                action_count=len(actions),
                status=stats.get(agent_name, {}).get("status", "unknown")
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get agent info for {agent_name}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve agent information"
            )

    @router.get("/{agent_name}/actions", response_model=List[str])
    async def get_agent_actions_list(agent_name: str):
        """
        Get the list of valid actions for a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            List of valid actions
            
        Raises:
            404: Agent not found
        """
        try:
            if not is_valid_agent(agent_name):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Agent '{agent_name}' not found"
                )
            
            actions = get_agent_actions(agent_name)
            return actions
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get actions for {agent_name}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve agent actions"
            )

    @router.post("/validate", response_model=AgentValidationResponse)
    async def validate_agent_action(request: AgentValidationRequest):
        """
        Validate an agent-action combination.
        
        Args:
            request: Validation request with agent and action
            
        Returns:
            Validation result with details
        """
        try:
            agent_valid = is_valid_agent(request.agent)
            action_valid = is_valid_action(request.agent, request.action)
            valid = agent_valid and action_valid
            
            valid_actions = []
            if agent_valid:
                valid_actions = get_agent_actions(request.agent)
            
            return AgentValidationResponse(
                valid=valid,
                agent_valid=agent_valid,
                action_valid=action_valid,
                valid_actions=valid_actions
            )
            
        except Exception as e:
            logger.error(f"Failed to validate agent action: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to validate agent action"
            )

    @router.get("/names", response_model=List[str])
    async def get_agent_names_list():
        """
        Get the list of all valid agent names.
        
        Returns:
            List of valid agent names
        """
        try:
            return list(get_agent_names())
        except Exception as e:
            logger.error(f"Failed to get agent names: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve agent names"
            )

    return router 