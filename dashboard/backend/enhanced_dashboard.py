"""
Enhanced Dashboard Backend for Deepline MLOps Platform

Provides comprehensive agent monitoring, metrics visualization, and control capabilities.
Implements real-time health checks, workflow monitoring, and agent management.
"""

import asyncio
import httpx
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import motor.motor_asyncio
from bson import ObjectId

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Agent configuration
AGENT_CONFIG = {
    "eda_agent": {
        "url": "http://localhost:8001",
        "health_endpoint": "/health",
        "metrics_endpoint": None,
        "poll_interval": 30000,  # 30 seconds
        "actions": [
            "load_data", "basic_info", "statistical_summary", 
            "missing_data_analysis", "create_visualization", 
            "infer_schema", "detect_outliers"
        ],
        "description": "Exploratory Data Analysis Agent",
        "color": "#48bb78"
    },
    "refinery_agent": {
        "url": "http://localhost:8005",
        "health_endpoint": "/health",
        "metrics_endpoint": "/metrics",
        "poll_interval": 30000,
        "actions": [
            "execute", "check_schema_consistency", "check_missing_values",
            "check_distributions", "check_duplicates", "check_leakage",
            "check_drift", "comprehensive_quality_report"
        ],
        "description": "Data Quality & Feature Engineering Agent",
        "color": "#4299e1"
    },
    "ml_agent": {
        "url": "http://localhost:8002",
        "health_endpoint": "/health",
        "metrics_endpoint": "/metrics",
        "poll_interval": 30000,
        "actions": [
            "class_imbalance", "train_validation_test", 
            "baseline_sanity", "experiment_tracking"
        ],
        "description": "Machine Learning & Model Training Agent",
        "color": "#9f7aea"
    },
    "master_orchestrator": {
        "url": "http://localhost:8000",
        "health_endpoint": "/health",
        "metrics_endpoint": None,
        "poll_interval": 30000,
        "actions": ["workflows/start", "datasets/upload", "runs/status"],
        "description": "Workflow Orchestration & Management",
        "color": "#667eea"
    }
}

# Pydantic models
class AgentHealth(BaseModel):
    status: str
    response_time: float
    last_check: str
    url: str
    error: Optional[str] = None

class AgentMetrics(BaseModel):
    agent_name: str
    metrics: Dict[str, Any]
    timestamp: str

class WorkflowRequest(BaseModel):
    workflow_type: str
    agents: List[str]
    parameters: Dict[str, Any]

class AgentActionRequest(BaseModel):
    action: str
    parameters: Dict[str, Any]

# Initialize FastAPI app
app = FastAPI(title="Enhanced Deepline Dashboard", version="2.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
mongo_client = motor.motor_asyncio.AsyncIOMotorClient("mongodb://localhost:27017")
db = mongo_client.deepline

# WebSocket connections
active_connections: List[WebSocket] = []

# Agent health monitoring
agent_health_status: Dict[str, AgentHealth] = {}
health_monitoring_task: Optional[asyncio.Task] = None

class AgentMonitor:
    """Real-time agent health monitoring"""
    
    def __init__(self):
        self.agent_status = {}
        self.monitoring_active = False
    
    async def start_monitoring(self):
        """Start monitoring all agents"""
        self.monitoring_active = True
        logger.info("Starting agent health monitoring")
        
        while self.monitoring_active:
            await self.check_all_agents()
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def check_all_agents(self):
        """Check health of all agents"""
        for agent_name, config in AGENT_CONFIG.items():
            await self.check_agent_health(agent_name, config)
    
    async def check_agent_health(self, agent_name: str, config: Dict[str, Any]):
        """Check health of a specific agent"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                start_time = datetime.now()
                response = await client.get(f"{config['url']}{config['health_endpoint']}")
                response_time = (datetime.now() - start_time).total_seconds()
                
                health_status = AgentHealth(
                    status="healthy" if response.status_code == 200 else "unhealthy",
                    response_time=response_time,
                    last_check=datetime.utcnow().isoformat(),
                    url=config['url']
                )
                
                if response.status_code != 200:
                    health_status.error = f"HTTP {response.status_code}"
                
                self.agent_status[agent_name] = health_status
                
                # Broadcast health update
                await broadcast_agent_health_update(agent_name, health_status)
                
        except Exception as e:
            health_status = AgentHealth(
                status="error",
                response_time=0.0,
                last_check=datetime.utcnow().isoformat(),
                url=config['url'],
                error=str(e)
            )
            
            self.agent_status[agent_name] = health_status
            await broadcast_agent_health_update(agent_name, health_status)
    
    def get_agent_status(self, agent_name: str) -> Optional[AgentHealth]:
        """Get current status of an agent"""
        return self.agent_status.get(agent_name)
    
    def get_all_status(self) -> Dict[str, AgentHealth]:
        """Get status of all agents"""
        return self.agent_status.copy()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        logger.info("Stopped agent health monitoring")

# Initialize agent monitor
agent_monitor = AgentMonitor()

async def broadcast_agent_health_update(agent_name: str, health_status: AgentHealth):
    """Broadcast agent health update to all connected clients"""
    if active_connections:
        message = {
            "type": "agent_health_update",
            "agent_name": agent_name,
            "health_status": health_status.dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        disconnected = []
        for connection in active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending to WebSocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            active_connections.remove(connection)

async def broadcast_event(event: Dict[str, Any]):
    """Broadcast event to all connected WebSocket clients"""
    if active_connections:
        message = {
            **event,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        disconnected = []
        for connection in active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending to WebSocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            active_connections.remove(connection)

# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "Enhanced Deepline Dashboard",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/agents")
async def get_agents():
    """Get all agent configurations"""
    return {
        "agents": AGENT_CONFIG,
        "total_agents": len(AGENT_CONFIG)
    }

@app.get("/agents/health")
async def get_all_agent_health():
    """Get health status of all agents"""
    return {
        "agents": agent_monitor.get_all_status(),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/agents/{agent_name}/health")
async def get_agent_health(agent_name: str):
    """Get health status of a specific agent"""
    if agent_name not in AGENT_CONFIG:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    health_status = agent_monitor.get_agent_status(agent_name)
    if not health_status:
        # Trigger immediate health check
        await agent_monitor.check_agent_health(agent_name, AGENT_CONFIG[agent_name])
        health_status = agent_monitor.get_agent_status(agent_name)
    
    return health_status

@app.get("/agents/{agent_name}/metrics")
async def get_agent_metrics(agent_name: str):
    """Get metrics for a specific agent"""
    if agent_name not in AGENT_CONFIG:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    config = AGENT_CONFIG[agent_name]
    if not config.get('metrics_endpoint'):
        raise HTTPException(status_code=404, detail="Agent has no metrics endpoint")
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{config['url']}{config['metrics_endpoint']}")
            if response.status_code == 200:
                return {
                    "agent_name": agent_name,
                    "metrics": response.json(),
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                raise HTTPException(status_code=response.status_code, detail="Failed to fetch metrics")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch metrics: {str(e)}")

@app.post("/agents/{agent_name}/actions/{action}")
async def execute_agent_action(agent_name: str, action: str, request: AgentActionRequest):
    """Execute an action on a specific agent"""
    if agent_name not in AGENT_CONFIG:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    config = AGENT_CONFIG[agent_name]
    if action not in config.get('actions', []):
        raise HTTPException(status_code=400, detail="Action not supported by agent")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{config['url']}/{action}", 
                json=request.parameters
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Broadcast action execution event
                await broadcast_event({
                    "type": "agent_action_executed",
                    "agent_name": agent_name,
                    "action": action,
                    "status": "success",
                    "result": result
                })
                
                return {
                    "agent_name": agent_name,
                    "action": action,
                    "status": "success",
                    "result": result,
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                raise HTTPException(status_code=response.status_code, detail="Action execution failed")
                
    except Exception as e:
        # Broadcast action failure event
        await broadcast_event({
            "type": "agent_action_executed",
            "agent_name": agent_name,
            "action": action,
            "status": "error",
            "error": str(e)
        })
        
        raise HTTPException(status_code=500, detail=f"Failed to execute action: {str(e)}")

@app.post("/workflows/execute")
async def execute_workflow_with_monitoring(workflow_request: WorkflowRequest):
    """Execute workflow with comprehensive monitoring"""
    
    # Validate all required agents are healthy
    agent_health = agent_monitor.get_all_status()
    unhealthy_agents = []
    
    for agent_name in workflow_request.agents:
        if agent_name not in AGENT_CONFIG:
            raise HTTPException(status_code=400, detail=f"Unknown agent: {agent_name}")
        
        health_status = agent_health.get(agent_name)
        if not health_status or health_status.status != "healthy":
            unhealthy_agents.append(agent_name)
    
    if unhealthy_agents:
        raise HTTPException(
            status_code=503, 
            detail=f"Unhealthy agents: {unhealthy_agents}"
        )
    
    # Create workflow execution record
    workflow_id = f"workflow_{int(datetime.utcnow().timestamp())}"
    workflow_doc = {
        "workflow_id": workflow_id,
        "workflow_type": workflow_request.workflow_type,
        "agents": workflow_request.agents,
        "parameters": workflow_request.parameters,
        "status": "started",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    await db.workflows.insert_one(workflow_doc)
    
    # Start workflow monitoring
    monitoring_task = asyncio.create_task(
        monitor_workflow_execution(workflow_id)
    )
    
    # Broadcast workflow start event
    await broadcast_event({
        "type": "workflow_started",
        "workflow_id": workflow_id,
        "workflow_type": workflow_request.workflow_type,
        "agents": workflow_request.agents
    })
    
    return {
        "workflow_id": workflow_id,
        "status": "started",
        "monitoring": "enabled",
        "timestamp": datetime.utcnow().isoformat()
    }

async def monitor_workflow_execution(workflow_id: str):
    """Monitor and orchestrate workflow execution step by step"""
    logger.info(f"Starting workflow execution monitor for {workflow_id}")
    
    while True:
        try:
            # Get current workflow record
            record = await db.workflows.find_one({"workflow_id": workflow_id})
            if not record:
                logger.error(f"Workflow {workflow_id} not found in database")
                break
                
            if record["status"] in ["completed", "failed", "cancelled"]:
                await broadcast_event({
                    "type": "workflow_status_updated",
                    "workflow_id": workflow_id,
                    "status": record["status"],
                    "timestamp": datetime.utcnow().isoformat()
                })
                break
            
            # Execute each step in order
            for step in record["agents"]:
                if not record.get("step_status", {}).get(step, False):
                    # Dispatch to agent
                    await broadcast_event({
                        "type": "step.update",
                        "workflow_id": workflow_id,
                        "step": step,
                        "status": "in_progress",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                    try:
                        # Get agent URL and endpoint
                        agent_config = AGENT_CONFIG.get(step)
                        if not agent_config:
                            raise Exception(f"Agent {step} not found in configuration")
                        
                        # Determine step-specific endpoint based on workflow type
                        endpoint = get_step_endpoint(step, record["workflow_type"])
                        
                        # Prepare parameters for this step
                        step_params = prepare_step_parameters(step, record["parameters"])
                        
                        # Execute step
                        async with httpx.AsyncClient(timeout=30.0) as client:
                            response = await client.post(
                                f"{agent_config['url']}/{endpoint}",
                                json=step_params
                            )
                        
                        if response.status_code == 200:
                            # Update database
                            await db.workflows.update_one(
                                {"workflow_id": workflow_id},
                                {"$set": {f"step_status.{step}": "completed"}}
                            )
                            
                            await broadcast_event({
                                "type": "step.update",
                                "workflow_id": workflow_id,
                                "step": step,
                                "status": "completed",
                                "details": response.json(),
                                "timestamp": datetime.utcnow().isoformat()
                            })
                        else:
                            raise Exception(f"Step {step} failed: {response.text}")
                            
                    except Exception as e:
                        logger.error(f"Step {step} failed: {str(e)}")
                        
                        # Update database
                        await db.workflows.update_one(
                            {"workflow_id": workflow_id},
                            {"$set": {"status": "failed", f"step_status.{step}": "failed"}}
                        )
                        
                        await broadcast_event({
                            "type": "step.update",
                            "workflow_id": workflow_id,
                            "step": step,
                            "status": "failed",
                            "error": str(e),
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        return
            
            # Check if all steps are completed
            all_completed = all(
                record.get("step_status", {}).get(step, False) 
                for step in record["agents"]
            )
            
            if all_completed:
                # Update database
                await db.workflows.update_one(
                    {"workflow_id": workflow_id},
                    {"$set": {"status": "completed"}}
                )
                
                await broadcast_event({
                    "type": "workflow_status_updated",
                    "workflow_id": workflow_id,
                    "status": "completed",
                    "timestamp": datetime.utcnow().isoformat()
                })
                break
            
            # Wait before next iteration
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"Error in workflow execution monitor: {str(e)}")
            await broadcast_event({
                "type": "workflow_status_updated",
                "workflow_id": workflow_id,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            break

def get_step_endpoint(step: str, workflow_type: str) -> str:
    """Get the appropriate endpoint for a workflow step"""
    endpoints = {
        "eda_agent": {
            "data_analysis": "execute",
            "ml_pipeline": "execute",
            "full_pipeline": "execute"
        },
        "refinery_agent": {
            "data_analysis": "execute",
            "ml_pipeline": "execute", 
            "full_pipeline": "execute"
        },
        "ml_agent": {
            "data_analysis": "health",  # Skip for data analysis
            "ml_pipeline": "execute",
            "full_pipeline": "execute"
        }
    }
    
    return endpoints.get(step, {}).get(workflow_type, "execute")

def prepare_step_parameters(step: str, workflow_params: dict) -> dict:
    """Prepare parameters specific to each step"""
    if step == "eda_agent":
        return {
            "dataset": workflow_params.get("dataset"),
            "mission_dsl": workflow_params.get("mission_dsl"),
            "options": workflow_params.get("options", {})
        }
    elif step == "refinery_agent":
        return {
            "data_path": workflow_params.get("dataset"),
            "config": workflow_params.get("options", {})
        }
    elif step == "ml_agent":
        return {
            "data_path": workflow_params.get("dataset"),
            "model_config": workflow_params.get("options", {})
        }
    else:
        return workflow_params

@app.get("/workflows/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """Get status of a specific workflow"""
    workflow = await db.workflows.find_one({"workflow_id": workflow_id})
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Convert ObjectId to string for JSON serialization
    workflow["_id"] = str(workflow["_id"])
    
    return workflow

@app.get("/workflows")
async def get_all_workflows(limit: int = 50):
    """Get all workflows"""
    workflows = await db.workflows.find().sort([("created_at", -1)]).limit(limit).to_list(limit)
    
    # Convert ObjectIds to strings
    for workflow in workflows:
        workflow["_id"] = str(workflow["_id"])
    
    return {
        "workflows": workflows,
        "total": len(workflows),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.put("/workflows/{workflow_id}/status")
async def update_workflow_status(workflow_id: str, status: str):
    """Update workflow status"""
    valid_statuses = ["running", "completed", "failed", "cancelled"]
    if status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")
    
    result = await db.workflows.update_one(
        {"workflow_id": workflow_id},
        {"$set": {"status": status, "updated_at": datetime.utcnow()}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Broadcast status update
    await broadcast_event({
        "type": "workflow_status_updated",
        "workflow_id": workflow_id,
        "status": status,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    return {"workflow_id": workflow_id, "status": status}

@app.websocket("/ws/dashboard")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates"""
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"Dashboard client connected. Total connections: {len(active_connections)}")
    
    try:
        # Send initial agent health status
        initial_status = agent_monitor.get_all_status()
        await websocket.send_text(json.dumps({
            "type": "initial_status",
            "agents": {name: health.dict() for name, health in initial_status.items()},
            "timestamp": datetime.utcnow().isoformat()
        }))
        
        # Keep connection alive
        while True:
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        logger.info("Dashboard client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        active_connections.remove(websocket)
        logger.info(f"Dashboard client disconnected. Total connections: {len(active_connections)}")

@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    logger.info("Starting Enhanced Deepline Dashboard")
    
    # Start agent health monitoring
    global health_monitoring_task
    health_monitoring_task = asyncio.create_task(agent_monitor.start_monitoring())
    logger.info("Agent health monitoring started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Enhanced Deepline Dashboard")
    
    # Stop agent monitoring
    agent_monitor.stop_monitoring()
    if health_monitoring_task:
        health_monitoring_task.cancel()
    
    # Close WebSocket connections
    for connection in active_connections:
        try:
            await connection.close()
        except Exception:
            pass
    
    logger.info("Enhanced Deepline Dashboard shutdown complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 