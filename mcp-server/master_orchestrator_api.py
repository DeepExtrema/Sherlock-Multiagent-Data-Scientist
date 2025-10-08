#!/usr/bin/env python3
"""
Master Orchestrator API

Coordinates workflows between different agents (EDA Agent, etc.)
Provides REST API for workflow management, dataset upload, and artifact retrieval.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import asyncio
import json
import uuid
import time
import os
import shutil
from pathlib import Path
import requests
from datetime import datetime
import logging
import sys
from pathlib import Path
# Define root location for artifacts
ARTIFACTS_DIR = Path("artifacts")  # You may change as appropriate
# Ensure local packages are importable
CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Routers
from api.data_router import create_data_router
from api.agent_router import create_agent_router


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Master Orchestrator API",
    description="Orchestrates workflows between different AI agents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount feature routers
app.include_router(create_data_router())
app.include_router(create_agent_router())

# ─── DATA MODELS ──────────────────────────────────────────────────────────────

class Task(BaseModel):
    agent: str
    action: str
    args: Dict[str, Any]

class WorkflowRequest(BaseModel):
    run_name: str
    tasks: List[Task]
    priority: Optional[int] = 1

class WorkflowResponse(BaseModel):
    run_id: str
    status: str
    message: str

class RunStatus(BaseModel):
    run_id: str
    status: str
    progress: float
    current_task: Optional[str]
    start_time: str
    end_time: Optional[str]
    error_message: Optional[str]

class Artifact(BaseModel):
    artifact_id: str
    type: str
    filename: str
    size: int
    created_at: str
    download_url: Optional[str]

# ─── GLOBAL STATE ─────────────────────────────────────────────────────────────

# In-memory storage (replace with database in production)
workflows = {}
datasets = {}
artifacts = {}
run_status = {}

# Configuration
# Prefer environment variable inside containers; default to localhost for local dev
EDA_AGENT_URL = os.getenv("EDA_AGENT_URL", "http://localhost:8001")
UPLOAD_DIR = "uploads"
ARTIFACT_DIR = "artifacts"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def get_agent_url(agent_name: str) -> str:
    """Get the URL for a specific agent."""
    agent_urls = {
        "eda_agent": EDA_AGENT_URL,
        # Add more agents here
    }
    return agent_urls.get(agent_name, "")

async def execute_task(task: Task, run_id: str) -> Dict[str, Any]:
    """Execute a single task by calling the appropriate agent."""
    agent_url = get_agent_url(task.agent)
    if not agent_url:
        raise HTTPException(status_code=400, detail=f"Unknown agent: {task.agent}")
    
    try:
        logger.info(f"Executing task: {task.action} on {task.agent}")
        
        # Call the agent's endpoint
        response = requests.post(
            f"{agent_url}/{task.action}",
            json=task.args,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"Task completed successfully: {task.action}")
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Task failed: {task.action} - {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent task failed: {str(e)}")

async def run_workflow(run_id: str, workflow_request: WorkflowRequest):
    """Execute a workflow asynchronously."""
    try:
        logger.info(f"Starting workflow: {run_id}")
        
        # Update status
        run_status[run_id] = {
            "run_id": run_id,
            "status": "RUNNING",
            "progress": 0.0,
            "current_task": None,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "error_message": None
        }
        
        total_tasks = len(workflow_request.tasks)
        
        for i, task in enumerate(workflow_request.tasks):
            try:
                # Update current task
                run_status[run_id]["current_task"] = f"{task.agent}:{task.action}"
                run_status[run_id]["progress"] = (i / total_tasks) * 100
                
                # Execute task
                result = await execute_task(task, run_id)
                
                # Store result
                if run_id not in artifacts:
                    artifacts[run_id] = []
                
                # Check if result contains visualization files
                if "visualization_files" in result:
                    for viz_file in result["visualization_files"]:
                        artifact = {
                            "artifact_id": str(uuid.uuid4()),
                            "type": "visualization",
                            "filename": viz_file,
                            "size": os.path.getsize(viz_file) if os.path.exists(viz_file) else 0,
                            "created_at": datetime.now().isoformat(),
                            "download_url": f"/artifacts/{run_id}/{viz_file}"
                        }
                        artifacts[run_id].append(artifact)
                
                logger.info(f"Task {i+1}/{total_tasks} completed: {task.action}")
                
            except Exception as e:
                logger.error(f"Task {task.action} failed: {str(e)}")
                run_status[run_id]["status"] = "FAILED"
                run_status[run_id]["error_message"] = str(e)
                run_status[run_id]["end_time"] = datetime.now().isoformat()
                return
        
        # Mark as completed
        run_status[run_id]["status"] = "COMPLETED"
        run_status[run_id]["progress"] = 100.0
        run_status[run_id]["end_time"] = datetime.now().isoformat()
        run_status[run_id]["current_task"] = None
        
        logger.info(f"Workflow completed: {run_id}")
        
    except Exception as e:
        logger.error(f"Workflow failed: {run_id} - {str(e)}")
        if run_id in run_status:
            run_status[run_id]["status"] = "FAILED"
            run_status[run_id]["error_message"] = str(e)
            run_status[run_id]["end_time"] = datetime.now().isoformat()

# ─── API ENDPOINTS ───────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Master Orchestrator API",
        "version": "1.0.0",
        "status": "operational",
        "agents": ["eda_agent"],
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agents": {
            "eda_agent": {
                "url": EDA_AGENT_URL,
                "status": "checking..."
            }
        }
    }

@app.post("/datasets/upload", response_model=Dict[str, str])
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...)
):
    """Upload a dataset file."""
    try:
        # Generate unique dataset ID
        dataset_id = str(uuid.uuid4())
        
        # Save file
        file_path = os.path.join(UPLOAD_DIR, f"{dataset_id}_{file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Store dataset info
        datasets[dataset_id] = {
            "id": dataset_id,
            "name": name,
            "filename": file.filename,
            "file_path": file_path,
            "size": os.path.getsize(file_path),
            "uploaded_at": datetime.now().isoformat()
        }
        
        logger.info(f"Dataset uploaded: {name} -> {dataset_id}")
        
        return {
            "dataset_id": dataset_id,
            "name": name,
            "filename": file.filename,
            "message": "Dataset uploaded successfully"
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/datasets")
async def list_datasets():
    """List all uploaded datasets."""
    return {
        "datasets": list(datasets.values()),
        "count": len(datasets)
    }

@app.post("/workflows/start", response_model=WorkflowResponse)
async def start_workflow(
    workflow_request: WorkflowRequest,
    background_tasks: BackgroundTasks
):
    """Start a new workflow."""
    try:
        # Generate run ID
        run_id = str(uuid.uuid4())
        
        # Store workflow
        workflows[run_id] = {
            "run_id": run_id,
            "request": workflow_request.dict(),
            "created_at": datetime.now().isoformat()
        }
        
        # Start workflow in background
        background_tasks.add_task(run_workflow, run_id, workflow_request)
        
        logger.info(f"Workflow started: {run_id} - {workflow_request.run_name}")
        
        return WorkflowResponse(
            run_id=run_id,
            status="STARTED",
            message=f"Workflow '{workflow_request.run_name}' started successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to start workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start workflow: {str(e)}")

@app.get("/runs/{run_id}/status", response_model=RunStatus)
async def get_run_status(run_id: str):
    """Get the status of a workflow run."""
    if run_id not in run_status:
        raise HTTPException(status_code=404, detail="Run not found")
    
    return RunStatus(**run_status[run_id])

@app.get("/runs")
async def list_runs():
    """List all workflow runs."""
    return {
        "runs": list(run_status.values()),
        "count": len(run_status)
    }

@app.get("/runs/{run_id}/artifacts")
async def get_run_artifacts(run_id: str):
    """Get artifacts generated by a workflow run."""
    if run_id not in artifacts:
        return {"artifacts": [], "count": 0}
    
    return {
        "artifacts": artifacts[run_id],
        "count": len(artifacts[run_id])
    }

@app.get("/artifacts/{run_id}/{filename}")
async def download_artifact(run_id: str, filename: str):
    """Download a specific artifact file."""
    # Compose the intended artifact file path within ARTIFACTS_DIR
    base_dir = ARTIFACTS_DIR / run_id
    file_path = (base_dir / filename).resolve()
    # Ensure the file_path is strictly within the artifacts directory
    if not str(file_path).startswith(str(base_dir.resolve()) + os.sep):
        raise HTTPException(status_code=403, detail="Invalid artifact path")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Artifact file not found")
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )

@app.delete("/runs/{run_id}")
async def delete_run(run_id: str):
    """Delete a workflow run and its artifacts."""
    if run_id in workflows:
        del workflows[run_id]
    if run_id in run_status:
        del run_status[run_id]
    if run_id in artifacts:
        del artifacts[run_id]
    
    return {"message": f"Run {run_id} deleted successfully"}

# ─── STARTUP AND SHUTDOWN ────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Initialize the orchestrator on startup."""
    logger.info("Master Orchestrator starting up...")
    
    # Test agent connectivity
    try:
        response = requests.get(f"{EDA_AGENT_URL}/health", timeout=5)
        if response.status_code == 200:
            logger.info("EDA Agent is accessible")
        else:
            logger.warning("EDA Agent health check failed")
    except Exception as e:
        logger.warning(f"EDA Agent not accessible: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Master Orchestrator shutting down...")

# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "master_orchestrator_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 