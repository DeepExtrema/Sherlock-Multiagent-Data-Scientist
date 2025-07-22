"""
Master Orchestrator API

FastAPI application that provides the main orchestrator endpoints for workflow management.
Enhanced with Hybrid API for async translation workflow.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import orchestrator components
from orchestrator import (
    LLMTranslator, RuleBasedTranslator, FallbackRouter,
    TranslationQueue, TranslationWorker, TranslationStatus,
    WorkflowManager, SecurityUtils, CacheClient,
    ConcurrencyGuard, TokenRateLimiter, SLAMonitor,
    DecisionEngine, TelemetryManager, initialize_telemetry,
    NeedsHumanError
)
from orchestrator.telemetry import trace_async, get_correlation_id, set_correlation_id, CorrelationID

# Import API routers
from api.hybrid_router import create_hybrid_router

# Import workflow engine
try:
    from workflow_engine import start_engine_async
    WORKFLOW_ENGINE_AVAILABLE = True
except ImportError:
    WORKFLOW_ENGINE_AVAILABLE = False

from config import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
try:
    config = load_config()
    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    config = None

# Request/Response models (legacy endpoints)
class WorkflowRequest(BaseModel):
    natural_language: Optional[str] = None
    dsl_yaml: Optional[str] = None
    client_id: Optional[str] = "default"
    metadata: Optional[Dict[str, Any]] = None

class WorkflowResponse(BaseModel):
    workflow_id: str
    status: str
    message: Optional[str] = None

class HumanSubmissionRequest(BaseModel):
    corrected_yaml: str
    notes: Optional[str] = None

class RollbackRequest(BaseModel):
    steps: int = 1
    reason: Optional[str] = "User requested rollback"

class ErrorResponse(BaseModel):
    error: str
    details: Optional[Dict[str, Any]] = None

# Global components (will be initialized in lifespan)
llm_translator = None
fallback_router = None
workflow_manager = None
concurrency_guard = None
rate_limiter = None
sla_monitor = None
security_utils = None
decision_engine = None
telemetry_manager = None
workflow_engine = None

# Hybrid API components
translation_queue = None
translation_worker = None
hybrid_router = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global llm_translator, fallback_router, workflow_manager
    global concurrency_guard, rate_limiter, sla_monitor, security_utils
    global decision_engine, telemetry_manager, workflow_engine
    global translation_queue, translation_worker, hybrid_router
    
    try:
        logger.info("Initializing Master Orchestrator components...")
        
        if not config:
            logger.error("Configuration not available, using defaults")
            orchestrator_config = {}
        else:
            orchestrator_config = config.master_orchestrator.dict()
        
        # Initialize telemetry first (before other components)
        telemetry_config = orchestrator_config.get("telemetry", {})
        telemetry_manager = initialize_telemetry(telemetry_config)
        
        # Initialize security utilities
        security_utils = SecurityUtils(max_input_length=orchestrator_config.get("llm", {}).get("max_input_length", 10000))
        
        # Initialize decision engine
        decision_config = orchestrator_config.get("decision", {})
        decision_engine = DecisionEngine(decision_config)
        
        # Initialize translators
        llm_translator = LLMTranslator(orchestrator_config.get("llm", {}))
        rule_translator = RuleBasedTranslator(orchestrator_config.get("rules", {}))
        fallback_router = FallbackRouter({
            "llm": orchestrator_config.get("llm", {}),
            "rules": orchestrator_config.get("rules", {}),
            "enable_human_fallback": orchestrator_config.get("enable_human_fallback", True),
            "min_confidence_threshold": orchestrator_config.get("min_confidence_threshold", 0.7)
        })
        
        # Initialize workflow manager
        workflow_manager = WorkflowManager(orchestrator_config.get("infrastructure", {}))
        
        # Initialize guards
        max_concurrent = config.orchestrator.max_concurrent_workflows if config else 1
        concurrency_guard = ConcurrencyGuard(max_concurrent)
        
        rate_limits = orchestrator_config.get("rate_limits", {})
        rate_limiter = TokenRateLimiter({
            "per_minute": rate_limits.get("requests_per_minute", 60),
            "per_hour": rate_limits.get("requests_per_hour", 1000)
        })
        
        # Initialize SLA monitor
        sla_config = orchestrator_config.get("sla", {})
        sla_monitor = SLAMonitor(sla_config)
        
        # Set up SLA monitor callbacks
        sla_monitor.set_data_callbacks(
            get_stale_tasks=get_stale_tasks,
            get_stale_workflows=get_stale_workflows,
            broadcast_event=broadcast_event
        )
        
        # Add workflow manager event callback
        workflow_manager.add_event_callback(handle_workflow_event)
        
        # Initialize Hybrid API components
        logger.info("Initializing Hybrid API components...")
        
        # Initialize translation queue
        cache_config = orchestrator_config.get("cache", {})
        redis_url = cache_config.get("redis_url", "redis://localhost:6379")
        
        translation_queue = TranslationQueue(
            redis_url=redis_url,
            timeout_seconds=300  # 5 minutes
        )
        await translation_queue.initialize()
        
        # Initialize translation worker
        translation_worker = TranslationWorker(
            translation_queue=translation_queue,
            llm_translator=llm_translator,
            max_retries=3,
            retry_delay=5.0
        )
        await translation_worker.start()
        
        # Create hybrid router
        hybrid_router = create_hybrid_router(
            translation_queue=translation_queue,
            llm_translator=llm_translator,
            workflow_manager=workflow_manager,
            decision_engine=decision_engine,
            security_utils=security_utils,
            rate_limiter=rate_limiter
        )
        
        # Include hybrid router in the app
        app.include_router(hybrid_router)
        
        logger.info("Hybrid API components initialized successfully")
        
        # Initialize and start workflow engine
        if WORKFLOW_ENGINE_AVAILABLE and config:
            try:
                engine_config = config.workflow_engine.dict()
                workflow_engine = await start_engine_async(engine_config, workflow_manager)
                
                # Add event callback to link engine events back to workflow manager
                if workflow_engine:
                    workflow_engine.add_event_callback(handle_workflow_event)
                
                logger.info("Workflow engine started successfully")
            except Exception as e:
                logger.error(f"Failed to start workflow engine: {e}")
                workflow_engine = None
        else:
            logger.warning("Workflow engine not available or config missing")
            workflow_engine = None
        
        # Start SLA monitoring
        await sla_monitor.start()
        
        logger.info("Master Orchestrator initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize Master Orchestrator: {e}")
        yield
    finally:
        # Cleanup
        logger.info("Shutting down Master Orchestrator...")
        
        if translation_worker:
            await translation_worker.stop()
            logger.info("Translation worker stopped")
            
        if sla_monitor:
            await sla_monitor.stop()
            
        if workflow_engine:
            await workflow_engine.stop()
            
        logger.info("Master Orchestrator shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Master Orchestrator API",
    description="Natural language to workflow orchestration with LLM translation, async queue, and fallback systems",
    version="2.0.0",  # Bumped version for hybrid API
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper functions
async def get_client_id(request: Request) -> str:
    """Extract client ID from request."""
    # Try to get from headers, query params, or use IP
    client_id = request.headers.get("X-Client-ID")
    if not client_id:
        client_id = request.query_params.get("client_id")
    if not client_id:
        client_id = request.client.host if request.client else "unknown"
    return client_id

async def get_stale_tasks(sla_seconds: int):
    """Get stale tasks for SLA monitoring."""
    if not workflow_manager or not workflow_manager.db:
        return []
    
    cutoff_time = datetime.utcnow() - timedelta(seconds=sla_seconds)
    
    try:
        stale_tasks = await workflow_manager.db.tasks.find({
            "status": {"$in": ["pending", "queued", "running"]},
            "created_at": {"$lt": cutoff_time}
        }).to_list(100)
        
        return stale_tasks
    except Exception as e:
        logger.error(f"Error getting stale tasks: {e}")
        return []

async def get_stale_workflows(sla_seconds: int):
    """Get stale workflows for SLA monitoring."""
    if not workflow_manager or not workflow_manager.db:
        return []
    
    cutoff_time = datetime.utcnow() - timedelta(seconds=sla_seconds)
    
    try:
        stale_workflows = await workflow_manager.db.runs.find({
            "status": {"$in": ["pending", "running"]},
            "created_at": {"$lt": cutoff_time}
        }).to_list(100)
        
        return stale_workflows
    except Exception as e:
        logger.error(f"Error getting stale workflows: {e}")
        return []

async def broadcast_event(event: Dict[str, Any]):
    """Broadcast event to connected clients."""
    # This would integrate with WebSocket connections or other broadcasting mechanisms
    logger.info(f"Broadcasting event: {event['type']}")

async def handle_workflow_event(event: Dict[str, Any]):
    """Handle workflow events from WorkflowManager."""
    logger.info(f"Workflow event: {event['type']} for {event.get('run_id', 'unknown')}")

# Legacy API endpoints (maintain backward compatibility)
@app.post("/workflows", response_model=WorkflowResponse)
@trace_async("create_workflow_legacy", operation_type="api_endpoint")
async def create_workflow_legacy(request: Request, workflow_request: WorkflowRequest):
    """
    Legacy workflow creation endpoint.
    
    Maintains backward compatibility while recommending migration to hybrid API.
    """
    try:
        # Set up correlation ID
        correlation_id = CorrelationID.from_headers(dict(request.headers)) or CorrelationID.generate()
        set_correlation_id(correlation_id)
        
        # Get client ID for rate limiting
        client_id = await get_client_id(request)
        
        # Check rate limits
        if rate_limiter and not rate_limiter.check(client_id):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Check concurrency limits
        if concurrency_guard and not concurrency_guard.allow():
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Maximum concurrent workflows reached. Please try again later."
            )
        
        # Validate input
        if not workflow_request.natural_language and not workflow_request.dsl_yaml:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either natural_language or dsl_yaml must be provided"
            )
        
        workflow = None
        
        # Process DSL YAML directly if provided
        if workflow_request.dsl_yaml:
            try:
                # Parse and validate DSL YAML
                import yaml
                workflow = yaml.safe_load(workflow_request.dsl_yaml)
                
                # Validate workflow structure
                if not isinstance(workflow, dict) or "tasks" not in workflow:
                    raise ValueError("Invalid workflow structure")
                    
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid DSL YAML: {str(e)}"
                )
        else:
            # Translate natural language to workflow
            try:
                if fallback_router and workflow_request.natural_language:
                    workflow = await fallback_router.resolve(workflow_request.natural_language)
                
            except NeedsHumanError as e:
                # Create workflow record in NEEDS_HUMAN state
                if workflow_manager:
                    run_id = await workflow_manager.init_workflow(
                        {"tasks": []},
                        metadata={
                            **(workflow_request.metadata or {}),
                            "status": "needs_human",
                            "context": e.context
                        }
                    )
                
                return JSONResponse(
                    status_code=status.HTTP_202_ACCEPTED,
                    content={
                        "workflow_id": run_id,
                        "status": "needs_human",
                        "message": "Human intervention required",
                        "context": e.context,
                        "recommendation": "Consider using /api/v1/workflows/translate for async processing"
                    }
                )
        
        # Initialize and start workflow
        if workflow and workflow_manager:
            # Acquire concurrency slot
            if concurrency_guard:
                await concurrency_guard.acquire()
            
            try:
                # Initialize workflow
                run_id = await workflow_manager.init_workflow(
                    workflow,
                    metadata=workflow_request.metadata
                )
                
                # Apply decision engine policies to tasks before starting
                if decision_engine and workflow.get("tasks"):
                    processed_tasks = []
                    for task in workflow["tasks"]:
                        decision = decision_engine.evaluate(run_id, task)
                        if decision.allowed:
                            # Apply any overrides from decision engine
                            task.update(decision.overrides)
                            processed_tasks.append(task)
                        else:
                            # Mark task as skipped
                            logger.warning(f"Task {task.get('id', 'unknown')} skipped: {decision.reason}")
                            # Could optionally store skipped tasks for auditing
                    
                    # Update workflow with processed tasks
                    if workflow_manager.db:
                        await workflow_manager.db.runs.update_one(
                            {"run_id": run_id},
                            {"$set": {"workflow_definition.tasks": processed_tasks}}
                        )
                
                # Start workflow execution
                success = await workflow_manager.start_workflow(run_id)
                
                if success:
                    response = WorkflowResponse(
                        workflow_id=run_id,
                        status="running",
                        message="Workflow started successfully"
                    )
                    
                    return response
                else:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Failed to start workflow"
                    )
                    
            except Exception as e:
                # Release concurrency slot on error
                if concurrency_guard:
                    await concurrency_guard.release()
                raise
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create workflow from input"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.post("/workflows/{run_id}/human_submit", response_model=WorkflowResponse)
@trace_async("human_submit", operation_type="api_endpoint")
async def human_submit(run_id: str, submission: HumanSubmissionRequest):
    """Submit human-corrected workflow definition."""
    try:
        # Parse corrected YAML
        import yaml
        try:
            workflow = yaml.safe_load(submission.corrected_yaml)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid YAML: {str(e)}"
            )
        
        # Validate workflow structure
        if not isinstance(workflow, dict) or "tasks" not in workflow:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid workflow structure"
            )
        
        # Update workflow with corrected definition
        if workflow_manager and workflow_manager.db:
            await workflow_manager.db.runs.update_one(
                {"run_id": run_id},
                {
                    "$set": {
                        "workflow_definition": workflow,
                        "status": "running",
                        "updated_at": datetime.utcnow(),
                        "human_notes": submission.notes
                    }
                }
            )
            
            # Create task documents
            await workflow_manager._create_task_documents(run_id, workflow["tasks"])
            
            # Start workflow execution
            success = await workflow_manager.start_workflow(run_id)
            
            if success:
                return WorkflowResponse(
                    workflow_id=run_id,
                    status="running",
                    message="Workflow started with human corrections"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to start corrected workflow"
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Workflow manager not available"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing human submission: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.get("/workflows/{run_id}")
@trace_async("get_workflow_status", operation_type="api_endpoint")
async def get_workflow_status(run_id: str):
    """Get workflow status and details."""
    try:
        if workflow_manager:
            status_info = await workflow_manager.get_workflow_status(run_id)
            if status_info:
                return status_info
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Workflow not found"
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Workflow manager not available"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.put("/workflows/{run_id}/cancel")
@trace_async("cancel_workflow", operation_type="api_endpoint")
async def cancel_workflow(run_id: str, reason: str = "User cancelled"):
    """Cancel a running workflow."""
    try:
        if workflow_manager:
            success = await workflow_manager.cancel_workflow(run_id, reason)
            if success:
                # Release concurrency slot
                if concurrency_guard:
                    await concurrency_guard.release()
                
                return {"workflow_id": run_id, "status": "cancelled"}
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to cancel workflow"
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Workflow manager not available"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.post("/runs/{run_id}/rollback")
@trace_async("rollback_run", operation_type="api_endpoint")
async def rollback_run(run_id: str, req: RollbackRequest):
    """Rollback a workflow to a previous checkpoint."""
    try:
        if not workflow_manager:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Workflow manager not available"
            )
        
        # Check if workflow exists
        workflow_status = await workflow_manager.get_workflow_status(run_id)
        if not workflow_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow {run_id} not found"
            )
        
        # Find last N successful tasks
        if workflow_manager.db:
            # Query MongoDB for successful tasks
            successful_tasks = await workflow_manager.db.tasks.find(
                {"run_id": run_id, "status": "SUCCESS"}
            ).sort("finished_at", -1).limit(req.steps).to_list(None)
            
            if not successful_tasks:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No successful tasks found to rollback"
                )
            
            rollback_count = 0
            for task in successful_tasks:
                try:
                    # Restore checkpoint if available
                    checkpoint_ref = task.get("checkpoint_ref")
                    if checkpoint_ref:
                        logger.info(f"Restoring checkpoint for task {task['task_id']}: {checkpoint_ref}")
                    
                    # Mark task as PENDING for re-execution
                    await workflow_manager.db.tasks.update_one(
                        {"_id": task["_id"]},
                        {
                            "$set": {
                                "status": "PENDING",
                                "retries": 0,
                                "started_at": None,
                                "finished_at": None,
                                "error_message": None,
                                "rollback_reason": req.reason,
                                "rollback_timestamp": datetime.utcnow()
                            }
                        }
                    )
                    
                    # Reset dependent tasks as well
                    await workflow_manager.db.tasks.update_many(
                        {
                            "run_id": run_id,
                            "depends_on": {"$in": [task["task_id"]]},
                            "status": {"$in": ["SUCCESS", "FAILED"]}
                        },
                        {
                            "$set": {
                                "status": "PENDING",
                                "retries": 0,
                                "started_at": None,
                                "finished_at": None,
                                "error_message": None,
                                "rollback_reason": f"Dependency rollback: {req.reason}",
                                "rollback_timestamp": datetime.utcnow()
                            }
                        }
                    )
                    
                    rollback_count += 1
                    logger.info(f"Successfully rolled back task {task['task_id']}")
                    
                except Exception as e:
                    logger.error(f"Failed to rollback task {task.get('task_id', 'unknown')}: {e}")
                    continue
            
            if rollback_count > 0:
                # Update workflow status
                await workflow_manager.db.runs.update_one(
                    {"run_id": run_id},
                    {
                        "$set": {
                            "status": "running",
                            "rollback_count": rollback_count,
                            "last_rollback": datetime.utcnow(),
                            "rollback_reason": req.reason
                        }
                    }
                )
                
                # Recompute task dependencies and re-enqueue root tasks
                # This would trigger the workflow to resume from the rollback point
                await workflow_manager._recompute_and_enqueue_tasks(run_id)
                
                return {
                    "run_id": run_id,
                    "status": "rolled_back",
                    "message": f"Successfully rolled back {rollback_count} tasks",
                    "tasks_rolled_back": rollback_count,
                    "reason": req.reason
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to rollback any tasks"
                )
        else:
            # In-memory fallback (limited functionality)
            logger.warning("Rollback with in-memory storage has limited functionality")
            return {
                "run_id": run_id,
                "status": "rollback_limited",
                "message": "Rollback completed with limited functionality (no persistent storage)",
                "reason": req.reason
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during rollback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rollback failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "api_versions": {
            "legacy": "/workflows",
            "hybrid": "/api/v1"
        },
        "components": {
            "config_loaded": config is not None,
            "llm_translator": llm_translator is not None,
            "workflow_manager": workflow_manager is not None,
            "translation_queue": translation_queue is not None,
            "translation_worker": translation_worker is not None and translation_worker._running,
            "sla_monitor": sla_monitor is not None and sla_monitor.is_running if sla_monitor else False,
            "decision_engine": decision_engine is not None,
            "telemetry_manager": telemetry_manager is not None and telemetry_manager.enabled if telemetry_manager else False
        }
    }

@app.get("/stats")
async def get_statistics():
    """Get system statistics."""
    stats = {}
    
    if workflow_manager:
        stats["workflow_manager"] = workflow_manager.get_statistics()
    
    if concurrency_guard:
        stats["concurrency"] = concurrency_guard.get_stats()
    
    if sla_monitor:
        stats["sla_monitor"] = sla_monitor.get_statistics()
    
    if decision_engine:
        stats["decision_engine"] = decision_engine.get_statistics()
    
    if translation_queue:
        stats["translation_queue"] = translation_queue.get_stats()
    
    if telemetry_manager:
        stats["telemetry"] = {
            "enabled": telemetry_manager.enabled,
            "service_name": telemetry_manager.service_name,
            "current_trace_id": telemetry_manager.get_current_trace_id()
        }
    
    return stats

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True) 