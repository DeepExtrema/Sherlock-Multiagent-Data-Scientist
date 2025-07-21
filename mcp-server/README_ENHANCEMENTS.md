# Master Orchestrator Enhancements

This document describes the four major enhancements added to the Master Orchestrator following the pseudocode integration plan:

## 1. Decision Engine Hook 🎯

**Purpose**: Centralize all policy decisions ("should I run?", "which model?", "GPU vs. CPU?") outside of DAG plumbing.

### Features
- **Resource Allocation**: Automatically determines CPU vs GPU vs Memory-optimized resources
- **Model Selection**: Chooses appropriate LLM models based on task complexity
- **Priority Assignment**: Assigns task priorities (1-10) based on business rules
- **Business Rules**: Enforces blocked actions, maintenance windows, and resource limits
- **Auto-retry Logic**: Decides if failed tasks should be automatically retried

### Integration Points
```python
# Before enqueueing any task:
decision = decision_engine.evaluate(run_id, task_meta)
if not decision.allowed:
    # mark SKIPPED in MongoDB and skip enqueue
    await db.tasks.update_one({...}, {'$set': {'status':'SKIPPED'}})
    return
task_meta.update(decision.overrides)
```

### Configuration
```yaml
master_orchestrator:
  decision:
    gpu_agents: ["ml_agent", "deep_learning_agent"]
    cpu_agents: ["eda_agent", "data_agent", "analysis_agent"]
    max_task_count: 100
    blocked_actions: []
    priority_agents: ["critical_analysis_agent"]
    maintenance_windows: []
    resource_limits:
      max_concurrent_gpu_tasks: 2
      max_concurrent_cpu_tasks: 10
      max_memory_per_task_gb: 16
```

## 2. Telemetry & Tracing 📊

**Purpose**: End-to-end distributed tracing with correlation IDs for debugging cross-service latencies and failures.

### Features
- **OpenTelemetry Integration**: Full OTLP-compatible distributed tracing
- **Correlation IDs**: Automatic generation and propagation across services
- **Kafka Header Propagation**: Trace context flows through Kafka messages
- **API Endpoint Tracing**: All endpoints automatically traced
- **Performance Monitoring**: Spans show timing data and error states

### Integration Points
```python
@trace_async("create_workflow", operation_type="api_endpoint")
async def create_workflow(request: Request, workflow_request: WorkflowRequest):
    with tracer.start_as_current_span("create_workflow") as span:
        span.set_attribute("run.request_type", "NL" if req.is_natural_language else "DSL")
        run_id = str(uuid.uuid4())
        span.set_attribute("run.id", run_id)
        # rest of logic...
```

### Configuration
```yaml
master_orchestrator:
  telemetry:
    enabled: true
    service_name: "master-orchestrator"
    service_version: "1.0.0"
    otlp_endpoint: "http://localhost:4318/v1/traces"
```

### Dependencies
```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
```

## 3. Enhanced Cache-Backed LLM Memoization ⚡

**Status**: ✅ Already implemented in `translator.py` with CacheClient

**Purpose**: Avoid repeated LLM calls on identical prompts (cost, speed, consistency).

### Features
- **Hash-based Caching**: SHA-256 hashes of prompts for cache keys
- **Model Version Awareness**: Cache keys include model version
- **TTL Support**: Configurable cache expiration (default 1 hour)
- **Redis Backend**: Persistent caching across restarts

### Usage
```python
cache_key = f"llm:{hash(prompt)}:{config['model_version']}"
if cache.exists(cache_key):
    return cache.get(cache_key)
result = call_guardrails_llm(prompt)
cache.set(cache_key, result, ttl=3600)
return result
```

## 4. Background SLA Monitor 🔍

**Status**: ✅ Already implemented in `sla_monitor.py`

**Purpose**: Proactive detection and alerting on tasks or workflows breaching SLAs.

### Features
- **Automatic Monitoring**: Background task checks for SLA violations
- **Configurable Thresholds**: Separate timeouts for tasks and workflows
- **Alert Generation**: Broadcasts events for violations
- **Statistics Tracking**: Monitors performance metrics

### Integration
```python
async def sla_monitor_loop():
    while True:
        now = datetime.utcnow()
        # Check tasks stuck in QUEUED or RUNNING too long
        overdue = await db.tasks.find({
            "status": {"$in": ["QUEUED","RUNNING"]},
            "$expr": {"$gt": [
                {"$subtract": [now, "$started_at"]},
                config['sla']['task_max_seconds'] * 1000
            ]}
        }).to_list(None)
        # Process violations...
        await asyncio.sleep(config['sla']['check_interval_s'])
```

## 5. Bonus: Rollback Implementation 🔄

**Purpose**: Enable workflow recovery by rolling back to previous checkpoints.

### Features
- **Checkpoint-based Recovery**: Restore from stored task checkpoints
- **Dependency Cascade**: Automatically resets dependent tasks
- **Configurable Steps**: Roll back N number of completed tasks
- **State Restoration**: Marks tasks as PENDING for re-execution
- **Audit Trail**: Tracks rollback reasons and timestamps

### API Usage
```bash
POST /runs/{run_id}/rollback
{
  "steps": 3,
  "reason": "Data corruption detected"
}
```

### Implementation
```python
@app.post("/runs/{run_id}/rollback")
async def rollback_run(run_id: str, req: RollbackRequest):
    # 1) Fetch last req.steps SUCCESS tasks
    successful_tasks = await db.tasks.find(
        {"run_id": run_id, "status": "SUCCESS"}
    ).sort("finished_at", -1).limit(req.steps).to_list(None)
    
    # 2) For each: restore state from checkpoint, mark PENDING
    for task in successful_tasks:
        await db.tasks.update_one({"_id": task["_id"]}, 
            {"$set": {"status": "PENDING", "retries": 0}})
    
    # 3) Recompute dependencies and enqueue root tasks
    await workflow_manager._recompute_and_enqueue_tasks(run_id)
```

## Architecture Impact

### Logic Placement Diagram
```
Master Orchestrator
├─ API Intake & Guards
│   ├─ TokenRateLimiter (/workflows endpoint)
│   ├─ ConcurrencyGuard (max concurrent runs)
│   └─ SLA Monitor (background task) ✅
├─ Input Translators
│   ├─ DSL Parser & Validator (Pydantic + Guardrails)
│   └─ LLM Translator (+CacheClient & GuardrailsAI) ✅
├─ Fallback Router (DSL vs NL vs Suggest)
├─ Decision Engine Hook 🆕
│   └─ evaluate() before enqueue
├─ Workflow Initialization
│   ├─ initialize_run()
│   └─ enqueue_initial_tasks()
├─ Event Handling
│   ├─ handle_event() (TASK_STARTED/SUCCESS/FAILURE)
│   └─ RetryTracker backoff & re-enqueue
└─ Telemetry & Tracing 🆕
    ├─ OpenTelemetry spans around APIs and Kafka
    └─ Correlation IDs (run_id, task_id) in headers
```

## Benefits

1. **Maintainability**: Policy decisions centralized in DecisionEngine, not scattered in DAG logic
2. **Observability**: Full distributed tracing enables rapid debugging and performance analysis
3. **Reliability**: SLA monitoring and rollback capabilities improve system resilience
4. **Cost Efficiency**: LLM memoization reduces redundant API calls
5. **Scalability**: Resource allocation decisions adapt to workload characteristics

## Configuration Summary

All enhancements are configurable through `config.yaml`:

```yaml
master_orchestrator:
  # Decision Engine
  decision:
    gpu_agents: ["ml_agent", "deep_learning_agent"]
    cpu_agents: ["eda_agent", "data_agent"]
    max_task_count: 100
    resource_limits:
      max_concurrent_gpu_tasks: 2
      max_memory_per_task_gb: 16
  
  # Telemetry  
  telemetry:
    enabled: true
    service_name: "master-orchestrator"
    otlp_endpoint: "http://localhost:4318/v1/traces"
  
  # SLA Monitoring (existing)
  sla:
    task_max_seconds: 600
    workflow_max_seconds: 3600
    check_interval_seconds: 30
  
  # Cache (existing)
  cache:
    redis_url: "redis://localhost:6379"
    default_ttl: 3600
```

## Usage Examples

### Creating a Workflow with Enhanced Decision Making
```python
# POST /workflows
{
  "natural_language": "Train a machine learning model on large dataset",
  "metadata": {"urgent": true, "dataset_size": 1000000}
}

# Decision Engine automatically:
# - Assigns GPU resources (ml_agent detected)
# - Sets high priority (urgent=true)
# - Allocates 16GB memory (large dataset)
# - Selects distributed training (dataset_size > 100K)
```

### Monitoring with Telemetry
```bash
# Check telemetry status
GET /stats

{
  "telemetry": {
    "enabled": true,
    "service_name": "master-orchestrator", 
    "current_trace_id": "abc123..."
  }
}
```

### Rollback on Failure
```bash
# Rollback last 2 completed tasks
POST /runs/run_123/rollback
{
  "steps": 2,
  "reason": "Model accuracy below threshold"
}
```

This comprehensive enhancement package transforms the Master Orchestrator into a production-ready, observable, and resilient workflow orchestration system. 