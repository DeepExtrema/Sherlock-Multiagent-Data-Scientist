# Hybrid API Implementation

## 🎯 Overview

This document describes the **Hybrid API (Router + Translation Queue)** implementation - the first scaffold from the clean-room design package. This implementation adds async natural language to DSL translation capabilities while maintaining backward compatibility with existing endpoints.

## 🏗️ Architecture

```
┌──────────────────────────────┐
│          Clients             │
│ • Web UI  • SDK  • CLI       │
└─────────────┬────────────────┘
              │ REST API calls
┌─────────────▼────────────────┐
│  FastAPI Application         │
│ ─────────────────────────────│
│ Legacy: /workflows           │
│ Hybrid: /api/v1/workflows/*  │
└─────────────┬────────────────┘
              │
    ┌─────────▼─────────┐
    │  Hybrid Router    │
    │ ─────────────────│
    │ • /translate      │
    │ • /translation/*  │
    │ • /dsl           │
    │ • /suggest       │
    └─────────┬─────────┘
              │
    ┌─────────▼─────────┐
    │ Translation Queue │
    │ ─────────────────│
    │ • Redis backend   │
    │ • Token tracking  │
    │ • Status polling  │
    │ • Retry handling  │
    └─────────┬─────────┘
              │
    ┌─────────▼─────────┐
    │ Translation Worker │
    │ ──────────────────│
    │ • Background proc │
    │ • LLM calls       │
    │ • Validation      │
    │ • Error handling  │
    └───────────────────┘
```

## 🚀 New Endpoints

### 1. `/api/v1/workflows/translate` (POST)

**Purpose**: Submit natural language for async translation to DSL

**Request Body**:
```json
{
  "natural_language": "Load sales_data.csv, analyze missing values, create histogram",
  "client_id": "optional_client_id",
  "priority": 5,
  "metadata": {"context": "user_workflow"}
}
```

**Response** (202 Accepted):
```json
{
  "token": "abc123def456...",
  "status": "queued",
  "estimated_completion_seconds": 45,
  "message": "Translation queued successfully. Use the token to poll for results."
}
```

### 2. `/api/v1/translation/{token}` (GET)

**Purpose**: Poll translation status by token

**Response Examples**:

**Queued/Processing**:
```json
{
  "token": "abc123def456...",
  "status": "processing",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:15Z",
  "retries": 0,
  "metadata": {"priority": 5}
}
```

**Success**:
```json
{
  "token": "abc123def456...",
  "status": "done",
  "dsl": "name: 'Data Analysis'\ntasks:\n  - id: load_data\n    agent: eda_agent\n    action: load_data\n    params:\n      file: sales_data.csv\n  - id: missing_analysis\n    agent: eda_agent\n    action: missing_data_analysis\n    depends_on: [load_data]\n  - id: histogram\n    agent: eda_agent\n    action: create_visualization\n    params:\n      type: histogram\n    depends_on: [load_data]",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:45Z"
}
```

**Error States**:
```json
{
  "token": "abc123def456...",
  "status": "error",
  "error_message": "Translation failed after 3 attempts: Invalid YAML structure",
  "error_details": {"retries": 3, "last_error": "YAML parsing failed"},
  "updated_at": "2024-01-15T10:32:00Z"
}
```

**Needs Human**:
```json
{
  "token": "abc123def456...",
  "status": "needs_human",
  "error_message": "Request too complex for automated translation",
  "error_details": {"context": "Complex multi-step ML pipeline"},
  "updated_at": "2024-01-15T10:31:30Z"
}
```

### 3. `/api/v1/workflows/dsl` (POST)

**Purpose**: Execute workflow from DSL YAML directly

**Request Body**:
```json
{
  "dsl_yaml": "name: 'Test Workflow'\ntasks:\n  - id: task1\n    agent: eda_agent\n    action: load_data\n    params:\n      file: data.csv",
  "client_id": "optional_client_id",
  "validate_only": false,
  "metadata": {"source": "direct_dsl"}
}
```

**Response** (200 OK):
```json
{
  "workflow_id": "wf_789xyz",
  "status": "running",
  "message": "DSL workflow started successfully",
  "validation_results": {
    "valid": true,
    "warnings": [],
    "parsed_workflow": {...}
  }
}
```

### 4. `/api/v1/workflows/suggest` (POST)

**Purpose**: Generate workflow suggestions based on context

**Request Body**:
```json
{
  "context": "I want to analyze customer purchase patterns",
  "domain": "data-science",
  "complexity": "medium"
}
```

**Response** (200 OK):
```json
{
  "suggestions": [
    {
      "title": "Customer Segmentation Analysis",
      "description": "Analyze purchase patterns and segment customers",
      "dsl": "name: 'Customer Analysis'\ntasks: [....]",
      "estimated_minutes": 15
    },
    {...}
  ],
  "context": "I want to analyze customer purchase patterns",
  "domain": "data-science",
  "complexity": "medium",
  "generated_at": "2024-01-15T10:30:00Z"
}
```

## 🛡️ Edge Case Handling

The implementation includes comprehensive edge-case handling:

### Translation Failures
- **Schema Validation**: DSL must contain valid YAML with required fields
- **Retry Logic**: Failed translations retry up to 3 times with exponential backoff
- **Timeout Handling**: Translations older than 5 minutes are automatically marked as timed out
- **LLM Errors**: Guardrails detect invalid outputs and trigger repair loops

### Error States
| Error Type | Status Code | Response |
|------------|-------------|----------|
| Invalid token format | 400 | `{"detail": "Invalid token format"}` |
| Token not found | 404 | `{"detail": "Translation token not found or expired"}` |
| Empty natural language | 422 | Validation error with details |
| Invalid YAML | 400 | `{"detail": "Invalid YAML format: ..."}` |
| Rate limit exceeded | 429 | `{"detail": "Rate limit exceeded. Please try again later."}` |

### Infrastructure Resilience
- **Redis Fallback**: If Redis is unavailable, falls back to in-memory queue
- **LLM Fallback**: If LLM translation fails, provides rule-based suggestions
- **Graceful Degradation**: Components continue working even if some dependencies fail

## 🔧 Configuration

Add to `config.yaml`:

```yaml
master_orchestrator:
  # Hybrid API Translation Queue Settings
  translation_queue:
    redis_url: "redis://localhost:6379"
    queue_name: "translation:q"
    token_prefix: "translation:"
    timeout_seconds: 300          # 5 minutes
    max_retries: 3
    retry_delay_seconds: 5
    cleanup_interval_seconds: 300 # Clean up expired tokens every 5 minutes
```

## 🧪 Testing

A comprehensive test suite is provided in `test_hybrid_api.py`:

```bash
# Run tests (requires aiohttp)
pip install aiohttp
python test_hybrid_api.py
```

Test coverage includes:
- ✅ Health check and component status
- ✅ Complete async translation workflow
- ✅ DSL validation and execution
- ✅ Workflow suggestions generation
- ✅ Edge cases (invalid tokens, malformed requests)
- ✅ Rate limiting functionality

## 📊 Monitoring

### Health Endpoint

`GET /health` provides component status:

```json
{
  "status": "healthy",
  "version": "2.0.0",
  "api_versions": {
    "legacy": "/workflows",
    "hybrid": "/api/v1"
  },
  "components": {
    "translation_queue": true,
    "translation_worker": true,
    "llm_translator": true,
    "workflow_manager": true,
    "redis_available": true
  }
}
```

### Statistics Endpoint

`GET /stats` provides operational metrics:

```json
{
  "translation_queue": {
    "translations_queued": 150,
    "translations_completed": 145,
    "translations_failed": 3,
    "translations_timed_out": 2,
    "translations_needs_human": 5,
    "use_redis": true,
    "queue_size": 2
  },
  "workflow_manager": {...},
  "decision_engine": {...}
}
```

## 🚀 Usage Examples

### Basic Workflow Translation

```bash
# 1. Submit translation
curl -X POST http://localhost:8001/api/v1/workflows/translate \
  -H "Content-Type: application/json" \
  -d '{
    "natural_language": "Load iris.csv and create a scatter plot of sepal length vs width",
    "priority": 7
  }'

# Response: {"token": "abc123...", "status": "queued", ...}

# 2. Poll for completion
curl http://localhost:8001/api/v1/translation/abc123...

# 3. When status="done", get DSL and execute
curl -X POST http://localhost:8001/api/v1/workflows/dsl \
  -H "Content-Type: application/json" \
  -d '{
    "dsl_yaml": "<generated_dsl_from_step_2>"
  }'
```

### Direct DSL Execution

```bash
curl -X POST http://localhost:8001/api/v1/workflows/dsl \
  -H "Content-Type: application/json" \
  -d '{
    "dsl_yaml": "name: '\''Quick Analysis'\''\ntasks:\n  - id: load\n    agent: eda_agent\n    action: load_data\n    params:\n      file: data.csv",
    "validate_only": true
  }'
```

### Get Suggestions

```bash
curl -X POST http://localhost:8001/api/v1/workflows/suggest \
  -H "Content-Type: application/json" \
  -d '{
    "context": "I need to predict house prices",
    "domain": "machine-learning",
    "complexity": "medium"
  }'
```

## 🔄 Migration Guide

### From Legacy API

**Before** (Legacy):
```python
response = requests.post("/workflows", json={
    "natural_language": "analyze data"
})
# Synchronous, may timeout, limited error handling
```

**After** (Hybrid API):
```python
# Step 1: Submit
response = requests.post("/api/v1/workflows/translate", json={
    "natural_language": "analyze data",
    "priority": 5
})
token = response.json()["token"]

# Step 2: Poll
while True:
    status_response = requests.get(f"/api/v1/translation/{token}")
    status_data = status_response.json()
    
    if status_data["status"] == "done":
        dsl = status_data["dsl"]
        break
    elif status_data["status"] in ["error", "timeout"]:
        handle_error(status_data)
        break
    
    time.sleep(2)

# Step 3: Execute
requests.post("/api/v1/workflows/dsl", json={"dsl_yaml": dsl})
```

## 🎯 Next Steps

This Hybrid API implementation provides the foundation for:

1. **Deadlock Monitor** - Detect and handle stuck workflows
2. **Guardrails Repair Loop** - Advanced DSL validation and correction
3. **Enhanced Scheduling** - Priority-based task queue with αβγ scoring
4. **Dashboard Integration** - Real-time workflow monitoring UI

The async translation queue architecture scales horizontally and provides the reliability needed for production workflow orchestration.

## 🏁 Summary

✅ **Implemented**: Async translation queue with Redis backing  
✅ **Implemented**: Token-based polling for translation status  
✅ **Implemented**: Direct DSL execution with validation  
✅ **Implemented**: Workflow suggestions generation  
✅ **Implemented**: Comprehensive edge-case handling  
✅ **Implemented**: Backward compatibility with legacy endpoints  
✅ **Implemented**: Full test coverage  

The Hybrid API successfully bridges the gap between simple synchronous translation and the advanced workflow orchestration capabilities described in your clean-room design. 