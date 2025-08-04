# Refinery Agent + FE Module Integration Guide

## Overview

The refinery agent has been enhanced with seamless integration of the FE module, providing a unified interface for both basic and advanced feature engineering capabilities. This integration follows a smart routing approach that automatically chooses the best backend for each task.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Master Orchestrator                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Refinery Agent v2.0                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Data Quality  │  │ Feature Engine  │  │   Pipeline   │ │
│  │   (Read-only)   │  │   (Basic)       │  │  Management  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    FE Module (Advanced Backend)             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Advanced Imput. │  │ Advanced Encod. │  │ Feature Sel. │ │
│  │ (KNN, MICE)     │  │ (Target, Hash)  │  │ (VIF, MI)    │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. **Smart Routing**
- Automatic detection of task complexity
- Seamless delegation to appropriate backend
- Fallback mechanisms for reliability

### 2. **Unified Context Management**
- Redis-backed persistent pipeline state
- Shared context between refinery and FE module
- Pipeline reproducibility and resumption

### 3. **Progressive Enhancement**
- Start with basic capabilities
- Automatically upgrade to advanced when needed
- Consistent API regardless of backend

## Configuration

### Environment Variables

```bash
# FE Module Integration
FE_MODULE_ENABLED=true                    # Enable/disable FE module
AUTO_ADVANCED_ROUTING=true               # Auto-detect complexity
FE_CONTEXT_TTL=3600                      # Context TTL in seconds
REDIS_URL=redis://localhost:6379         # Redis connection

# Data Quality Thresholds
DRIFT_NUMERIC_P95_THRESHOLD=0.1
DRIFT_CATEGORICAL_PSI_THRESHOLD=0.25
MISSING_VALUES_THRESHOLD=0.5
CORRELATION_THRESHOLD=0.95
```

### Configuration File

```yaml
# Enhanced config.yaml
refinery_agent:
  # Basic capabilities
  basic_imputation_strategies: ["mean", "median", "mode", "zero"]
  basic_encoding_strategies: ["one_hot", "label", "ordinal"]
  
  # Advanced capabilities (FE module integration)
  advanced_processing:
    enabled: true
    auto_detect_complexity: true
    fallback_to_basic: true
    
  # FE module integration
  fe_module:
    redis_url: "redis://localhost:6379"
    context_ttl: 3600
    max_context_size: "100MB"
    
  # Smart routing thresholds
  routing_thresholds:
    cardinality_threshold: 50
    missing_rate_threshold: 0.3
    dataset_size_threshold: 10000
```

## API Usage

### 1. **Data Quality Operations (Read-only)**

```python
# Basic quality check
quality_report = await refinery_agent.execute({
    "action": "comprehensive_quality_report",
    "params": {
        "data_path": "data.csv",
        "target_col": "target"
    }
})

# Drift detection
drift_report = await refinery_agent.execute({
    "action": "check_drift",
    "params": {
        "current_data_path": "current.csv",
        "reference_data_path": "reference.csv"
    }
})
```

### 2. **Basic Feature Engineering (Refinery Agent)**

```python
# Basic imputation
result = await refinery_agent.execute({
    "action": "basic_impute_missing_values",
    "params": {
        "data_path": "data.csv",
        "run_id": "run_123",
        "strategy": "auto"
    }
})

# Feature role assignment
result = await refinery_agent.execute({
    "action": "assign_feature_roles",
    "params": {
        "data_path": "data.csv",
        "run_id": "run_123",
        "target_col": "target"
    }
})
```

### 3. **Advanced Feature Engineering (FE Module)**

```python
# Advanced imputation with pattern detection
result = await refinery_agent.execute({
    "action": "advanced_impute_missing_values",
    "params": {
        "data_path": "data.csv",
        "run_id": "run_123",
        "strategy": "auto",
        "pattern_analysis": True
    }
})

# Target encoding with cross-validation
result = await refinery_agent.execute({
    "action": "advanced_encode_categorical_features",
    "params": {
        "data_path": "data.csv",
        "run_id": "run_123",
        "strategy": "target",
        "target_column": "target",
        "cv_folds": 5
    }
})

# Feature selection with VIF analysis
result = await refinery_agent.execute({
    "action": "advanced_feature_selection",
    "params": {
        "data_path": "data.csv",
        "run_id": "run_123",
        "vif_analysis": True,
        "vif_threshold": 5.0
    }
})
```

### 4. **Automatic Routing (Recommended)**

```python
# Smart routing - refinery agent automatically chooses best backend
result = await refinery_agent.execute({
    "action": "impute_missing_values",  # Generic action name
    "params": {
        "data_path": "data.csv",
        "run_id": "run_123",
        "strategy": "knn",  # Triggers advanced processing
        "k": 5
    }
})
```

## Smart Routing Logic

### Complexity Indicators

The system automatically routes to the FE module when it detects:

1. **High Cardinality**: `cardinality > 50`
2. **Missing Pattern Analysis**: `pattern_analysis = True`
3. **Target Encoding**: `encoding_strategy = "target"`
4. **Multicollinearity Detection**: `vif_analysis = True`
5. **Cross-Validation**: `cv_folds > 0`
6. **Advanced Imputation**: `strategy in ["knn", "mice"]`
7. **Hash Encoding**: `encoding_strategy = "hash"`
8. **Feature Interactions**: `interaction_degree > 2`

### Routing Examples

```python
# Routes to refinery agent (basic)
{
    "action": "impute_missing_values",
    "params": {"strategy": "mean"}
}

# Routes to FE module (advanced)
{
    "action": "impute_missing_values",
    "params": {"strategy": "knn", "k": 5}
}

# Routes to FE module (complex)
{
    "action": "encode_categorical_features",
    "params": {"strategy": "target", "cv_folds": 5}
}
```

## Monitoring and Observability

### Prometheus Metrics

```python
# Request metrics by backend
refinery_requests_total{action="impute_missing_values", mode="feature_engineering", backend="refinery_basic", status="success"}
refinery_requests_total{action="impute_missing_values", mode="feature_engineering", backend="fe_module", status="success"}

# FE module usage
fe_module_usage_total{action="advanced_impute_missing_values", complexity="high"}

# Performance metrics
refinery_request_duration_seconds{action="impute_missing_values", mode="feature_engineering", backend="fe_module"}
```

### Health Checks

```bash
# Check service health
curl http://localhost:8005/health

# Response includes backend status
{
    "status": "ok",
    "version": "2.0.0",
    "modes": ["data_quality", "feature_engineering"],
    "backends": ["refinery_basic", "fe_module"],
    "fe_module_available": true,
    "fe_module_enabled": true
}
```

### Pipeline Status

```bash
# Check active pipelines
curl http://localhost:8005/pipelines

# Response includes backend usage
{
    "active_pipelines": 5,
    "pipelines": ["run_123:unified", "run_124:unified"],
    "modes": {
        "data_quality": 2,
        "feature_engineering": 3
    },
    "fe_module_available": true,
    "fe_module_enabled": true
}
```

## Error Handling

### Graceful Degradation

```python
# If FE module is unavailable, falls back to basic
try:
    result = await refinery_agent.execute({
        "action": "advanced_impute_missing_values",
        "params": {"strategy": "knn"}
    })
except ValueError as e:
    if "FE module not available" in str(e):
        # Fallback to basic imputation
        result = await refinery_agent.execute({
            "action": "basic_impute_missing_values",
            "params": {"strategy": "median"}
        })
```

### Context Recovery

```python
# Pipeline context is automatically recovered
result = await refinery_agent.execute({
    "action": "advanced_impute_missing_values",
    "params": {
        "data_path": "data.csv",
        "run_id": "run_123"  # Context automatically loaded
    }
})
```

## Best Practices

### 1. **Use Generic Action Names**
```python
# Good - lets system choose backend
{"action": "impute_missing_values", "params": {"strategy": "auto"}}

# Avoid - forces specific backend
{"action": "basic_impute_missing_values", "params": {"strategy": "mean"}}
```

### 2. **Leverage Auto-Routing**
```python
# Enable automatic complexity detection
AUTO_ADVANCED_ROUTING=true

# System automatically chooses best backend based on parameters
```

### 3. **Monitor Backend Usage**
```python
# Track which backend is being used
response = await refinery_agent.execute(request)
print(f"Backend used: {response.backend}")
print(f"Mode: {response.mode}")
```

### 4. **Use Pipeline Context**
```python
# Maintain context across operations
run_id = "my_pipeline_123"

# All operations share the same context
await refinery_agent.execute({
    "action": "assign_feature_roles",
    "params": {"run_id": run_id, "data_path": "data.csv"}
})

await refinery_agent.execute({
    "action": "advanced_impute_missing_values",
    "params": {"run_id": run_id, "data_path": "data.csv"}
})
```

## Migration Guide

### From Basic to Advanced

```python
# Before (basic only)
result = await refinery_agent.execute({
    "action": "basic_impute_missing_values",
    "params": {"strategy": "mean"}
})

# After (with auto-routing)
result = await refinery_agent.execute({
    "action": "impute_missing_values",  # Generic name
    "params": {"strategy": "knn"}       # Triggers advanced
})
```

### From Separate Agents

```python
# Before (separate agents)
if complex_task:
    result = await fe_agent.execute(request)
else:
    result = await refinery_agent.execute(request)

# After (unified agent)
result = await refinery_agent.execute(request)  # Automatic routing
```

## Troubleshooting

### Common Issues

1. **FE Module Not Available**
   - Check if FE module is installed
   - Verify Redis connection
   - Check environment variables

2. **Context Loss**
   - Verify Redis is running
   - Check TTL settings
   - Ensure run_id consistency

3. **Performance Issues**
   - Monitor backend usage
   - Check complexity indicators
   - Consider disabling auto-routing for simple tasks

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger("refinery_agent").setLevel(logging.DEBUG)

# Check routing decisions
logger.info(f"Routing decision: {req.backend} for action {req.action}")
```

## Future Enhancements

1. **Additional Backends**: Support for more specialized backends
2. **ML Pipeline Integration**: Direct integration with ML frameworks
3. **Advanced Monitoring**: Real-time performance analytics
4. **A/B Testing**: Compare backend performance
5. **Custom Routing Rules**: User-defined routing logic

This integration provides a powerful, flexible, and user-friendly system that automatically chooses the best tool for each task while maintaining a consistent interface. 