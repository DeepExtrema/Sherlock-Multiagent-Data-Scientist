# Refinery Agent - Universal Data Quality & Feature Engineering Service

## Overview

The Refinery Agent is a unified service that provides both **data quality validation** and **feature engineering pipeline management**. It operates in two distinct modes to ensure clear separation between read-only quality checks and transformative feature engineering operations.

## Architecture

### Dual-Mode Operation

The agent operates in two distinct modes to prevent accidental data transformations:

1. **Data Quality Mode** (`data_quality`) - Read-only operations
2. **Feature Engineering Mode** (`feature_engineering`) - Transformative operations

### Mode Validation

- Each action is automatically classified into its appropriate mode
- Mode consistency is validated before execution
- Prevents accidental transformations during quality checks

## Data Quality Actions (Read-only)

### 1. Schema Consistency Check
```json
{
  "action": "check_schema_consistency",
  "params": {
    "data_path": "/path/to/data.csv",
    "expected_schema": {
      "columns": ["col1", "col2", "col3"]
    }
  }
}
```

### 2. Missing Values Analysis
```json
{
  "action": "check_missing_values",
  "params": {
    "data_path": "/path/to/data.csv",
    "threshold": 0.5
  }
}
```

### 3. Distribution Analysis
```json
{
  "action": "check_distributions",
  "params": {
    "data_path": "/path/to/data.csv"
  }
}
```

### 4. Duplicate Detection
```json
{
  "action": "check_duplicates",
  "params": {
    "data_path": "/path/to/data.csv",
    "subset": ["col1", "col2"]
  }
}
```

### 5. Data Leakage Detection
```json
{
  "action": "check_leakage",
  "params": {
    "data_path": "/path/to/data.csv",
    "target_col": "target"
  }
}
```

### 6. Data Drift Analysis
```json
{
  "action": "check_drift",
  "params": {
    "current_data_path": "/path/to/current.csv",
    "reference_data_path": "/path/to/reference.csv"
  }
}
```

### 7. Comprehensive Quality Report
```json
{
  "action": "comprehensive_quality_report",
  "params": {
    "data_path": "/path/to/data.csv",
    "target_col": "target"
  }
}
```

## Feature Engineering Actions (Transformative)

### 1. Feature Role Assignment
```json
{
  "action": "assign_feature_roles",
  "params": {
    "data_path": "/path/to/data.csv",
    "target_col": "target",
    "run_id": "run_123",
    "session_id": "session_1"
  }
}
```

### 2. Missing Value Imputation
```json
{
  "action": "impute_missing_values",
  "params": {
    "data_path": "/path/to/data.csv",
    "run_id": "run_123",
    "strategy": "auto"
  }
}
```

### 3. Numeric Feature Scaling
```json
{
  "action": "scale_numeric_features",
  "params": {
    "run_id": "run_123",
    "method": "standard"
  }
}
```

### 4. Categorical Feature Encoding
```json
{
  "action": "encode_categorical_features",
  "params": {
    "run_id": "run_123",
    "strategy": "auto"
  }
}
```

### 5. Datetime Feature Generation
```json
{
  "action": "generate_datetime_features",
  "params": {
    "run_id": "run_123",
    "country": "US"
  }
}
```

### 6. Text Feature Vectorization
```json
{
  "action": "vectorise_text_features",
  "params": {
    "run_id": "run_123",
    "model": "mini-lm",
    "max_features": 5000
  }
}
```

### 7. Feature Interaction Generation
```json
{
  "action": "generate_interactions",
  "params": {
    "run_id": "run_123",
    "max_degree": 2
  }
}
```

### 8. Feature Selection
```json
{
  "action": "select_features",
  "params": {
    "run_id": "run_123",
    "method": "shap_top_k",
    "k": 100
  }
}
```

### 9. Pipeline Execution
```json
{
  "action": "execute_feature_pipeline",
  "params": {
    "data_path": "/path/to/data.csv",
    "run_id": "run_123"
  }
}
```

### 10. Pipeline Persistence
```json
{
  "action": "save_fe_pipeline",
  "params": {
    "run_id": "run_123"
  }
}
```

## API Endpoints

### Main Execution Endpoint
```
POST /execute
```

### Health Check
```
GET /health
```

### Metrics (Prometheus)
```
GET /metrics
```

### Pipeline Status
```
GET /pipelines
```

## Configuration

### Environment Variables
- `DRIFT_NUMERIC_P95_THRESHOLD`: Drift detection threshold (default: 0.1)
- `DRIFT_CATEGORICAL_PSI_THRESHOLD`: PSI threshold for categorical drift (default: 0.25)
- `MISSING_VALUES_THRESHOLD`: Missing values threshold (default: 0.5)
- `CORRELATION_THRESHOLD`: Correlation threshold (default: 0.95)

### Service Configuration
- **Port**: 8005
- **Health Check**: `/health`
- **Metrics**: Prometheus-compatible

## Usage Examples

### Data Quality Workflow
```python
# 1. Check data quality
quality_report = await refinery_agent.execute({
    "action": "comprehensive_quality_report",
    "params": {"data_path": "data.csv"}
})

# 2. Check for drift
drift_report = await refinery_agent.execute({
    "action": "check_drift",
    "params": {
        "current_data_path": "current.csv",
        "reference_data_path": "reference.csv"
    }
})
```

### Feature Engineering Workflow
```python
# 1. Assign feature roles
await refinery_agent.execute({
    "action": "assign_feature_roles",
    "params": {
        "data_path": "data.csv",
        "target_col": "target",
        "run_id": "run_123"
    }
})

# 2. Impute missing values
await refinery_agent.execute({
    "action": "impute_missing_values",
    "params": {
        "data_path": "data.csv",
        "run_id": "run_123",
        "strategy": "auto"
    }
})

# 3. Execute complete pipeline
pipeline_result = await refinery_agent.execute({
    "action": "execute_feature_pipeline",
    "params": {
        "data_path": "data.csv",
        "run_id": "run_123"
    }
})
```

## Safety Features

### Mode Validation
- Automatic mode detection based on action
- Validation to prevent mode-action mismatches
- Clear error messages for invalid combinations

### Pipeline Context
- Each feature engineering operation maintains pipeline context
- Tracks all transformations applied
- Enables pipeline reproducibility

### Metrics and Monitoring
- Prometheus metrics for both modes
- Request duration tracking
- Error rate monitoring
- Dataset size tracking

## Integration with Master Orchestrator

The refinery agent integrates seamlessly with the Master Orchestrator:

1. **Automatic Routing**: Orchestrator routes data quality and feature engineering tasks to refinery agent
2. **Mode Awareness**: Orchestrator understands the dual-mode operation
3. **Pipeline Management**: Orchestrator can manage complete feature engineering pipelines
4. **Quality Gates**: Orchestrator can use data quality reports as quality gates

## Best Practices

### Data Quality Checks
1. Always run comprehensive quality reports before feature engineering
2. Use quality reports as quality gates in ML pipelines
3. Monitor drift regularly in production systems

### Feature Engineering
1. Always start with feature role assignment
2. Use pipeline execution for reproducible transformations
3. Save pipelines for inference-time application
4. Monitor pipeline performance and drift

### Error Handling
1. Check response status before proceeding
2. Handle mode validation errors gracefully
3. Implement retry logic for transient failures

## Monitoring and Observability

### Metrics Available
- Request counts by action and mode
- Request duration by action and mode
- Active pipeline counts
- Dataset sizes processed
- Error rates by action

### Health Checks
- Service availability
- Mode support verification
- Pipeline context health

### Logging
- Structured logging for all operations
- Error tracking with context
- Performance monitoring
- Audit trail for transformations 