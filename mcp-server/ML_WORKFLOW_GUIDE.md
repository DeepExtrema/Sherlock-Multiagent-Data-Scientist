# ML Workflow Implementation Guide (Steps 7-10)

## Overview

This guide covers the implementation of Steps 7-10 of the ML workflow in the Deepline system. The ML agent provides comprehensive machine learning capabilities with proper separation of concerns, reproducibility, and best practices.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ML Workflow Steps 7-10                   │
├─────────────────────────────────────────────────────────────┤
│  Step 7: Class Imbalance & Sampling Strategy               │
│  ├── Quantify imbalance (prevalence, G-mean)               │
│  ├── Choose sampling strategy (SMOTE, ADASYN, etc.)        │
│  └── Create data splits (stratified, time-series, group)   │
├─────────────────────────────────────────────────────────────┤
│  Step 8: Train/Validation/Test Protocol                    │
│  ├── Cross-validation with proper splits                   │
│  ├── Model training with hyperparameter tuning             │
│  ├── Overfitting detection and prevention                  │
│  └── Comprehensive evaluation metrics                      │
├─────────────────────────────────────────────────────────────┤
│  Step 9: Baseline & Sanity Checks                          │
│  ├── Baseline models (random, majority, naïve Bayes)      │
│  ├── Leakage detection (shuffled target test)              │
│  ├── Association analysis (correlation mining)             │
│  └── Sanity checks and recommendations                     │
├─────────────────────────────────────────────────────────────┤
│  Step 10: Experiment Tracking & Reproducibility            │
│  ├── MLflow integration for experiment tracking            │
│  ├── Model registry and versioning                         │
│  ├── Artifact storage and management                       │
│  └── Reproducibility guarantees                            │
└─────────────────────────────────────────────────────────────┘
```

## Step 7: Class Imbalance & Sampling Strategy

### Purpose
Address class imbalance before any model training to ensure fair evaluation and prevent bias towards majority classes.

### Implementation

#### 1. Class Imbalance Analysis

```python
# Analyze class imbalance
response = await ml_agent.class_imbalance_endpoint({
    "task_id": "task_123",
    "data_path": "data.csv",
    "target_column": "target",
    "sampling_strategy": "none",  # Start with analysis only
    "random_state": 42,
    "test_size": 0.2,
    "cv_folds": 5
})

# Response includes:
{
    "experiment_id": "exp_task_123_1234567890",
    "imbalance_metrics": {
        "class_counts": {"0": 800, "1": 200},
        "imbalance_ratio": 4.0,
        "minority_percentage": 20.0,
        "g_mean": 0.4,
        "severity": "moderate",
        "total_samples": 1000,
        "n_classes": 2
    },
    "recommendations": [
        "Use SMOTE or random undersampling",
        "Consider balanced accuracy as metric"
    ]
}
```

#### 2. Sampling Strategy Application

```python
# Apply SMOTE for moderate imbalance
response = await ml_agent.class_imbalance_endpoint({
    "task_id": "task_123",
    "data_path": "data.csv",
    "target_column": "target",
    "sampling_strategy": "smote",  # Apply SMOTE
    "random_state": 42,
    "test_size": 0.2,
    "cv_folds": 5
})

# Available sampling strategies:
# - "none": No sampling
# - "smote": Synthetic Minority Over-sampling Technique
# - "adasyn": Adaptive Synthetic Sampling
# - "borderline_smote": Borderline SMOTE
# - "random_under": Random Under-sampling
# - "tomek_links": Tomek Links cleaning
# - "smoteenn": SMOTE + Edited Nearest Neighbors
# - "smotetomek": SMOTE + Tomek Links
```

### Best Practices

1. **Always quantify imbalance first** before applying any sampling
2. **Use stratified splits** to maintain class distribution in train/test
3. **Monitor G-mean** for imbalanced datasets
4. **Consider domain knowledge** when choosing sampling strategy
5. **Validate sampling effectiveness** by comparing metrics before/after

## Step 8: Train/Validation/Test Protocol

### Purpose
Implement robust training protocols with proper validation, overfitting detection, and comprehensive evaluation.

### Implementation

#### 1. Model Training with Cross-Validation

```python
# Train Random Forest with cross-validation
response = await ml_agent.train_validation_test_endpoint({
    "task_id": "task_123",
    "experiment_id": "exp_task_123_1234567890",
    "model_type": "random_forest",
    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "class_weight": "balanced"
    },
    "split_strategy": "stratified",
    "cv_folds": 5,
    "random_state": 42,
    "early_stopping": True,
    "max_iterations": 1000
})

# Available model types:
# - "decision_tree": Decision Tree Classifier
# - "random_forest": Random Forest Classifier
# - "gradient_boosting": Gradient Boosting Classifier
# - "knn": K-Nearest Neighbors
# - "naive_bayes": Gaussian Naive Bayes
# - "logistic_regression": Logistic Regression
# - "svm": Support Vector Machine
```

#### 2. Comprehensive Evaluation Metrics

```python
# Response includes comprehensive metrics:
{
    "model_type": "random_forest",
    "metrics": {
        "accuracy": 0.85,
        "precision": 0.83,
        "recall": 0.87,
        "f1_score": 0.85,
        "cohen_kappa": 0.70,
        "matthews_corrcoef": 0.71,
        "log_loss": 0.45,
        "roc_auc": 0.92,
        "average_precision": 0.89,
        "confusion_matrix": [[150, 20], [15, 115]],
        "classification_report": {...}
    },
    "cv_results": {
        "scores": [0.83, 0.87, 0.85, 0.86, 0.84],
        "mean": 0.85,
        "std": 0.015
    },
    "training_time": 2.34
}
```

#### 3. Overfitting Detection

```python
# Monitor train vs validation gap
# The system automatically tracks:
# - Cross-validation score variance
# - Training time patterns
# - Model complexity vs performance

# Overfitting indicators:
# - High CV variance (> 0.05)
# - Large gap between train and CV scores
# - Performance degradation with more complex models
```

### Best Practices

1. **Use stratified cross-validation** for imbalanced datasets
2. **Monitor CV score variance** to detect instability
3. **Implement early stopping** for iterative algorithms
4. **Track multiple metrics** (accuracy, precision, recall, F1)
5. **Use class weights** for imbalanced problems
6. **Validate hyperparameters** on validation set only

## Step 9: Baseline & Sanity Checks

### Purpose
Establish baseline performance and detect data quality issues before complex modeling.

### Implementation

#### 1. Baseline Models

```python
# Run comprehensive baseline checks
response = await ml_agent.baseline_sanity_endpoint({
    "task_id": "task_123",
    "experiment_id": "exp_task_123_1234567890",
    "baseline_models": [
        "baseline_random",
        "baseline_majority", 
        "naive_bayes",
        "decision_tree"
    ],
    "leakage_test": True,
    "association_analysis": True,
    "max_rules": 10,
    "min_confidence": 0.5
})

# Response includes:
{
    "baseline_results": {
        "baseline_random": {"accuracy": 0.50, "f1_score": 0.33},
        "baseline_majority": {"accuracy": 0.80, "f1_score": 0.00},
        "naive_bayes": {"accuracy": 0.75, "f1_score": 0.60},
        "decision_tree": {"accuracy": 0.78, "f1_score": 0.65}
    },
    "sanity_checks": {
        "leakage_test": {
            "shuffled_accuracy": 0.52,
            "leakage_detected": False,
            "recommendation": "No obvious leakage"
        },
        "association_analysis": {
            "top_correlations": {"feature_1": 0.85, "feature_2": 0.72},
            "high_correlation_features": ["feature_1"],
            "recommendation": "Review high-correlation features for potential leakage"
        }
    }
}
```

#### 2. Leakage Detection

```python
# The system automatically:
# 1. Shuffles target variable
# 2. Trains simple model on shuffled data
# 3. Evaluates on original test set
# 4. Flags suspicious performance (> 70% accuracy)

# Leakage indicators:
# - High accuracy on shuffled data
# - Features with perfect correlation to target
# - Temporal leakage (future information in features)
```

#### 3. Association Analysis

```python
# Correlation-based association mining:
# - Identifies high-correlation features
# - Flags potential leakage sources
# - Suggests feature engineering opportunities

# High correlation thresholds:
# - > 0.8: Potential leakage
# - > 0.6: Strong association
# - > 0.4: Moderate association
```

### Best Practices

1. **Always run baseline models** before complex modeling
2. **Use multiple baseline types** (random, majority, simple ML)
3. **Monitor for suspicious performance** in baseline models
4. **Investigate high correlations** with target variable
5. **Document baseline performance** for comparison
6. **Use baseline as sanity check** for model improvements

## Step 10: Experiment Tracking & Reproducibility

### Purpose
Ensure complete reproducibility and track all experiments for comparison and audit trails.

### Implementation

#### 1. Experiment Tracking

```python
# Complete experiment tracking
response = await ml_agent.experiment_tracking_endpoint({
    "task_id": "task_123",
    "experiment_id": "exp_task_123_1234567890",
    "experiment_name": "customer_churn_prediction",
    "tags": {
        "dataset": "customer_data_v2",
        "business_unit": "marketing",
        "priority": "high"
    },
    "artifact_path": "./artifacts/exp_task_123_1234567890",
    "model_registry": True
})

# Response includes:
{
    "experiment_summary": {
        "experiment_id": "exp_task_123_1234567890",
        "experiment_name": "customer_churn_prediction",
        "experiment_hash": "a1b2c3d4e5f6...",
        "steps_completed": ["class_imbalance", "train_validation_test", "baseline_sanity"],
        "models_trained": ["random_forest", "gradient_boosting"],
        "best_model": "random_forest",
        "best_metric": 0.85
    },
    "artifacts": {
        "context": "./artifacts/exp_task_123_1234567890/context.json",
        "model_random_forest": "./artifacts/exp_task_123_1234567890/models/random_forest.pkl"
    },
    "registry_info": {
        "registered_model": "customer_churn_prediction_random_forest",
        "model_version": "1.0"
    }
}
```

#### 2. MLflow Integration

```python
# Automatic MLflow logging includes:
# - All hyperparameters
# - Training metrics
# - Validation metrics
# - Model artifacts
# - Confusion matrices
# - Feature importance plots

# MLflow tracking URI: http://localhost:5000
# Experiment name: deepline-ml-workflow
```

#### 3. Reproducibility Features

```python
# Reproducibility guarantees:
# - Experiment hash for exact reproduction
# - Complete context serialization
# - Model versioning
# - Artifact storage
# - Environment tracking

# To reproduce an experiment:
# 1. Use experiment_hash
# 2. Load saved context
# 3. Re-run with same parameters
# 4. Verify identical results
```

### Best Practices

1. **Always track experiments** before training
2. **Use descriptive experiment names** and tags
3. **Store all artifacts** (models, data, plots)
4. **Register best models** in model registry
5. **Document experiment context** completely
6. **Use experiment hashes** for reproducibility
7. **Monitor MLflow UI** for experiment comparison

## Complete Workflow Example

### End-to-End ML Pipeline

```python
# Step 1: Class Imbalance Analysis
imbalance_response = await ml_agent.class_imbalance_endpoint({
    "task_id": "churn_prediction_001",
    "data_path": "customer_data.csv",
    "target_column": "churned",
    "sampling_strategy": "smote",
    "random_state": 42
})

experiment_id = imbalance_response["result"]["experiment_id"]

# Step 2: Train Multiple Models
models = ["random_forest", "gradient_boosting", "logistic_regression"]
training_results = []

for model in models:
    result = await ml_agent.train_validation_test_endpoint({
        "task_id": "churn_prediction_001",
        "experiment_id": experiment_id,
        "model_type": model,
        "hyperparameters": get_default_hyperparameters(model),
        "split_strategy": "stratified",
        "cv_folds": 5
    })
    training_results.append(result)

# Step 3: Baseline and Sanity Checks
sanity_response = await ml_agent.baseline_sanity_endpoint({
    "task_id": "churn_prediction_001",
    "experiment_id": experiment_id,
    "baseline_models": ["baseline_random", "naive_bayes"],
    "leakage_test": True,
    "association_analysis": True
})

# Step 4: Experiment Tracking
tracking_response = await ml_agent.experiment_tracking_endpoint({
    "task_id": "churn_prediction_001",
    "experiment_id": experiment_id,
    "experiment_name": "customer_churn_prediction_v1",
    "tags": {
        "business_unit": "marketing",
        "model_type": "classification",
        "priority": "high"
    },
    "artifact_path": f"./artifacts/{experiment_id}",
    "model_registry": True
})

# Get best model for deployment
best_model = tracking_response["result"]["experiment_summary"]["best_model"]
print(f"Best model: {best_model}")
```

## Monitoring and Observability

### Prometheus Metrics

```python
# Available metrics:
# - ml_requests_total: Request counts by action and status
# - ml_request_duration_seconds: Request duration by action
# - ml_active_experiments: Number of active experiments
# - ml_model_training_seconds: Training time by algorithm

# Example queries:
# - Training time by model type
# - Success rate by workflow step
# - Active experiments over time
```

### Health Checks

```bash
# Check ML agent health
curl http://localhost:8002/health

# Response:
{
    "status": "ok",
    "version": "1.0.0",
    "mlflow_available": true,
    "imbalanced_available": true,
    "active_experiments": 5
}
```

### Experiment Management

```bash
# List active experiments
curl http://localhost:8002/experiments

# Response:
{
    "active_experiments": 5,
    "experiments": ["exp_001", "exp_002", "exp_003"],
    "experiment_summaries": {
        "exp_001": {
            "experiment_name": "customer_churn_v1",
            "best_model": "random_forest",
            "best_metric": 0.85
        }
    }
}
```

## Configuration

### Environment Variables

```bash
# ML Agent Configuration
REDIS_URL=redis://localhost:6379
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=deepline-ml-workflow

# Optional: Custom artifact paths
MLFLOW_ARTIFACT_ROOT=./mlflow_artifacts
```

### Dependencies

```python
# Required packages:
# - scikit-learn: Core ML algorithms
# - imbalanced-learn: Sampling strategies
# - mlflow: Experiment tracking
# - pandas, numpy: Data manipulation
# - matplotlib, seaborn: Visualization
```

## Troubleshooting

### Common Issues

1. **MLflow Connection Failed**
   - Check MLflow server is running
   - Verify tracking URI configuration
   - Check network connectivity

2. **Imbalanced-learn Not Available**
   - Install: `pip install imbalanced-learn`
   - Check import statements
   - Verify version compatibility

3. **Memory Issues with Large Datasets**
   - Use data sampling for initial experiments
   - Implement chunked processing
   - Monitor memory usage

4. **Model Training Failures**
   - Check data quality and preprocessing
   - Verify hyperparameter ranges
   - Monitor for convergence issues

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger("ml_agent").setLevel(logging.DEBUG)

# Check experiment context
context = get_experiment_context(experiment_id)
print(f"Experiment steps: {context['steps']}")
print(f"Models trained: {list(context['models'].keys())}")
```

## Future Enhancements

1. **Advanced Hyperparameter Tuning**
   - Bayesian optimization
   - Multi-objective optimization
   - Automated hyperparameter search

2. **Model Interpretability**
   - SHAP values
   - Feature importance analysis
   - Model explanation tools

3. **Advanced Sampling**
   - Cost-sensitive learning
   - Active learning
   - Semi-supervised approaches

4. **Production Deployment**
   - Model serving endpoints
   - A/B testing framework
   - Model monitoring and drift detection

This comprehensive ML workflow implementation ensures robust, reproducible, and well-documented machine learning experiments while maintaining best practices throughout the entire pipeline. 