#!/usr/bin/env python3
"""
ML Agent - Comprehensive Machine Learning Workflow Service
Implements Steps 7-10 of the ML workflow with class imbalance, training protocols, 
baseline models, and experiment tracking.
"""

import asyncio
import json
import logging
import time
import os
import pickle
import hashlib
from typing import Dict, Any, Optional, List, Tuple, Union, Literal
from pathlib import Path
from enum import Enum
from datetime import datetime

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, multiprocess
from fastapi.responses import Response

# ML Libraries
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, TimeSeriesSplit, 
    cross_val_score, GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    cohen_kappa_score, matthews_corrcoef
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Imbalanced Learning
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.metrics import geometric_mean_score
    IMBALANCED_AVAILABLE = True
except ImportError:
    IMBALANCED_AVAILABLE = False
    logging.warning("Imbalanced-learn not available - sampling methods disabled")

# Experiment Tracking
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available - experiment tracking disabled")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ML Agent",
    description="Comprehensive Machine Learning Workflow Service (Steps 7-10)",
    version="1.0.0"
)

# Prometheus metrics
REQUEST_COUNT = Counter('ml_requests_total', 'Total requests', ['action', 'status'])
REQUEST_DURATION = Histogram('ml_request_duration_seconds', 'Request duration', ['action'])
ACTIVE_EXPERIMENTS = Gauge('ml_active_experiments', 'Number of active experiments')
MODEL_TRAINING_TIME = Histogram('ml_model_training_seconds', 'Model training time', ['algorithm'])

# Configuration from environment
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'deepline-ml-workflow')

# Initialize MLflow if available
if MLFLOW_AVAILABLE:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            mlflow.create_experiment(EXPERIMENT_NAME)
        mlflow.set_experiment(EXPERIMENT_NAME)
    except Exception as e:
        logger.warning(f"Failed to initialize MLflow: {e}")

# In-memory experiment cache
_EXPERIMENTS: Dict[str, Dict[str, Any]] = {}

# Step definitions
class MLStep(Enum):
    """ML workflow steps."""
    CLASS_IMBALANCE = "class_imbalance"           # Step 7
    TRAIN_VALIDATION_TEST = "train_validation_test"  # Step 8
    BASELINE_SANITY = "baseline_sanity"           # Step 9
    EXPERIMENT_TRACKING = "experiment_tracking"   # Step 10

# Sampling strategies
class SamplingStrategy(Enum):
    """Class imbalance sampling strategies."""
    NONE = "none"
    SMOTE = "smote"
    ADASYN = "adasyn"
    BORDERLINE_SMOTE = "borderline_smote"
    RANDOM_UNDER = "random_under"
    TOMEK_LINKS = "tomek_links"
    SMOTEENN = "smoteenn"
    SMOTETOMEK = "smotetomek"
    WEIGHTED_LOSS = "weighted_loss"

# Model types
class ModelType(Enum):
    """Available model types."""
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    KNN = "knn"
    NAIVE_BAYES = "naive_bayes"
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"
    BASELINE_RANDOM = "baseline_random"
    BASELINE_MAJORITY = "baseline_majority"

# Split strategies
class SplitStrategy(Enum):
    """Data splitting strategies."""
    RANDOM = "random"
    STRATIFIED = "stratified"
    TIME_SERIES = "time_series"
    GROUP = "group"

# Pydantic models
class ClassImbalanceRequest(BaseModel):
    """Step 7: Class imbalance analysis and sampling strategy."""
    task_id: str
    data_path: str
    target_column: str
    sampling_strategy: Optional[SamplingStrategy] = SamplingStrategy.NONE
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    
    @validator('test_size')
    def validate_test_size(cls, v):
        if not 0.1 <= v <= 0.5:
            raise ValueError('test_size must be between 0.1 and 0.5')
        return v

class TrainingRequest(BaseModel):
    """Step 8: Training/validation/test protocol."""
    task_id: str
    experiment_id: str
    model_type: ModelType
    hyperparameters: Optional[Dict[str, Any]] = None
    split_strategy: SplitStrategy = SplitStrategy.STRATIFIED
    cv_folds: int = 5
    random_state: int = 42
    early_stopping: bool = True
    max_iterations: int = 1000
    
    @validator('cv_folds')
    def validate_cv_folds(cls, v):
        if not 2 <= v <= 10:
            raise ValueError('cv_folds must be between 2 and 10')
        return v

class BaselineRequest(BaseModel):
    """Step 9: Baseline and sanity checks."""
    task_id: str
    experiment_id: str
    baseline_models: List[ModelType] = Field(default_factory=lambda: [
        ModelType.BASELINE_RANDOM, 
        ModelType.BASELINE_MAJORITY,
        ModelType.NAIVE_BAYES,
        ModelType.DECISION_TREE
    ])
    leakage_test: bool = True
    association_analysis: bool = True
    max_rules: int = 10
    min_confidence: float = 0.5

class ExperimentTrackingRequest(BaseModel):
    """Step 10: Experiment tracking and reproducibility."""
    task_id: str
    experiment_id: str
    experiment_name: str
    tags: Optional[Dict[str, str]] = None
    artifact_path: Optional[str] = None
    model_registry: bool = True

class MLResponse(BaseModel):
    """Standard ML response."""
    task_id: str
    step: MLStep
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float
    timestamp: float

# Utility functions
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from various file formats."""
    path = Path(file_path)
    if path.suffix == '.parquet':
        return pd.read_parquet(file_path)
    elif path.suffix == '.csv':
        return pd.read_csv(file_path)
    elif path.suffix == '.json':
        return pd.read_json(file_path)
    elif path.suffix in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

def get_experiment_context(experiment_id: str) -> Dict[str, Any]:
    """Get or create experiment context."""
    if experiment_id not in _EXPERIMENTS:
        _EXPERIMENTS[experiment_id] = {
            "created_at": time.time(),
            "steps": [],
            "data_splits": {},
            "models": {},
            "metrics": {},
            "artifacts": {}
        }
        ACTIVE_EXPERIMENTS.inc()
    return _EXPERIMENTS[experiment_id]

def calculate_class_imbalance_metrics(y: pd.Series) -> Dict[str, Any]:
    """Calculate comprehensive class imbalance metrics."""
    class_counts = y.value_counts()
    total_samples = len(y)
    minority_class = class_counts.min()
    majority_class = class_counts.max()
    
    # Basic metrics
    imbalance_ratio = majority_class / minority_class
    minority_percentage = (minority_class / total_samples) * 100
    
    # G-mean calculation
    if IMBALANCED_AVAILABLE:
        g_mean = geometric_mean_score(y, y)  # Perfect prediction for calculation
    else:
        g_mean = np.sqrt((minority_class / total_samples) * (majority_class / total_samples))
    
    # Determine imbalance severity
    if imbalance_ratio > 100:
        severity = "extreme"
    elif imbalance_ratio > 20:
        severity = "severe"
    elif imbalance_ratio > 10:
        severity = "moderate"
    elif imbalance_ratio > 3:
        severity = "mild"
    else:
        severity = "balanced"
    
    return {
        "class_counts": class_counts.to_dict(),
        "imbalance_ratio": float(imbalance_ratio),
        "minority_percentage": float(minority_percentage),
        "g_mean": float(g_mean),
        "severity": severity,
        "total_samples": total_samples,
        "n_classes": len(class_counts)
    }

def apply_sampling_strategy(X: pd.DataFrame, y: pd.Series, strategy: SamplingStrategy, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply sampling strategy to address class imbalance."""
    if not IMBALANCED_AVAILABLE:
        raise ValueError("Imbalanced-learn not available for sampling")
    
    if strategy == SamplingStrategy.NONE:
        return X, y
    elif strategy == SamplingStrategy.SMOTE:
        sampler = SMOTE(random_state=random_state)
    elif strategy == SamplingStrategy.ADASYN:
        sampler = ADASYN(random_state=random_state)
    elif strategy == SamplingStrategy.BORDERLINE_SMOTE:
        sampler = BorderlineSMOTE(random_state=random_state)
    elif strategy == SamplingStrategy.RANDOM_UNDER:
        sampler = RandomUnderSampler(random_state=random_state)
    elif strategy == SamplingStrategy.TOMEK_LINKS:
        sampler = TomekLinks()
    elif strategy == SamplingStrategy.SMOTEENN:
        sampler = SMOTEENN(random_state=random_state)
    elif strategy == SamplingStrategy.SMOTETOMEK:
        sampler = SMOTETomek(random_state=random_state)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)

def create_model(model_type: ModelType, hyperparameters: Optional[Dict[str, Any]] = None) -> Any:
    """Create model instance based on type."""
    if hyperparameters is None:
        hyperparameters = {}
    
    if model_type == ModelType.DECISION_TREE:
        return DecisionTreeClassifier(random_state=42, **hyperparameters)
    elif model_type == ModelType.RANDOM_FOREST:
        return RandomForestClassifier(random_state=42, **hyperparameters)
    elif model_type == ModelType.GRADIENT_BOOSTING:
        return GradientBoostingClassifier(random_state=42, **hyperparameters)
    elif model_type == ModelType.KNN:
        return KNeighborsClassifier(**hyperparameters)
    elif model_type == ModelType.NAIVE_BAYES:
        return GaussianNB(**hyperparameters)
    elif model_type == ModelType.LOGISTIC_REGRESSION:
        return LogisticRegression(random_state=42, **hyperparameters)
    elif model_type == ModelType.SVM:
        return SVC(random_state=42, probability=True, **hyperparameters)
    elif model_type == ModelType.BASELINE_RANDOM:
        return RandomBaselineClassifier()
    elif model_type == ModelType.BASELINE_MAJORITY:
        return MajorityBaselineClassifier()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def create_data_splits(X: pd.DataFrame, y: pd.Series, strategy: SplitStrategy, 
                      test_size: float = 0.2, cv_folds: int = 5, 
                      random_state: int = 42, group_column: Optional[str] = None) -> Dict[str, Any]:
    """Create data splits based on strategy."""
    if strategy == SplitStrategy.RANDOM:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
    elif strategy == SplitStrategy.STRATIFIED:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
    elif strategy == SplitStrategy.TIME_SERIES:
        # For time series, we don't shuffle
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        cv = TimeSeriesSplit(n_splits=cv_folds)
        
    elif strategy == SplitStrategy.GROUP:
        if group_column is None:
            raise ValueError("group_column required for group split strategy")
        # Group-based split (simplified implementation)
        groups = X[group_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "cv": cv,
        "strategy": strategy.value
    }

def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics."""
    metrics = {}
    
    # Basic classification metrics
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, average='weighted'))
    metrics["recall"] = float(recall_score(y_true, y_pred, average='weighted'))
    metrics["f1_score"] = float(f1_score(y_true, y_pred, average='weighted'))
    metrics["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))
    metrics["matthews_corrcoef"] = float(matthews_corrcoef(y_true, y_pred))
    
    # Probability-based metrics
    if y_prob is not None:
        metrics["log_loss"] = float(log_loss(y_true, y_prob))
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob, average='weighted'))
        metrics["average_precision"] = float(average_precision_score(y_true, y_prob, average='weighted'))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics["classification_report"] = report
    
    return metrics

# Baseline classifier implementations
class RandomBaselineClassifier:
    """Random baseline classifier for sanity checks."""
    
    def __init__(self):
        self.classes_ = None
        self.n_classes_ = 0
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        return self
        
    def predict(self, X):
        return np.random.choice(self.classes_, size=len(X))
        
    def predict_proba(self, X):
        probs = np.random.dirichlet(np.ones(self.n_classes_), size=len(X))
        return probs

class MajorityBaselineClassifier:
    """Majority class baseline classifier."""
    
    def __init__(self):
        self.majority_class_ = None
        
    def fit(self, X, y):
        from collections import Counter
        self.majority_class_ = Counter(y).most_common(1)[0][0]
        return self
        
    def predict(self, X):
        return np.full(len(X), self.majority_class_)
        
    def predict_proba(self, X):
        n_samples = len(X)
        probs = np.zeros((n_samples, 2))  # Assuming binary classification
        probs[:, 1] = 1.0  # All probability to majority class
        return probs

# Step 7: Class Imbalance & Sampling Strategy
async def handle_class_imbalance(req: ClassImbalanceRequest) -> Dict[str, Any]:
    """Step 7: Analyze class imbalance and apply sampling strategy."""
    start_time = time.time()
    
    # Load data
    df = load_data(req.data_path)
    
    if req.target_column not in df.columns:
        raise ValueError(f"Target column '{req.target_column}' not found in data")
    
    # Separate features and target
    X = df.drop(columns=[req.target_column])
    y = df[req.target_column]
    
    # Calculate imbalance metrics
    imbalance_metrics = calculate_class_imbalance_metrics(y)
    
    # Apply sampling strategy if specified
    sampling_info = {"strategy": req.sampling_strategy.value, "applied": False}
    if req.sampling_strategy != SamplingStrategy.NONE:
        if not IMBALANCED_AVAILABLE:
            raise ValueError("Imbalanced-learn not available for sampling")
        
        X_resampled, y_resampled = apply_sampling_strategy(
            X, y, req.sampling_strategy, req.random_state
        )
        sampling_info["applied"] = True
        sampling_info["original_shape"] = X.shape
        sampling_info["resampled_shape"] = X_resampled.shape
        
        # Recalculate metrics after sampling
        resampled_metrics = calculate_class_imbalance_metrics(y_resampled)
        sampling_info["resampled_metrics"] = resampled_metrics
    else:
        X_resampled, y_resampled = X, y
    
    # Create data splits
    splits = create_data_splits(
        X_resampled, y_resampled, 
        SplitStrategy.STRATIFIED, 
        req.test_size, req.cv_folds, req.random_state
    )
    
    # Store in experiment context
    experiment_id = f"exp_{req.task_id}_{int(time.time())}"
    context = get_experiment_context(experiment_id)
    context["data_splits"] = splits
    context["imbalance_metrics"] = imbalance_metrics
    context["sampling_info"] = sampling_info
    context["steps"].append("class_imbalance")
    
    execution_time = time.time() - start_time
    
    return {
        "experiment_id": experiment_id,
        "imbalance_metrics": imbalance_metrics,
        "sampling_info": sampling_info,
        "data_splits": {
            "train_shape": splits["X_train"].shape,
            "test_shape": splits["X_test"].shape,
            "strategy": splits["strategy"]
        },
        "recommendations": generate_imbalance_recommendations(imbalance_metrics)
    }

def generate_imbalance_recommendations(metrics: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on imbalance metrics."""
    recommendations = []
    
    severity = metrics["severity"]
    imbalance_ratio = metrics["imbalance_ratio"]
    
    if severity == "extreme":
        recommendations.extend([
            "Consider collecting more minority class samples",
            "Use SMOTE or ADASYN for oversampling",
            "Implement weighted loss functions",
            "Consider anomaly detection approaches"
        ])
    elif severity == "severe":
        recommendations.extend([
            "Use SMOTE or BorderlineSMOTE",
            "Consider ensemble methods",
            "Implement class weights in models"
        ])
    elif severity == "moderate":
        recommendations.extend([
            "Use SMOTE or random undersampling",
            "Consider balanced accuracy as metric"
        ])
    elif severity == "mild":
        recommendations.extend([
            "Standard approaches should work well",
            "Monitor for bias in predictions"
        ])
    
    return recommendations

# Step 8: Train/Validation/Test Protocol
async def train_validation_test(req: TrainingRequest) -> Dict[str, Any]:
    """Step 8: Comprehensive training, validation, and testing protocol."""
    start_time = time.time()
    
    # Get experiment context
    context = get_experiment_context(req.experiment_id)
    if "data_splits" not in context:
        raise ValueError("Data splits not found. Run class imbalance step first.")
    
    splits = context["data_splits"]
    X_train, X_test = splits["X_train"], splits["X_test"]
    y_train, y_test = splits["y_train"], splits["y_test"]
    cv = splits["cv"]
    
    # Create model
    model = create_model(req.model_type, req.hyperparameters)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    # Train final model
    model_start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - model_start_time
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(y_test.values, y_pred, y_prob)
    
    # Add cross-validation results
    metrics["cv_scores"] = cv_scores.tolist()
    metrics["cv_mean"] = float(cv_scores.mean())
    metrics["cv_std"] = float(cv_scores.std())
    metrics["training_time"] = training_time
    
    # Store model and metrics
    context["models"][req.model_type.value] = {
        "model": model,
        "hyperparameters": req.hyperparameters,
        "metrics": metrics,
        "training_time": training_time
    }
    context["steps"].append("train_validation_test")
    
    # Log to MLflow if available
    if MLFLOW_AVAILABLE:
        try:
            with mlflow.start_run(run_name=f"{req.model_type.value}_{req.experiment_id}"):
                mlflow.log_params(req.hyperparameters or {})
                mlflow.log_metrics(metrics)
                mlflow.log_metric("training_time", training_time)
                mlflow.log_metric("cv_mean", metrics["cv_mean"])
                mlflow.log_metric("cv_std", metrics["cv_std"])
                
                # Log model
                mlflow.sklearn.log_model(model, "model")
                
                # Log confusion matrix as artifact
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(metrics["confusion_matrix"], annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix - {req.model_type.value}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                
                confusion_matrix_path = f"confusion_matrix_{req.model_type.value}.png"
                plt.savefig(confusion_matrix_path)
                mlflow.log_artifact(confusion_matrix_path)
                plt.close()
                
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")
    
    execution_time = time.time() - start_time
    
    return {
        "model_type": req.model_type.value,
        "metrics": metrics,
        "training_time": training_time,
        "cv_results": {
            "scores": cv_scores.tolist(),
            "mean": float(cv_scores.mean()),
            "std": float(cv_scores.std())
        },
        "model_info": {
            "hyperparameters": req.hyperparameters,
            "features": list(X_train.columns),
            "n_features": len(X_train.columns)
        }
    }

# Step 9: Baseline & Sanity Checks
async def baseline_sanity_checks(req: BaselineRequest) -> Dict[str, Any]:
    """Step 9: Run baseline models and sanity checks."""
    start_time = time.time()
    
    # Get experiment context
    context = get_experiment_context(req.experiment_id)
    if "data_splits" not in context:
        raise ValueError("Data splits not found. Run training step first.")
    
    splits = context["data_splits"]
    X_train, X_test = splits["X_train"], splits["X_test"]
    y_train, y_test = splits["y_train"], splits["y_test"]
    
    baseline_results = {}
    sanity_checks = {}
    
    # Run baseline models
    for model_type in req.baseline_models:
        try:
            model = create_model(model_type)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            metrics = calculate_comprehensive_metrics(y_test.values, y_pred, y_prob)
            baseline_results[model_type.value] = metrics
            
        except Exception as e:
            baseline_results[model_type.value] = {"error": str(e)}
    
    # Leakage test
    if req.leakage_test:
        # Shuffle target and retrain
        y_train_shuffled = y_train.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Train a simple model on shuffled data
        leakage_model = DecisionTreeClassifier(random_state=42, max_depth=3)
        leakage_model.fit(X_train, y_train_shuffled)
        y_pred_shuffled = leakage_model.predict(X_test)
        
        shuffled_accuracy = accuracy_score(y_test, y_pred_shuffled)
        sanity_checks["leakage_test"] = {
            "shuffled_accuracy": float(shuffled_accuracy),
            "leakage_detected": shuffled_accuracy > 0.7,  # Suspicious if > 70%
            "recommendation": "Investigate data leakage" if shuffled_accuracy > 0.7 else "No obvious leakage"
        }
    
    # Association analysis (simplified)
    if req.association_analysis:
        try:
            # Simple correlation-based association
            correlations = X_train.corrwith(y_train).abs().sort_values(ascending=False)
            top_correlations = correlations.head(req.max_rules)
            
            sanity_checks["association_analysis"] = {
                "top_correlations": top_correlations.to_dict(),
                "high_correlation_features": list(top_correlations[top_correlations > req.min_confidence].index),
                "recommendation": "Review high-correlation features for potential leakage"
            }
        except Exception as e:
            sanity_checks["association_analysis"] = {"error": str(e)}
    
    # Store results
    context["baseline_results"] = baseline_results
    context["sanity_checks"] = sanity_checks
    context["steps"].append("baseline_sanity")
    
    execution_time = time.time() - start_time
    
    return {
        "baseline_results": baseline_results,
        "sanity_checks": sanity_checks,
        "recommendations": generate_sanity_recommendations(baseline_results, sanity_checks)
    }

def generate_sanity_recommendations(baseline_results: Dict[str, Any], 
                                  sanity_checks: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on sanity check results."""
    recommendations = []
    
    # Check baseline performance
    for model_name, results in baseline_results.items():
        if "error" not in results:
            accuracy = results.get("accuracy", 0)
            if accuracy < 0.5:
                recommendations.append(f"{model_name} accuracy is very low - check data quality")
            elif accuracy > 0.95:
                recommendations.append(f"{model_name} accuracy is suspiciously high - potential overfitting or leakage")
    
    # Check leakage test
    if "leakage_test" in sanity_checks:
        leakage_test = sanity_checks["leakage_test"]
        if leakage_test.get("leakage_detected", False):
            recommendations.append("Data leakage detected - investigate feature engineering")
    
    # Check associations
    if "association_analysis" in sanity_checks:
        assoc_analysis = sanity_checks["association_analysis"]
        high_corr_features = assoc_analysis.get("high_correlation_features", [])
        if len(high_corr_features) > 0:
            recommendations.append(f"High correlation features detected: {high_corr_features}")
    
    return recommendations

# Step 10: Experiment Tracking & Reproducibility
async def experiment_tracking(req: ExperimentTrackingRequest) -> Dict[str, Any]:
    """Step 10: Comprehensive experiment tracking and reproducibility."""
    start_time = time.time()
    
    # Get experiment context
    context = get_experiment_context(req.experiment_id)
    
    # Generate experiment hash for reproducibility
    experiment_hash = hashlib.md5(
        json.dumps(context, sort_keys=True, default=str).encode()
    ).hexdigest()
    
    # Create experiment summary
    experiment_summary = {
        "experiment_id": req.experiment_id,
        "experiment_name": req.experiment_name,
        "experiment_hash": experiment_hash,
        "created_at": context["created_at"],
        "steps_completed": context["steps"],
        "models_trained": list(context["models"].keys()),
        "total_training_time": sum(
            model_info["training_time"] 
            for model_info in context["models"].values()
        ),
        "best_model": None,
        "best_metric": None
    }
    
    # Find best model
    if context["models"]:
        best_model = None
        best_score = -1
        
        for model_name, model_info in context["models"].items():
            if "error" not in model_info["metrics"]:
                score = model_info["metrics"].get("cv_mean", 0)
                if score > best_score:
                    best_score = score
                    best_model = model_name
        
        experiment_summary["best_model"] = best_model
        experiment_summary["best_metric"] = best_score
    
    # Save experiment artifacts
    artifacts = {}
    if req.artifact_path:
        artifact_dir = Path(req.artifact_path)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Save experiment context
        context_path = artifact_dir / f"{req.experiment_id}_context.json"
        with open(context_path, 'w') as f:
            json.dump(context, f, default=str, indent=2)
        artifacts["context"] = str(context_path)
        
        # Save models
        models_dir = artifact_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        for model_name, model_info in context["models"].items():
            model_path = models_dir / f"{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model_info["model"], f)
            artifacts[f"model_{model_name}"] = str(model_path)
    
    # Register in model registry if requested
    registry_info = {}
    if req.model_registry and MLFLOW_AVAILABLE and context["models"]:
        try:
            best_model_name = experiment_summary["best_model"]
            if best_model_name:
                best_model_info = context["models"][best_model_name]
                
                with mlflow.start_run(run_name=f"registry_{req.experiment_id}"):
                    mlflow.log_params({
                        "experiment_id": req.experiment_id,
                        "experiment_hash": experiment_hash,
                        "best_model": best_model_name
                    })
                    
                    # Log the best model
                    mlflow.sklearn.log_model(
                        best_model_info["model"], 
                        "best_model",
                        registered_model_name=f"{req.experiment_name}_{best_model_name}"
                    )
                    
                    registry_info = {
                        "registered_model": f"{req.experiment_name}_{best_model_name}",
                        "model_version": "1.0"
                    }
        except Exception as e:
            logger.warning(f"Failed to register model: {e}")
    
    # Store final experiment info
    context["experiment_summary"] = experiment_summary
    context["artifacts"] = artifacts
    context["registry_info"] = registry_info
    context["steps"].append("experiment_tracking")
    
    execution_time = time.time() - start_time
    
    return {
        "experiment_summary": experiment_summary,
        "artifacts": artifacts,
        "registry_info": registry_info,
        "reproducibility": {
            "experiment_hash": experiment_hash,
            "timestamp": datetime.now().isoformat(),
            "steps_completed": context["steps"]
        }
    }

# API endpoints
@app.post("/class_imbalance", response_model=MLResponse)
async def class_imbalance_endpoint(req: ClassImbalanceRequest):
    """Step 7: Class imbalance analysis and sampling strategy."""
    start_time = time.time()
    
    try:
        result = await handle_class_imbalance(req)
        
        REQUEST_COUNT.labels(action="class_imbalance", status="success").inc()
        REQUEST_DURATION.labels(action="class_imbalance").observe(time.time() - start_time)
        
        return MLResponse(
            task_id=req.task_id,
            step=MLStep.CLASS_IMBALANCE,
            success=True,
            result=result,
            execution_time=time.time() - start_time,
            timestamp=time.time()
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(action="class_imbalance", status="error").inc()
        REQUEST_DURATION.labels(action="class_imbalance").observe(time.time() - start_time)
        
        return MLResponse(
            task_id=req.task_id,
            step=MLStep.CLASS_IMBALANCE,
            success=False,
            error=str(e),
            execution_time=time.time() - start_time,
            timestamp=time.time()
        )

@app.post("/train_validation_test", response_model=MLResponse)
async def train_validation_test_endpoint(req: TrainingRequest):
    """Step 8: Training/validation/test protocol."""
    start_time = time.time()
    
    try:
        result = await train_validation_test(req)
        
        REQUEST_COUNT.labels(action="train_validation_test", status="success").inc()
        REQUEST_DURATION.labels(action="train_validation_test").observe(time.time() - start_time)
        MODEL_TRAINING_TIME.labels(algorithm=req.model_type.value).observe(result["training_time"])
        
        return MLResponse(
            task_id=req.task_id,
            step=MLStep.TRAIN_VALIDATION_TEST,
            success=True,
            result=result,
            execution_time=time.time() - start_time,
            timestamp=time.time()
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(action="train_validation_test", status="error").inc()
        REQUEST_DURATION.labels(action="train_validation_test").observe(time.time() - start_time)
        
        return MLResponse(
            task_id=req.task_id,
            step=MLStep.TRAIN_VALIDATION_TEST,
            success=False,
            error=str(e),
            execution_time=time.time() - start_time,
            timestamp=time.time()
        )

@app.post("/baseline_sanity", response_model=MLResponse)
async def baseline_sanity_endpoint(req: BaselineRequest):
    """Step 9: Baseline and sanity checks."""
    start_time = time.time()
    
    try:
        result = await baseline_sanity_checks(req)
        
        REQUEST_COUNT.labels(action="baseline_sanity", status="success").inc()
        REQUEST_DURATION.labels(action="baseline_sanity").observe(time.time() - start_time)
        
        return MLResponse(
            task_id=req.task_id,
            step=MLStep.BASELINE_SANITY,
            success=True,
            result=result,
            execution_time=time.time() - start_time,
            timestamp=time.time()
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(action="baseline_sanity", status="error").inc()
        REQUEST_DURATION.labels(action="baseline_sanity").observe(time.time() - start_time)
        
        return MLResponse(
            task_id=req.task_id,
            step=MLStep.BASELINE_SANITY,
            success=False,
            error=str(e),
            execution_time=time.time() - start_time,
            timestamp=time.time()
        )

@app.post("/experiment_tracking", response_model=MLResponse)
async def experiment_tracking_endpoint(req: ExperimentTrackingRequest):
    """Step 10: Experiment tracking and reproducibility."""
    start_time = time.time()
    
    try:
        result = await experiment_tracking(req)
        
        REQUEST_COUNT.labels(action="experiment_tracking", status="success").inc()
        REQUEST_DURATION.labels(action="experiment_tracking").observe(time.time() - start_time)
        
        return MLResponse(
            task_id=req.task_id,
            step=MLStep.EXPERIMENT_TRACKING,
            success=True,
            result=result,
            execution_time=time.time() - start_time,
            timestamp=time.time()
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(action="experiment_tracking", status="error").inc()
        REQUEST_DURATION.labels(action="experiment_tracking").observe(time.time() - start_time)
        
        return MLResponse(
            task_id=req.task_id,
            step=MLStep.EXPERIMENT_TRACKING,
            success=False,
            error=str(e),
            execution_time=time.time() - start_time,
            timestamp=time.time()
        )

# Health and monitoring endpoints
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "mlflow_available": MLFLOW_AVAILABLE,
        "imbalanced_available": IMBALANCED_AVAILABLE,
        "active_experiments": len(_EXPERIMENTS)
    }

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/experiments")
async def list_experiments():
    return {
        "active_experiments": len(_EXPERIMENTS),
        "experiments": list(_EXPERIMENTS.keys()),
        "experiment_summaries": {
            exp_id: exp.get("experiment_summary", {})
            for exp_id, exp in _EXPERIMENTS.items()
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002) 