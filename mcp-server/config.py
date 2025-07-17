"""
Configuration management for EDA Server.
Loads and validates settings from config.yaml.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel, Field

class RetryConfig(BaseModel):
    max_retries: int = Field(3, ge=0)
    backoff_base_s: int = Field(30, gt=0)
    backoff_max_s: int = Field(300, gt=0)

class SchedulingConfig(BaseModel):
    sla_task_complete_s: int = Field(600, gt=0)  # 10 minutes
    sla_workflow_complete_s: int = Field(3600, gt=0)  # 1 hour

class WorkloadEstimateConfig(BaseModel):
    tasks_per_hour: int = Field(30, gt=0)
    avg_task_duration_s: int = Field(240, gt=0)  # 4 minutes

class OrchestratorConfig(BaseModel):
    max_concurrent_workflows: int = Field(1, gt=0)
    retry: RetryConfig = Field(default_factory=lambda: RetryConfig())
    scheduling: SchedulingConfig = Field(default_factory=lambda: SchedulingConfig())
    workload_estimate: WorkloadEstimateConfig = Field(default_factory=lambda: WorkloadEstimateConfig())

class MissingDataConfig(BaseModel):
    column_drop_threshold: float = Field(0.50, ge=0.0, le=1.0)
    row_drop_threshold: float = Field(0.50, ge=0.0, le=1.0)
    systematic_correlation_threshold: float = Field(0.70, ge=0.0, le=1.0)
    imputation: Dict[str, float] = Field(default_factory=dict)

class OutlierDetectionConfig(BaseModel):
    iqr_factor: float = Field(1.5, gt=0.0)
    contamination_default: float = Field(0.05, ge=0.0, le=1.0)
    mahalanobis_confidence: float = Field(0.975, ge=0.0, le=1.0)
    max_columns_visualized: int = Field(10, gt=0)
    sample_size_limit: int = Field(10000, gt=0)

class SchemaInferenceConfig(BaseModel):
    id_uniqueness_threshold: float = Field(0.90, ge=0.0, le=1.0)
    datetime_success_rate: float = Field(0.80, ge=0.0, le=1.0)
    precision_sample_size: int = Field(100, gt=0)
    max_sample_values: int = Field(5, gt=0)

class FeatureTransformationConfig(BaseModel):
    rare_category_threshold: float = Field(0.005, ge=0.0, le=1.0)
    vif_severe_threshold: float = Field(10.0, gt=0.0)
    vif_moderate_threshold: float = Field(5.0, gt=0.0)
    boxcox_epsilon: float = Field(1e-6, gt=0.0)
    skew_improvement_threshold: float = Field(0.5, gt=0.0)
    binning_n_bins: int = Field(5, gt=1)
    supervised_binning_min_samples: int = Field(10, gt=0)

class VisualizationConfig(BaseModel):
    correlation_sample_size: int = Field(10000, gt=0)
    max_points_scatter: int = Field(5000, gt=0)
    figure_dpi: int = Field(150, gt=0)
    correlation_label_threshold: float = Field(0.5, ge=0.0, le=1.0)

class PerformanceConfig(BaseModel):
    memory_warning_threshold: int = Field(1000, gt=0)  # MB
    max_rows_processed: int = Field(100000, gt=0)
    chunk_size: int = Field(10000, gt=0)

class CheckpointsConfig(BaseModel):
    require_approval: bool = True
    approval_timeout: int = Field(300, gt=0)  # seconds
    auto_approve_small_changes: bool = True

class LLMConfig(BaseModel):
    model_version: str = Field("claude-3-sonnet-20240229")
    max_input_length: int = Field(10000, gt=0)
    llm_max_tokens: int = Field(4000, gt=0)
    llm_max_retries: int = Field(3, ge=0)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    rail_schema_path: str = Field("orchestrator/rail_schema.xml")
    system_prompt: str = Field(default="")

class RulesConfig(BaseModel):
    rule_mappings: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

class InfrastructureConfig(BaseModel):
    mongo_url: str = Field("mongodb://localhost:27017")
    db_name: str = Field("deepline")
    kafka_bootstrap_servers: str = Field("localhost:9092")
    task_requests_topic: str = Field("task.requests")
    task_events_topic: str = Field("task.events")

class RateLimitsConfig(BaseModel):
    requests_per_minute: int = Field(60, gt=0)
    requests_per_hour: int = Field(1000, gt=0)
    burst_requests: int = Field(10, gt=0)

class SLAConfig(BaseModel):
    check_interval_seconds: int = Field(30, gt=0)
    task_timeout_seconds: int = Field(600, gt=0)
    workflow_timeout_seconds: int = Field(3600, gt=0)

class CacheConfig(BaseModel):
    redis_url: str = Field("redis://localhost:6379")
    namespace: str = Field("master_orchestrator")
    default_ttl: int = Field(3600, gt=0)

class MasterOrchestratorConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=lambda: LLMConfig())
    rules: RulesConfig = Field(default_factory=lambda: RulesConfig())
    enable_human_fallback: bool = Field(True)
    min_confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)
    infrastructure: InfrastructureConfig = Field(default_factory=lambda: InfrastructureConfig())
    rate_limits: RateLimitsConfig = Field(default_factory=lambda: RateLimitsConfig())
    sla: SLAConfig = Field(default_factory=lambda: SLAConfig())
    cache: CacheConfig = Field(default_factory=lambda: CacheConfig())

class EDAConfig(BaseModel):
    missing_data: MissingDataConfig
    outlier_detection: OutlierDetectionConfig
    schema_inference: SchemaInferenceConfig
    feature_transformation: FeatureTransformationConfig
    visualization: VisualizationConfig
    performance: PerformanceConfig
    checkpoints: CheckpointsConfig
    orchestrator: OrchestratorConfig
    master_orchestrator: MasterOrchestratorConfig = Field(default_factory=lambda: MasterOrchestratorConfig())

def load_config(config_path: str = "config.yaml") -> EDAConfig:
    """
    Load configuration from YAML file with validation.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Validated configuration object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
        ValidationError: If config values are invalid
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return EDAConfig(**config_data)

def get_config() -> EDAConfig:
    """
    Get the global configuration instance.
    Creates default config if none exists.
    """
    try:
        return load_config()
    except (FileNotFoundError, yaml.YAMLError, Exception):
        # Return default configuration if file is missing or invalid
        print("Warning: Using default configuration. Create config.yaml for customization.")
        return EDAConfig(
            missing_data=MissingDataConfig(),
            outlier_detection=OutlierDetectionConfig(),
            schema_inference=SchemaInferenceConfig(),
            feature_transformation=FeatureTransformationConfig(),
            visualization=VisualizationConfig(),
            performance=PerformanceConfig(),
            checkpoints=CheckpointsConfig(),
            orchestrator=OrchestratorConfig(
                retry=RetryConfig(),
                scheduling=SchedulingConfig(),
                workload_estimate=WorkloadEstimateConfig()
            ),
            master_orchestrator=MasterOrchestratorConfig(
                llm=LLMConfig(),
                rules=RulesConfig(),
                infrastructure=InfrastructureConfig(),
                rate_limits=RateLimitsConfig(),
                sla=SLAConfig(),
                cache=CacheConfig()
            )
        )

# Global configuration instance
config = get_config() 