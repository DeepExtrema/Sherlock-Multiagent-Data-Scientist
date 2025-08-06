#!/usr/bin/env python3
"""
Configuration Management for Deepline ML Agent
Centralized configuration with environment variable support and validation.
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from pathlib import Path

class MLAgentConfig(BaseModel):
    """Configuration for ML Agent with environment variable support."""
    
    # Service Configuration
    APP_NAME: str = "Deepline ML Agent"
    APP_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8002
    DEBUG: bool = False
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI: Optional[str] = None
    MLFLOW_EXPERIMENT_NAME: str = "deepline-ml-workflow"
    
    # Redis Configuration (for persistent storage)
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    
    # Prometheus Configuration
    PROMETHEUS_NAMESPACE: str = "deepline_ml_agent"
    ENABLE_METRICS: bool = True
    
    # ML Configuration
    DEFAULT_RANDOM_STATE: int = 42
    IMBALANCE_THRESHOLD: float = 0.1
    MAX_FILE_SIZE_MB: int = 100
    
    # Model Configuration
    DEFAULT_ALGORITHMS: Dict[str, str] = {"classification": "random_forest", "regression": "linear_regression"}
    
    # Storage Configuration
    ARTIFACT_DIR: str = "artifacts"
    MODEL_CACHE_SIZE: int = 100
    
    # Security Configuration
    ENABLE_AUTH: bool = False
    JWT_SECRET_KEY: Optional[str] = None
    CORS_ORIGINS: list = ["*"]
    
    # Performance Configuration
    MAX_WORKERS: int = 4
    REQUEST_TIMEOUT: int = 300
    MODEL_TIMEOUT: int = 600
    
    @field_validator('LOG_LEVEL')
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'LOG_LEVEL must be one of {valid_levels}')
        return v.upper()
    
    @field_validator('IMBALANCE_THRESHOLD')
    @classmethod
    def validate_imbalance_threshold(cls, v):
        if not 0 < v < 1:
            raise ValueError('IMBALANCE_THRESHOLD must be between 0 and 1')
        return v
    
    @field_validator('MAX_FILE_SIZE_MB')
    @classmethod
    def validate_max_file_size(cls, v):
        if v <= 0:
            raise ValueError('MAX_FILE_SIZE_MB must be positive')
        return v
    
    @field_validator('PORT')
    @classmethod
    def validate_port(cls, v):
        if not 1024 <= v <= 65535:
            raise ValueError('PORT must be between 1024 and 65535')
        return v
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False
    }
    
    def get_artifact_path(self) -> Path:
        """Get the artifact directory path."""
        path = Path(self.ARTIFACT_DIR)
        path.mkdir(exist_ok=True)
        return path
    
    def get_model_path(self, model_id: str) -> Path:
        """Get the path for a specific model."""
        return self.get_artifact_path() / f"{model_id}.pkl"
    
    def get_experiment_path(self, experiment_id: str) -> Path:
        """Get the path for experiment artifacts."""
        return self.get_artifact_path() / "experiments" / experiment_id
    
    def is_mlflow_enabled(self) -> bool:
        """Check if MLflow is enabled."""
        return self.MLFLOW_TRACKING_URI is not None
    
    def is_redis_enabled(self) -> bool:
        """Check if Redis is enabled."""
        return self.REDIS_URL != "redis://localhost:6379/0" or os.getenv("REDIS_URL") is not None
    
    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration."""
        return {
            "url": self.REDIS_URL,
            "db": self.REDIS_DB,
            "password": self.REDIS_PASSWORD
        }

def load_config_from_env() -> MLAgentConfig:
    """Load configuration with environment variable overrides."""
    config = MLAgentConfig()
    
    # Override with environment variables if present
    if os.getenv("ML_AGENT_HOST"):
        config.HOST = os.getenv("ML_AGENT_HOST")
    if os.getenv("ML_AGENT_PORT"):
        config.PORT = int(os.getenv("ML_AGENT_PORT"))
    if os.getenv("ML_AGENT_DEBUG"):
        config.DEBUG = os.getenv("ML_AGENT_DEBUG").lower() == "true"
    if os.getenv("ML_AGENT_LOG_LEVEL"):
        config.LOG_LEVEL = os.getenv("ML_AGENT_LOG_LEVEL")
    if os.getenv("MLFLOW_TRACKING_URI"):
        config.MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    if os.getenv("MLFLOW_EXPERIMENT_NAME"):
        config.MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")
    if os.getenv("REDIS_URL"):
        config.REDIS_URL = os.getenv("REDIS_URL")
    if os.getenv("REDIS_DB"):
        config.REDIS_DB = int(os.getenv("REDIS_DB"))
    if os.getenv("REDIS_PASSWORD"):
        config.REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
    
    return config

# Global configuration instance
config = load_config_from_env()

def get_config() -> MLAgentConfig:
    """Get the global configuration instance."""
    return config

def reload_config() -> MLAgentConfig:
    """Reload configuration from environment."""
    global config
    config = MLAgentConfig()
    return config

# Configuration validation
def validate_config() -> bool:
    """Validate the current configuration."""
    try:
        # Test artifact directory creation
        config.get_artifact_path()
        
        # Validate MLflow configuration if enabled
        if config.is_mlflow_enabled():
            if not config.MLFLOW_TRACKING_URI:
                raise ValueError("MLFLOW_TRACKING_URI is required when MLflow is enabled")
        
        # Validate Redis configuration if enabled
        if config.is_redis_enabled():
            redis_config = config.get_redis_config()
            if not redis_config["url"]:
                raise ValueError("REDIS_URL is required when Redis is enabled")
        
        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False

if __name__ == "__main__":
    # Print configuration for debugging
    print("ML Agent Configuration:")
    print(f"  App Name: {config.APP_NAME}")
    print(f"  Version: {config.APP_VERSION}")
    print(f"  Host: {config.HOST}")
    print(f"  Port: {config.PORT}")
    print(f"  Debug: {config.DEBUG}")
    print(f"  Log Level: {config.LOG_LEVEL}")
    print(f"  MLflow Enabled: {config.is_mlflow_enabled()}")
    print(f"  Redis Enabled: {config.is_redis_enabled()}")
    print(f"  Metrics Enabled: {config.ENABLE_METRICS}")
    print(f"  Artifact Dir: {config.get_artifact_path()}")
    
    # Validate configuration
    if validate_config():
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration validation failed") 