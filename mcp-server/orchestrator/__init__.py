"""
Master Orchestrator Package

This package contains the core orchestrator components for the Deepline system:
- LLM translation with Guardrails
- Async translation queue with Redis backing
- Rule-based fallback systems
- Workflow management and execution
- Rate limiting and concurrency control
- SLA monitoring and alerting
"""

from .translator import LLMTranslator, RuleBasedTranslator, FallbackRouter, NeedsHumanError
from .translation_queue import TranslationQueue, TranslationWorker, TranslationStatus
from .workflow_manager import WorkflowManager
from .security import SecurityUtils
from .cache_client import CacheClient
from .guards import ConcurrencyGuard, RateLimiter, TokenRateLimiter
from .sla_monitor import SLAMonitor
from .decision_engine import DecisionEngine
from .telemetry import TelemetryManager, initialize_telemetry
from .deadlock_monitor import DeadlockMonitor
from .dsl_repair_pipeline import repair as repair_dsl
from .llm_client import LlmClient, call_llm

__version__ = "1.0.0"
__all__ = [
    "LLMTranslator",
    "RuleBasedTranslator",
    "FallbackRouter",
    "NeedsHumanError",
    "TranslationQueue",
    "TranslationWorker",
    "TranslationStatus",
    "WorkflowManager",
    "SecurityUtils",
    "CacheClient",
    "ConcurrencyGuard",
    "TokenRateLimiter",
    "RateLimiter",
    "SLAMonitor",
    "DecisionEngine",
    "TelemetryManager",
    "initialize_telemetry",
    "DeadlockMonitor",
    "repair_dsl",
    "LlmClient",
    "call_llm"
]