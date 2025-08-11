"""
Data source registry and connector abstractions.

Provides simple in-memory registry for API, DB dump, and stream sources.
"""

from .registry import (
    DataSourceType,
    DataSource,
    ApiSourceConfig,
    DbDumpSourceConfig,
    StreamSourceConfig,
    DataSourceRegistry,
)

__all__ = [
    "DataSourceType",
    "DataSource",
    "ApiSourceConfig",
    "DbDumpSourceConfig",
    "StreamSourceConfig",
    "DataSourceRegistry",
]


