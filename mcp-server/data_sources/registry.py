#!/usr/bin/env python3
"""
Data Source Registry

Minimal registry to track and retrieve data sources for ingestion
(APIs, database dumps, and streaming sources).
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime


class DataSourceType(str, Enum):
    API = "api"
    DB_DUMP = "db_dump"
    STREAM = "stream"


class ApiSourceConfig(BaseModel):
    name: str
    base_url: HttpUrl
    auth_header: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    params: Dict[str, Any] = Field(default_factory=dict)


class DbDumpSourceConfig(BaseModel):
    name: str
    location: str  # file path, S3 URI, etc.
    format: str = Field(default="csv", pattern=r"^(csv|parquet|json|xlsx)$")


class StreamSourceConfig(BaseModel):
    name: str
    protocol: str = Field(pattern=r"^(kafka|mqtt|websocket)$")
    endpoint: str
    topic: Optional[str] = None


class DataSource(BaseModel):
    id: str
    type: DataSourceType
    config: Dict[str, Any]
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    tags: List[str] = Field(default_factory=list)
    owner: Optional[str] = None
    license: Optional[str] = None
    pii_expected: bool = False


class DataSourceRegistry:
    """In-memory registry. Replace with persistent store later."""

    def __init__(self):
        self._sources: Dict[str, DataSource] = {}

    def upsert(self, source: DataSource) -> DataSource:
        source.updated_at = datetime.utcnow().isoformat()
        self._sources[source.id] = source
        return source

    def get(self, source_id: str) -> Optional[DataSource]:
        return self._sources.get(source_id)

    def list(self, type_filter: Optional[DataSourceType] = None) -> List[DataSource]:
        if type_filter is None:
            return list(self._sources.values())
        return [s for s in self._sources.values() if s.type == type_filter]

    def delete(self, source_id: str) -> bool:
        return self._sources.pop(source_id, None) is not None




