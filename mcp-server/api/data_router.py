"""
Data API Router

Endpoints for:
- Registering data sources (API, DB dumps, streams)
- Listing and retrieving sources
- Uploading datasets with governance checks and immutable snapshotting
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from pydantic import BaseModel, Field

from data_sources import (
    DataSourceType,
    DataSource,
    ApiSourceConfig,
    DbDumpSourceConfig,
    StreamSourceConfig,
    DataSourceRegistry,
)
from versioning import LocalSnapshotStore
from security.authentication import data_governance


class RegisterSourceRequest(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12], description="Source identifier")
    type: DataSourceType
    config: Dict[str, Any]
    tags: List[str] = Field(default_factory=list)
    owner: Optional[str] = None
    license: Optional[str] = None
    pii_expected: bool = False


class RegisterSourceResponse(BaseModel):
    id: str
    type: DataSourceType
    created: bool


class SourceListResponse(BaseModel):
    sources: List[DataSource]
    count: int


def create_data_router(
    registry: DataSourceRegistry | None = None,
    snapshot_store: LocalSnapshotStore | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/data", tags=["data"])

    _registry = registry or DataSourceRegistry()
    _snapshots = snapshot_store or LocalSnapshotStore()

    @router.post("/sources", response_model=RegisterSourceResponse)
    async def register_source(request: RegisterSourceRequest):
        try:
            # Basic config validation based on type
            if request.type == DataSourceType.API:
                ApiSourceConfig(**request.config)
            elif request.type == DataSourceType.DB_DUMP:
                DbDumpSourceConfig(**request.config)
            elif request.type == DataSourceType.STREAM:
                StreamSourceConfig(**request.config)

            source = DataSource(
                id=request.id,
                type=request.type,
                config=request.config,
                tags=request.tags,
                owner=request.owner,
                license=request.license,
                pii_expected=bool(request.pii_expected),
            )
            created = _registry.get(source.id) is None
            _registry.upsert(source)
            return RegisterSourceResponse(id=source.id, type=source.type, created=created)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    @router.get("/sources", response_model=SourceListResponse)
    async def list_sources(type: Optional[DataSourceType] = None):
        items = _registry.list(type_filter=type)
        return SourceListResponse(sources=items, count=len(items))

    @router.get("/sources/{source_id}", response_model=DataSource)
    async def get_source(source_id: str):
        source = _registry.get(source_id)
        if source is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Source not found")
        return source

    @router.post("/upload", response_model=Dict[str, Any])
    async def upload_with_snapshot(
        file: UploadFile = File(...),
        name: str = Form(...),
        license: Optional[str] = Form(default=None),
        uploader: Optional[str] = Form(default=None),
    ):
        try:
            # Save to temp path
            uploads_dir = Path("uploads")
            uploads_dir.mkdir(exist_ok=True)
            temp_path = uploads_dir / f"{uuid.uuid4().hex}_{file.filename}"
            with open(temp_path, "wb") as buffer:
                buffer.write(await file.read())

            # Governance: detect PII for audit
            try:
                sample_bytes = temp_path.read_bytes()[:200_000]
                pii_matches = data_governance.detect_pii(sample_bytes.decode("utf-8", errors="ignore"))
            except Exception:
                pii_matches = []

            pii_summary = {"count": len(pii_matches), "types": sorted({m["type"] for m in pii_matches})} if pii_matches else None

            # Snapshot immutably
            snapshot_meta = _snapshots.snapshot(temp_path, uploader=uploader, license=license, pii_summary=pii_summary)

            return {
                "dataset_name": name,
                "filename": file.filename,
                "snapshot": snapshot_meta.to_dict(),
                "pii_detected": bool(pii_matches),
                "pii_matches_count": len(pii_matches),
                "message": "Dataset uploaded and snapshotted immutably",
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    return router


