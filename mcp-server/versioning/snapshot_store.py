#!/usr/bin/env python3
"""
Local Snapshot Store

Stores immutable copies of uploaded files under a content-addressed path:
  snapshots/<sha256>/<original_filename>

Provides metadata for audit: uploader, license, pii flags.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


SNAPSHOT_DIR = Path("snapshots")
SNAPSHOT_DIR.mkdir(exist_ok=True)


@dataclass
class SnapshotMetadata:
    snapshot_id: str
    created_at: str
    original_filename: str
    content_hash: str
    size_bytes: int
    stored_path: str
    uploader: Optional[str] = None
    license: Optional[str] = None
    pii_detected: bool = False
    pii_summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LocalSnapshotStore:
    def __init__(self, base_dir: Path | str = SNAPSHOT_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

    def _hash_file(self, file_path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def snapshot(self, src_path: Path | str, uploader: Optional[str] = None,
                 license: Optional[str] = None,
                 pii_summary: Optional[Dict[str, Any]] = None) -> SnapshotMetadata:
        src_path = Path(src_path)
        if not src_path.exists():
            raise FileNotFoundError(src_path)

        content_hash = self._hash_file(src_path)
        dest_dir = self.base_dir / content_hash
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / src_path.name

        # If file already exists with same hash+name, do not overwrite
        if not dest_path.exists():
            # Copy file bytes without altering timestamps of the store
            with open(src_path, "rb") as src, open(dest_path, "wb") as dst:
                for chunk in iter(lambda: src.read(1024 * 1024), b""):
                    dst.write(chunk)

        stat = dest_path.stat()
        snapshot_id = f"{content_hash[:12]}-{int(stat.st_mtime)}"
        meta = SnapshotMetadata(
            snapshot_id=snapshot_id,
            created_at=datetime.utcnow().isoformat(),
            original_filename=src_path.name,
            content_hash=content_hash,
            size_bytes=stat.st_size,
            stored_path=str(dest_path.resolve()),
            uploader=uploader,
            license=license,
            pii_detected=bool(pii_summary),
            pii_summary=pii_summary,
        )
        # Persist metadata alongside file
        meta_path = dest_dir / f"{src_path.name}.meta.json"
        try:
            import json
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        return meta




