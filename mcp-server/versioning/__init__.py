"""
Snapshot and versioning abstraction for immutable raw copies.

Provides a simple local snapshot store with content hashing; can be
extended to S3/LakeFS later.
"""

from .snapshot_store import SnapshotMetadata, LocalSnapshotStore

__all__ = [
    "SnapshotMetadata",
    "LocalSnapshotStore",
]


