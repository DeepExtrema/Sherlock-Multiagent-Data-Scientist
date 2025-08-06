#!/usr/bin/env python3
"""
Persistent Storage Layer for ML Agent
Provides Redis and SQLite backends for experiment and model storage.
"""

import json
import pickle
import sqlite3
import asyncio
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime
import logging

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available - using SQLite fallback")

logger = logging.getLogger(__name__)

class StorageBackend:
    """Abstract base class for storage backends."""
    
    async def store_experiment(self, experiment_id: str, data: Dict[str, Any]) -> bool:
        """Store experiment data."""
        raise NotImplementedError
    
    async def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve experiment data."""
        raise NotImplementedError
    
    async def list_experiments(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List experiments with optional filtering."""
        raise NotImplementedError
    
    async def store_model(self, model_id: str, model_data: bytes) -> bool:
        """Store model binary data."""
        raise NotImplementedError
    
    async def get_model(self, model_id: str) -> Optional[bytes]:
        """Retrieve model binary data."""
        raise NotImplementedError
    
    async def delete_experiment(self, experiment_id: str) -> bool:
        """Delete experiment data."""
        raise NotImplementedError
    
    async def delete_model(self, model_id: str) -> bool:
        """Delete model data."""
        raise NotImplementedError
    
    async def close(self):
        """Close storage connection."""
        pass

class SQLiteStorage(StorageBackend):
    """SQLite-based storage backend."""
    
    def __init__(self, db_path: str = "ml_agent.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY,
                    data BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    async def store_experiment(self, experiment_id: str, data: Dict[str, Any]) -> bool:
        """Store experiment data in SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO experiments (id, data, updated_at) VALUES (?, ?, ?)",
                    (experiment_id, json.dumps(data), datetime.now().isoformat())
                )
                conn.commit()
            logger.info(f"Stored experiment {experiment_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store experiment {experiment_id}: {e}")
            return False
    
    async def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve experiment data from SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT data FROM experiments WHERE id = ?",
                    (experiment_id,)
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
                return None
        except Exception as e:
            logger.error(f"Failed to retrieve experiment {experiment_id}: {e}")
            return None
    
    async def list_experiments(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List experiments from SQLite with optional filtering."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT id, data, created_at, updated_at FROM experiments"
                params = []
                
                if filters:
                    conditions = []
                    if 'status' in filters:
                        conditions.append("json_extract(data, '$.status') = ?")
                        params.append(filters['status'])
                    if 'from_date' in filters:
                        conditions.append("created_at >= ?")
                        params.append(filters['from_date'])
                    if 'to_date' in filters:
                        conditions.append("created_at <= ?")
                        params.append(filters['to_date'])
                    
                    if conditions:
                        query += " WHERE " + " AND ".join(conditions)
                
                query += " ORDER BY created_at DESC"
                
                cursor = conn.execute(query, params)
                experiments = []
                for row in cursor.fetchall():
                    experiment_data = json.loads(row[1])
                    experiment_data['id'] = row[0]
                    experiment_data['created_at'] = row[2]
                    experiment_data['updated_at'] = row[3]
                    experiments.append(experiment_data)
                
                return experiments
        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
            return []
    
    async def store_model(self, model_id: str, model_data: bytes) -> bool:
        """Store model binary data in SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO models (id, data) VALUES (?, ?)",
                    (model_id, model_data)
                )
                conn.commit()
            logger.info(f"Stored model {model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store model {model_id}: {e}")
            return False
    
    async def get_model(self, model_id: str) -> Optional[bytes]:
        """Retrieve model binary data from SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT data FROM models WHERE id = ?",
                    (model_id,)
                )
                row = cursor.fetchone()
                if row:
                    return row[0]
                return None
        except Exception as e:
            logger.error(f"Failed to retrieve model {model_id}: {e}")
            return None
    
    async def delete_experiment(self, experiment_id: str) -> bool:
        """Delete experiment data from SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
                conn.commit()
            logger.info(f"Deleted experiment {experiment_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete experiment {experiment_id}: {e}")
            return False
    
    async def delete_model(self, model_id: str) -> bool:
        """Delete model data from SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
                conn.commit()
            logger.info(f"Deleted model {model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False

class RedisStorage(StorageBackend):
    """Redis-based storage backend."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not available")
        
        self.redis_url = redis_url
        self.redis_client = None
        self.namespace = "ml_agent"
    
    async def connect(self):
        """Connect to Redis."""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def store_experiment(self, experiment_id: str, data: Dict[str, Any]) -> bool:
        """Store experiment data in Redis."""
        try:
            key = f"{self.namespace}:experiment:{experiment_id}"
            await self.redis_client.set(key, json.dumps(data))
            await self.redis_client.expire(key, 86400 * 30)  # 30 days TTL
            logger.info(f"Stored experiment {experiment_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store experiment {experiment_id}: {e}")
            return False
    
    async def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve experiment data from Redis."""
        try:
            key = f"{self.namespace}:experiment:{experiment_id}"
            data = await self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve experiment {experiment_id}: {e}")
            return None
    
    async def list_experiments(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List experiments from Redis with optional filtering."""
        try:
            pattern = f"{self.namespace}:experiment:*"
            keys = await self.redis_client.keys(pattern)
            experiments = []
            
            for key in keys:
                data = await self.redis_client.get(key)
                if data:
                    experiment_data = json.loads(data)
                    experiment_id = key.decode().split(":")[-1]
                    experiment_data['id'] = experiment_id
                    experiments.append(experiment_data)
            
            # Apply filters if provided
            if filters:
                filtered_experiments = []
                for exp in experiments:
                    if self._matches_filters(exp, filters):
                        filtered_experiments.append(exp)
                experiments = filtered_experiments
            
            # Sort by creation time (newest first)
            experiments.sort(key=lambda x: x.get('created_at', 0), reverse=True)
            return experiments
        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
            return []
    
    def _matches_filters(self, experiment: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if experiment matches filters."""
        for key, value in filters.items():
            if key == 'status' and experiment.get('status') != value:
                return False
            if key == 'from_date' and experiment.get('created_at', 0) < value:
                return False
            if key == 'to_date' and experiment.get('created_at', 0) > value:
                return False
        return True
    
    async def store_model(self, model_id: str, model_data: bytes) -> bool:
        """Store model binary data in Redis."""
        try:
            key = f"{self.namespace}:model:{model_id}"
            await self.redis_client.set(key, model_data)
            await self.redis_client.expire(key, 86400 * 7)  # 7 days TTL for models
            logger.info(f"Stored model {model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store model {model_id}: {e}")
            return False
    
    async def get_model(self, model_id: str) -> Optional[bytes]:
        """Retrieve model binary data from Redis."""
        try:
            key = f"{self.namespace}:model:{model_id}"
            data = await self.redis_client.get(key)
            return data
        except Exception as e:
            logger.error(f"Failed to retrieve model {model_id}: {e}")
            return None
    
    async def delete_experiment(self, experiment_id: str) -> bool:
        """Delete experiment data from Redis."""
        try:
            key = f"{self.namespace}:experiment:{experiment_id}"
            await self.redis_client.delete(key)
            logger.info(f"Deleted experiment {experiment_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete experiment {experiment_id}: {e}")
            return False
    
    async def delete_model(self, model_id: str) -> bool:
        """Delete model data from Redis."""
        try:
            key = f"{self.namespace}:model:{model_id}"
            await self.redis_client.delete(key)
            logger.info(f"Deleted model {model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()

class StorageManager:
    """Storage manager that handles backend selection and fallback."""
    
    def __init__(self, backend_type: str = "auto", **kwargs):
        self.backend_type = backend_type
        self.backend = None
        self.kwargs = kwargs
    
    async def initialize(self):
        """Initialize the storage backend."""
        if self.backend_type == "redis" or (self.backend_type == "auto" and REDIS_AVAILABLE):
            try:
                # Extract Redis-specific parameters
                redis_url = self.kwargs.get('redis_url', 'redis://localhost:6379/0')
                self.backend = RedisStorage(redis_url=redis_url)
                await self.backend.connect()
                logger.info("Using Redis storage backend")
            except Exception as e:
                logger.warning(f"Redis initialization failed, falling back to SQLite: {e}")
                # Extract SQLite-specific parameters
                db_path = self.kwargs.get('db_path', 'ml_agent.db')
                self.backend = SQLiteStorage(db_path=db_path)
                logger.info("Using SQLite storage backend")
        else:
            # Extract SQLite-specific parameters
            db_path = self.kwargs.get('db_path', 'ml_agent.db')
            self.backend = SQLiteStorage(db_path=db_path)
            logger.info("Using SQLite storage backend")
    
    async def store_experiment(self, experiment_id: str, data: Dict[str, Any]) -> bool:
        """Store experiment data."""
        return await self.backend.store_experiment(experiment_id, data)
    
    async def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve experiment data."""
        return await self.backend.get_experiment(experiment_id)
    
    async def list_experiments(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List experiments with optional filtering."""
        return await self.backend.list_experiments(filters)
    
    async def store_model(self, model_id: str, model_data: bytes) -> bool:
        """Store model binary data."""
        return await self.backend.store_model(model_id, model_data)
    
    async def get_model(self, model_id: str) -> Optional[bytes]:
        """Retrieve model binary data."""
        return await self.backend.get_model(model_id)
    
    async def delete_experiment(self, experiment_id: str) -> bool:
        """Delete experiment data."""
        return await self.backend.delete_experiment(experiment_id)
    
    async def delete_model(self, model_id: str) -> bool:
        """Delete model data."""
        return await self.backend.delete_model(model_id)
    
    async def close(self):
        """Close storage connection."""
        if self.backend:
            await self.backend.close()

# Global storage manager instance
storage_manager = StorageManager()

async def get_storage() -> StorageManager:
    """Get the global storage manager instance."""
    return storage_manager

async def initialize_storage(backend_type: str = "auto", **kwargs):
    """Initialize the global storage manager."""
    global storage_manager
    storage_manager = StorageManager(backend_type, **kwargs)
    await storage_manager.initialize() 