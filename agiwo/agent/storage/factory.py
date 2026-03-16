"""
Storage factory for creating storage instances from configuration.

Storage instances handle their own lazy connection internally.
"""

from agiwo.config.settings import settings
from agiwo.agent.options import RunStepStorageConfig
from agiwo.agent.storage.base import RunStepStorage, InMemoryRunStepStorage
from agiwo.agent.storage.session import (
    InMemorySessionStorage,
    SessionStorage,
    SQLiteSessionStorage,
)
from agiwo.agent.storage.sqlite import SQLiteRunStepStorage
from agiwo.agent.storage.mongo import MongoRunStepStorage


class StorageFactory:
    """
    Factory for creating storage instances from configuration.

    Created instances connect lazily on first operation.
    """

    @staticmethod
    def create_run_step_storage(config: RunStepStorageConfig) -> RunStepStorage:
        storage_type = config.storage_type
        cfg = config.config

        if storage_type == "memory":
            return InMemoryRunStepStorage()
        elif storage_type == "sqlite":
            db_path = cfg.get("db_path", "agiwo.db")
            return SQLiteRunStepStorage(db_path=db_path)
        elif storage_type == "mongodb":
            uri = cfg.get("mongo_uri") or cfg.get("uri", "mongodb://localhost:27017")
            db_name = cfg.get("db_name", "agiwo")
            return MongoRunStepStorage(uri=uri, db_name=db_name)
        else:
            raise ValueError(f"Unknown run_step_storage_type: {storage_type}")

    @staticmethod
    def create_session_storage(config: RunStepStorageConfig) -> SessionStorage:
        storage_type = config.storage_type
        cfg = config.config
        if storage_type == "sqlite":
            db_path = cfg.get("db_path", "agiwo.db")
            resolved_path = settings.resolve_path(db_path)
            resolved = str(resolved_path) if resolved_path is not None else db_path
            return SQLiteSessionStorage(resolved)
        return InMemorySessionStorage()


__all__ = ["StorageFactory"]
