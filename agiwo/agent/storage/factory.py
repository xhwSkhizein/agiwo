"""
Storage factory for creating storage instances from configuration.

Storage instances handle their own lazy connection internally.
"""

from agiwo.agent.options import RunStepStorageConfig, TraceStorageConfig
from agiwo.agent.storage.base import RunStepStorage, InMemoryRunStepStorage
from agiwo.agent.storage.sqlite import SQLiteRunStepStorage
from agiwo.agent.storage.mongo import MongoRunStepStorage
from agiwo.observability.base import BaseTraceStorage
from agiwo.observability.collector import InMemoryTraceStorage
from agiwo.observability.sqlite_store import SQLiteTraceStorage
from agiwo.observability.store import MongoTraceStorage


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
            uri = cfg.get("uri", "mongodb://localhost:27017")
            db_name = cfg.get("db_name", "agiwo")
            return MongoRunStepStorage(uri=uri, db_name=db_name)
        else:
            raise ValueError(f"Unknown run_step_storage_type: {storage_type}")

    @staticmethod
    def create_trace_storage(config: TraceStorageConfig) -> BaseTraceStorage | None:
        storage_type = config.storage_type

        if storage_type is None:
            return None

        cfg = config.config

        if storage_type == "memory":
            buffer_size = cfg.get("buffer_size", 200)
            return InMemoryTraceStorage(buffer_size=buffer_size)
        elif storage_type == "sqlite":
            db_path = cfg.get("db_path", "agiwo.db")
            collection_name = cfg.get("collection_name", "agiwo_traces")
            buffer_size = cfg.get("buffer_size", 200)
            return SQLiteTraceStorage(
                db_path=db_path,
                collection_name=collection_name,
                buffer_size=buffer_size,
            )
        elif storage_type == "mongodb":
            mongo_uri = cfg.get("mongo_uri")
            db_name = cfg.get("db_name", "agiwo")
            collection_name = cfg.get("collection_name", "traces")
            buffer_size = cfg.get("buffer_size", 200)
            return MongoTraceStorage(
                mongo_uri=mongo_uri,
                db_name=db_name,
                collection_name=collection_name,
                buffer_size=buffer_size,
            )
        else:
            raise ValueError(f"Unknown trace_storage_type: {storage_type}")


__all__ = ["StorageFactory"]
