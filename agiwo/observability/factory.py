"""Factory helpers for constructing trace storage implementations."""

from typing import Any, Protocol

from agiwo.observability.base import BaseTraceStorage
from agiwo.observability.memory_store import InMemoryTraceStorage
from agiwo.observability.sqlite_store import SQLiteTraceStorage
from agiwo.observability.store import MongoTraceStorage


class TraceStorageConfigLike(Protocol):
    storage_type: str | None
    config: dict[str, Any]


def create_trace_storage(
    config: TraceStorageConfigLike,
) -> BaseTraceStorage | None:
    storage_type = config.storage_type
    if storage_type is None:
        return None

    cfg = dict(config.config)
    if storage_type == "memory":
        buffer_size = cfg.get("buffer_size", 200)
        return InMemoryTraceStorage(buffer_size=buffer_size)
    if storage_type == "sqlite":
        db_path = cfg.get("db_path", "agiwo.db")
        collection_name = cfg.get("collection_name", "agiwo_traces")
        return SQLiteTraceStorage(
            db_path=db_path,
            collection_name=collection_name,
        )  # noqa: AGW013 - observability factory owns trace storage construction
    if storage_type == "mongodb":
        mongo_uri = cfg.get("mongo_uri")
        db_name = cfg.get("db_name", "agiwo")
        collection_name = cfg.get("collection_name", "traces")
        return MongoTraceStorage(
            mongo_uri=mongo_uri,
            db_name=db_name,
            collection_name=collection_name,
        )  # noqa: AGW013 - observability factory owns trace storage construction
    raise ValueError(f"Unknown trace_storage_type: {storage_type}")


__all__ = ["TraceStorageConfigLike", "create_trace_storage"]
