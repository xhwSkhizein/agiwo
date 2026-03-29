"""Factory helpers for constructing trace storage implementations."""

from typing import Any, Protocol

from agiwo.observability.base import BaseTraceStorage
from agiwo.observability.memory_store import InMemoryTraceStorage
from agiwo.observability.sqlite_store import SQLiteTraceStorage
from agiwo.observability.store import MongoTraceStorage
from agiwo.utils.storage_factory import create_storage


class TraceStorageConfigLike(Protocol):
    storage_type: str | None
    config: dict[str, Any]


def _make_memory_trace(cfg: dict[str, Any]) -> BaseTraceStorage:
    return InMemoryTraceStorage(buffer_size=cfg.get("buffer_size", 200))


def _make_sqlite_trace(cfg: dict[str, Any]) -> BaseTraceStorage:
    return SQLiteTraceStorage(
        db_path=cfg.get("db_path", "agiwo.db"),
        collection_name=cfg.get("collection_name", "agiwo_traces"),
    )


def _make_mongo_trace(cfg: dict[str, Any]) -> BaseTraceStorage:
    return MongoTraceStorage(
        mongo_uri=cfg.get("mongo_uri"),
        db_name=cfg.get("db_name", "agiwo"),
        collection_name=cfg.get("collection_name", "traces"),
    )


_TRACE_BACKENDS: dict[str, Any] = {
    "memory": _make_memory_trace,
    "sqlite": _make_sqlite_trace,
    "mongodb": _make_mongo_trace,
}


def create_trace_storage(
    config: TraceStorageConfigLike,
) -> BaseTraceStorage | None:
    storage_type = config.storage_type
    if storage_type is None:
        return None
    return create_storage(
        storage_type,
        dict(config.config),
        _TRACE_BACKENDS,
        label="trace_storage",
    )


__all__ = ["TraceStorageConfigLike", "create_trace_storage"]
