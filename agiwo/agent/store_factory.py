"""
Centralized factory for creating stores based on settings.

Stores are internal to the Agent SDK. Users activate them via settings/env vars.
Default = no storage.
"""

from agiwo.agent.session.base import SessionStore, InMemorySessionStore
from agiwo.agent.session.sqlite import SQLiteSessionStore
from agiwo.agent.session.mongo import MongoSessionStore
from agiwo.config.settings import settings
from agiwo.observability.base import BaseTraceStore
from agiwo.observability.store import MongoTraceStore
from agiwo.observability.sqlite_store import SQLiteTraceStore


def create_session_store() -> SessionStore:
    """
    Create a SessionStore based on global settings.

    Session context management is fundamental to agent systems,
    so this always returns a valid SessionStore.
    Defaults to InMemorySessionStore when no persistent store is configured.
    """
    store_type = settings.default_session_store

    if store_type == "sqlite":
        return SQLiteSessionStore(db_path=settings.sqlite_db_path or "~/.agiwo/agiwo.db")
    if store_type == "mongo":
        return MongoSessionStore(
            uri=settings.mongo_uri or "mongodb://localhost:27017",
            db_name=settings.mongo_db_name or "agiwo",
        )

    return InMemorySessionStore()


def create_trace_store() -> BaseTraceStore | None:
    """
    Create a BaseTraceStore based on global settings.

    Returns None if no trace store is configured (default).
    """
    store_type = settings.default_trace_store
    if store_type is None:
        return None

    if store_type == "mongo":
        return MongoTraceStore(
            mongo_uri=settings.mongo_uri,
            db_name=settings.mongo_db_name or "agiwo",
            collection_name=settings.trace_collection_name or "agiwo_traces",
            buffer_size=settings.trace_buffer_size,
        )
    if store_type == "sqlite":
        return SQLiteTraceStore(
            db_path=settings.sqlite_db_path or "~/.agiwo/agiwo.db",
            collection_name=settings.trace_collection_name or "agiwo_traces",
            buffer_size=settings.trace_buffer_size,
        )

    return None


__all__ = ["create_session_store", "create_trace_store"]
