"""
Storage constructors for creating storage instances from configuration.

Storage instances handle their own lazy connection internally.
"""

from agiwo.config.settings import settings
from agiwo.agent.models.config import RunStepStorageConfig
from agiwo.agent.storage.base import RunStepStorage, InMemoryRunStepStorage
from agiwo.agent.storage.session import (
    InMemorySessionStorage,
    SessionStorage,
    SQLiteSessionStorage,
)
from agiwo.agent.storage.sqlite import SQLiteRunStepStorage
from agiwo.agent.storage.mongo import MongoRunStepStorage


def _resolve_db_path(raw: str) -> str:
    if raw.strip() == ":memory:":
        return ":memory:"
    resolved = settings.resolve_path(raw)
    return str(resolved) if resolved is not None else raw


def create_run_step_storage(config: RunStepStorageConfig) -> RunStepStorage:
    storage_type = config.storage_type
    cfg = config.config

    if storage_type == "memory":
        return InMemoryRunStepStorage()
    if storage_type == "sqlite":
        db_path = cfg.get("db_path", "agiwo.db")
        return SQLiteRunStepStorage(db_path=_resolve_db_path(db_path))
    if storage_type == "mongodb":
        uri = cfg.get("mongo_uri") or cfg.get("uri", "mongodb://localhost:27017")
        db_name = cfg.get("db_name", "agiwo")
        return MongoRunStepStorage(uri=uri, db_name=db_name)
    raise ValueError(f"Unknown run_step_storage_type: {storage_type}")


def create_session_storage(config: RunStepStorageConfig) -> SessionStorage:
    storage_type = config.storage_type
    cfg = config.config
    if storage_type == "sqlite":
        db_path = cfg.get("db_path", "agiwo.db")
        return SQLiteSessionStorage(_resolve_db_path(db_path))
    return InMemorySessionStorage()


__all__ = ["create_run_step_storage", "create_session_storage"]
