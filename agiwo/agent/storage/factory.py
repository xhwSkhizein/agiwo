"""
Storage constructors for creating storage instances from configuration.

Storage instances handle their own lazy connection internally.
"""

from typing import Any

from agiwo.config.settings import get_settings
from agiwo.agent.models.config import RunLogStorageConfig
from agiwo.agent.storage.base import RunLogStorage, InMemoryRunLogStorage
from agiwo.agent.storage.sqlite import SQLiteRunLogStorage
from agiwo.utils.storage_factory import create_storage


def _resolve_db_path(raw: str) -> str:
    if raw.strip() == ":memory:":
        return ":memory:"
    resolved = get_settings().resolve_path(raw)
    return str(resolved) if resolved is not None else raw


def _make_memory_run_log(_cfg: dict[str, Any]) -> RunLogStorage:
    return InMemoryRunLogStorage()


def _make_sqlite_run_log(cfg: dict[str, Any]) -> RunLogStorage:
    db_path = cfg.get("db_path", "agiwo.db")
    return SQLiteRunLogStorage(db_path=_resolve_db_path(db_path))


_RUN_LOG_BACKENDS = {
    "memory": _make_memory_run_log,
    "sqlite": _make_sqlite_run_log,
}


def create_run_log_storage(config: RunLogStorageConfig) -> RunLogStorage:
    return create_storage(
        config.storage_type,
        config.config,
        _RUN_LOG_BACKENDS,
        label="run_log_storage",
    )


__all__ = ["create_run_log_storage"]
