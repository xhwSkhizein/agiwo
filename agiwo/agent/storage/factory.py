"""
Storage constructors for creating storage instances from configuration.

Storage instances handle their own lazy connection internally.
"""

from typing import Any

from agiwo.config.settings import get_settings
from agiwo.agent.models.config import RunStepStorageConfig
from agiwo.agent.storage.base import RunStepStorage, InMemoryRunStepStorage
from agiwo.agent.storage.sqlite import SQLiteRunStepStorage
from agiwo.utils.storage_factory import create_storage


def _resolve_db_path(raw: str) -> str:
    if raw.strip() == ":memory:":
        return ":memory:"
    resolved = get_settings().resolve_path(raw)
    return str(resolved) if resolved is not None else raw


def _make_memory_run_step(_cfg: dict[str, Any]) -> RunStepStorage:
    return InMemoryRunStepStorage()


def _make_sqlite_run_step(cfg: dict[str, Any]) -> RunStepStorage:
    db_path = cfg.get("db_path", "agiwo.db")
    return SQLiteRunStepStorage(db_path=_resolve_db_path(db_path))


_RUN_STEP_BACKENDS = {
    "memory": _make_memory_run_step,
    "sqlite": _make_sqlite_run_step,
}


def create_run_step_storage(config: RunStepStorageConfig) -> RunStepStorage:
    return create_storage(
        config.storage_type,
        config.config,
        _RUN_STEP_BACKENDS,
        label="run_step_storage",
    )


__all__ = ["create_run_step_storage"]
