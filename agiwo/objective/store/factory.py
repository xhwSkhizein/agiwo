"""ObjectiveStore factory. MVP: memory / sqlite only; other backends fail-closed."""

from typing import Any

from agiwo.agent.models.config import RunLogStorageConfig
from agiwo.config.settings import get_settings
from agiwo.objective.errors import UnsupportedStorageBackend
from agiwo.objective.store.base import ObjectiveStore
from agiwo.objective.store.memory import InMemoryObjectiveStore
from agiwo.objective.store.sqlite import SQLiteObjectiveStore


def _resolve_db_path(raw: str) -> str:
    if raw.strip() == ":memory:":
        return ":memory:"
    resolved = get_settings().resolve_path(raw)
    return str(resolved) if resolved is not None else raw


def create_objective_store(
    config: RunLogStorageConfig | None = None,
    *,
    storage_type: str | None = None,
    backend_config: dict[str, Any] | None = None,
) -> ObjectiveStore:
    """Create ObjectiveStore following RunLog storage configuration.

    Accepts either a ``RunLogStorageConfig`` or explicit type/config kwargs.
    Non memory/sqlite backends raise ``UnsupportedStorageBackend`` immediately.
    """
    if config is not None:
        storage_type = config.storage_type
        backend_config = config.config
    storage_type = storage_type or "memory"
    backend_config = backend_config or {}

    if storage_type == "memory":
        return InMemoryObjectiveStore()
    if storage_type == "sqlite":
        db_path = backend_config.get("db_path", "agiwo.db")
        return SQLiteObjectiveStore(db_path=_resolve_db_path(str(db_path)))

    raise UnsupportedStorageBackend(storage_type)


__all__ = ["create_objective_store"]
