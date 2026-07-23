"""SessionIntent store constructors aligned with RunLog storage config."""

from typing import Any

from agiwo.agent.intent.base import InMemorySessionIntentStore, SessionIntentStore
from agiwo.agent.intent.sqlite import SQLiteSessionIntentStore
from agiwo.agent.models.config import RunLogStorageConfig
from agiwo.agent.storage.factory import _resolve_db_path
from agiwo.utils.storage_factory import create_storage


def _make_memory_intent(_cfg: dict[str, Any]) -> SessionIntentStore:
    return InMemorySessionIntentStore()


def _make_sqlite_intent(cfg: dict[str, Any]) -> SessionIntentStore:
    db_path = cfg.get("db_path", "agiwo.db")
    return SQLiteSessionIntentStore(db_path=_resolve_db_path(db_path))


_SESSION_INTENT_BACKENDS = {
    "memory": _make_memory_intent,
    "sqlite": _make_sqlite_intent,
}


def create_session_intent_store(
    config: RunLogStorageConfig,
) -> SessionIntentStore:
    """Create a SessionIntent store using the RunLog storage backend selector."""
    return create_storage(
        config.storage_type,
        config.config,
        _SESSION_INTENT_BACKENDS,
        label="session_intent_store",
    )


__all__ = ["create_session_intent_store"]
