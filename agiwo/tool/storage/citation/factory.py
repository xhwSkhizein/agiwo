"""Citation store factory helpers."""

from dataclasses import dataclass, field
from typing import Literal

from agiwo.config.settings import get_settings

from .memory_store import InMemoryCitationStore
from .protocols import CitationSourceRepository
from .sqlite_store import SQLiteCitationStore

_STORE_CACHE: dict[tuple[str, str], CitationSourceRepository] = {}

_DEFAULT_CONFIG: "CitationStoreConfig | None" = None


def _get_default_config() -> "CitationStoreConfig":
    global _DEFAULT_CONFIG
    if _DEFAULT_CONFIG is None:
        _DEFAULT_CONFIG = CitationStoreConfig()
    return _DEFAULT_CONFIG


def _default_storage_type() -> Literal["memory", "sqlite"]:
    storage_type = get_settings().default_session_store
    if storage_type in {"memory", "sqlite"}:
        return storage_type
    return "memory"


def _default_sqlite_db_path() -> str:
    _s = get_settings()
    resolved = _s.resolve_path(_s.sqlite_db_path)
    if resolved is None:
        return "agiwo.db"
    return str(resolved)


@dataclass
class CitationStoreConfig:
    """Configuration for creating citation stores."""

    storage_type: Literal["memory", "sqlite"] = field(
        default_factory=_default_storage_type
    )
    sqlite_db_path: str = field(default_factory=_default_sqlite_db_path)


def create_citation_store(
    config: CitationStoreConfig | None = None,
) -> CitationSourceRepository:
    """Create a citation store from configuration."""
    effective = config or _get_default_config()
    cache_key = (effective.storage_type, effective.sqlite_db_path)
    cached = _STORE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    store: CitationSourceRepository
    if effective.storage_type == "sqlite":
        store = SQLiteCitationStore(db_path=effective.sqlite_db_path)
    else:
        store = InMemoryCitationStore()

    _STORE_CACHE[cache_key] = store
    return store


__all__ = ["CitationStoreConfig", "create_citation_store"]
