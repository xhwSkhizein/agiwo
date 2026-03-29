"""Citation store factory helpers."""

from dataclasses import dataclass, field
from typing import Literal

from agiwo.config.settings import get_settings
from agiwo.utils.logging import get_logger

from .memory_store import InMemoryCitationStore
from .mongo_store import MongoCitationStore
from .protocols import CitationSourceRepository
from .sqlite_store import SQLiteCitationStore

logger = get_logger(__name__)
_STORE_CACHE: dict[tuple[str, str, str, str, str], CitationSourceRepository] = {}

# Global singleton for default config to ensure all tools share the same store
_DEFAULT_CONFIG: "CitationStoreConfig | None" = None


def _get_default_config() -> "CitationStoreConfig":
    """Get or create the singleton default CitationStoreConfig."""
    global _DEFAULT_CONFIG
    if _DEFAULT_CONFIG is None:
        _DEFAULT_CONFIG = CitationStoreConfig()
    return _DEFAULT_CONFIG


def _default_storage_type() -> Literal["memory", "sqlite", "mongodb"]:
    storage_type = get_settings().default_session_store
    if storage_type in {"memory", "sqlite", "mongodb"}:
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

    storage_type: Literal["memory", "sqlite", "mongodb"] = field(
        default_factory=_default_storage_type
    )
    sqlite_db_path: str = field(default_factory=_default_sqlite_db_path)
    mongo_uri: str | None = field(default_factory=lambda: get_settings().mongo_uri)
    mongo_db_name: str = field(
        default_factory=lambda: get_settings().mongo_db_name or "agiwo"
    )
    collection_name: str = "citation_sources"


def create_citation_store(
    config: CitationStoreConfig | None = None,
) -> CitationSourceRepository:
    """Create a citation store from configuration."""
    effective = config or _get_default_config()
    cache_key = (
        effective.storage_type,
        effective.sqlite_db_path,
        effective.mongo_uri or "",
        effective.mongo_db_name,
        effective.collection_name,
    )
    cached = _STORE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if effective.storage_type == "sqlite":
        store: CitationSourceRepository = SQLiteCitationStore(
            db_path=effective.sqlite_db_path
        )
        _STORE_CACHE[cache_key] = store
        return store

    if effective.storage_type == "mongodb":
        if not effective.mongo_uri:
            logger.warning(
                "citation_store_missing_mongo_uri_fallback_memory",
                storage_type=effective.storage_type,
            )
            store = InMemoryCitationStore()
            _STORE_CACHE[cache_key] = store
            return store
        store = MongoCitationStore(
            uri=effective.mongo_uri,
            db_name=effective.mongo_db_name,
            collection_name=effective.collection_name,
        )
        _STORE_CACHE[cache_key] = store
        return store

    store = InMemoryCitationStore()
    _STORE_CACHE[cache_key] = store
    return store


__all__ = ["CitationStoreConfig", "create_citation_store"]
