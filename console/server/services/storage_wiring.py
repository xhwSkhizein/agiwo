"""Console storage wiring — config builders."""

from typing import Any, Callable, TypeVar

from agiwo.agent.options import RunStepStorageConfig, TraceStorageConfig
from agiwo.agent.storage.base import RunStepStorage
from agiwo.agent.storage.factory import StorageFactory
from agiwo.observability.base import BaseTraceStorage, TraceQuery
from agiwo.observability.factory import (
    create_trace_storage as _sdk_create_trace_storage,
)
from agiwo.observability.trace import Trace
from agiwo.scheduler.models import AgentStateStorageConfig
from agiwo.tool.storage.citation import CitationStoreConfig

from agiwo.utils.logging import get_logger

from server.config import ConsoleConfig

logger = get_logger(__name__)

T = TypeVar("T")


# ── Storage config builders ──────────────────────────────────────────────────


def _build_storage_config(
    config_class: type[T],
    storage_type: str,
    sqlite_builder: Callable[[], dict[str, Any]] | None = None,
    mongo_builder: Callable[[], dict[str, Any]] | None = None,
) -> T:
    """通用存储配置构建函数，消除重复代码。"""
    if storage_type == "sqlite":
        if sqlite_builder is None:
            raise ValueError(f"{config_class.__name__} does not support sqlite storage")
        return config_class(storage_type="sqlite", config=sqlite_builder())
    if storage_type == "memory":
        return config_class(storage_type="memory")
    if mongo_builder is None:
        raise ValueError(f"{config_class.__name__} does not support mongodb storage")
    return config_class(storage_type="mongodb", config=mongo_builder())


def build_run_step_storage_config(console_config: ConsoleConfig) -> RunStepStorageConfig:
    return _build_storage_config(
        RunStepStorageConfig,
        console_config.run_step_storage_type,
        sqlite_builder=lambda: {"db_path": console_config.sqlite_db_path},
        mongo_builder=lambda: {
            "mongo_uri": console_config.mongodb_uri,
            "db_name": console_config.mongodb_db_name,
        },
    )


def build_trace_storage_config(console_config: ConsoleConfig) -> TraceStorageConfig:
    effective_type = console_config.effective_trace_storage_type
    return _build_storage_config(
        TraceStorageConfig,
        effective_type,
        sqlite_builder=lambda: {
            "db_path": console_config.sqlite_db_path,
            "collection_name": console_config.sqlite_trace_collection,
        },
        mongo_builder=lambda: {
            "mongo_uri": console_config.mongodb_uri,
            "db_name": console_config.mongodb_db_name,
            "collection_name": console_config.mongodb_trace_collection,
        },
    )


def build_agent_state_storage_config(
    console_config: ConsoleConfig,
) -> AgentStateStorageConfig:
    storage_type = console_config.metadata_storage_type
    if storage_type == "mongodb":
        logger.warning(
            "agent_state_storage_mongodb_fallback",
            reason="AgentStateStorageConfig does not support mongodb, falling back to memory",
        )
        storage_type = "memory"
    
    return _build_storage_config(
        AgentStateStorageConfig,
        storage_type,
        sqlite_builder=lambda: {"db_path": console_config.sqlite_db_path},
    )


def build_citation_store_config(console_config: ConsoleConfig) -> CitationStoreConfig:
    if console_config.metadata_storage_type == "sqlite":
        return CitationStoreConfig(
            storage_type="sqlite",
            sqlite_db_path=console_config.sqlite_db_path,
        )
    if console_config.metadata_storage_type == "memory":
        return CitationStoreConfig(storage_type="memory")
    return CitationStoreConfig(
        storage_type="mongodb",
        mongo_uri=console_config.mongodb_uri,
        mongo_db_name=console_config.mongodb_db_name,
        collection_name="citation_sources",
    )


def create_run_step_storage(config: ConsoleConfig) -> RunStepStorage:
    return StorageFactory.create_run_step_storage(build_run_step_storage_config(config))


def create_trace_storage(config: ConsoleConfig) -> BaseTraceStorage:
    storage = _sdk_create_trace_storage(build_trace_storage_config(config))
    if storage is None:
        raise ValueError("Console requires trace_storage to be configured")
    return storage
