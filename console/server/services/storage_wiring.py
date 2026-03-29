"""Console storage wiring — config builders."""

from typing import Any, Callable, TypeVar

from agiwo.agent import RunStepStorageConfig, TraceStorageConfig
from agiwo.agent.storage.base import RunStepStorage
from agiwo.agent.storage.factory import (
    create_run_step_storage as _create_run_step_storage,
)
from agiwo.observability.base import BaseTraceStorage
from agiwo.observability.factory import (
    create_trace_storage as _sdk_create_trace_storage,
)
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
) -> T:
    if storage_type == "sqlite":
        if sqlite_builder is None:
            raise ValueError(f"{config_class.__name__} does not support sqlite storage")
        return config_class(storage_type="sqlite", config=sqlite_builder())
    if storage_type == "memory":
        return config_class(storage_type="memory")
    raise ValueError(
        f"{config_class.__name__} does not support storage type: {storage_type}"
    )


def build_run_step_storage_config(
    console_config: ConsoleConfig,
) -> RunStepStorageConfig:
    return _build_storage_config(
        RunStepStorageConfig,
        console_config.run_step_storage_type,
        sqlite_builder=lambda: {"db_path": console_config.sqlite_db_path},
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
    )


def build_agent_state_storage_config(
    console_config: ConsoleConfig,
) -> AgentStateStorageConfig:
    return _build_storage_config(
        AgentStateStorageConfig,
        console_config.metadata_storage_type,
        sqlite_builder=lambda: {"db_path": console_config.sqlite_db_path},
    )


def build_citation_store_config(console_config: ConsoleConfig) -> CitationStoreConfig:
    if console_config.metadata_storage_type == "sqlite":
        return CitationStoreConfig(
            storage_type="sqlite",
            sqlite_db_path=console_config.sqlite_db_path,
        )
    return CitationStoreConfig(storage_type="memory")


def create_run_step_storage(config: ConsoleConfig) -> RunStepStorage:
    return _create_run_step_storage(build_run_step_storage_config(config))


def create_trace_storage(config: ConsoleConfig) -> BaseTraceStorage:
    storage = _sdk_create_trace_storage(build_trace_storage_config(config))
    if storage is None:
        raise ValueError("Console requires trace_storage to be configured")
    return storage
