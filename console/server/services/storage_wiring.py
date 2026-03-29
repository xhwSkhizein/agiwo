"""Console storage wiring — config builders."""

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


# ── Storage config builders ──────────────────────────────────────────────────


def build_run_step_storage_config(
    console_config: ConsoleConfig,
) -> RunStepStorageConfig:
    if console_config.run_step_storage_type == "sqlite":
        return RunStepStorageConfig(
            storage_type="sqlite",
            config={"db_path": console_config.sqlite_db_path},
        )
    return RunStepStorageConfig(storage_type="memory")


def build_trace_storage_config(console_config: ConsoleConfig) -> TraceStorageConfig:
    if console_config.trace_storage_type == "sqlite":
        return TraceStorageConfig(
            storage_type="sqlite",
            config={
                "db_path": console_config.sqlite_db_path,
                "collection_name": console_config.sqlite_trace_collection,
            },
        )
    return TraceStorageConfig(storage_type="memory")


def build_agent_state_storage_config(
    console_config: ConsoleConfig,
) -> AgentStateStorageConfig:
    if console_config.metadata_storage_type == "sqlite":
        return AgentStateStorageConfig(
            storage_type="sqlite",
            config={"db_path": console_config.sqlite_db_path},
        )
    return AgentStateStorageConfig(storage_type="memory")


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
