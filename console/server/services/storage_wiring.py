"""Console storage wiring — config builders."""

from agiwo.agent import RunLogStorageConfig, TraceStorageConfig
from agiwo.agent.storage.base import RunLogStorage
from agiwo.agent.storage.factory import (
    create_run_log_storage as _create_run_log_storage,
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


def build_run_log_storage_config(
    console_config: ConsoleConfig,
) -> RunLogStorageConfig:
    if console_config.storage.run_log_type == "sqlite":
        return RunLogStorageConfig(
            storage_type="sqlite",
            config={"db_path": console_config.sqlite_db_path},
        )
    return RunLogStorageConfig(storage_type="memory")


def build_trace_storage_config(console_config: ConsoleConfig) -> TraceStorageConfig:
    if console_config.storage.trace_type == "sqlite":
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
    if console_config.storage.metadata_type == "sqlite":
        return AgentStateStorageConfig(
            storage_type="sqlite",
            config={"db_path": console_config.sqlite_db_path},
        )
    return AgentStateStorageConfig(storage_type="memory")


def build_citation_store_config(console_config: ConsoleConfig) -> CitationStoreConfig:
    if console_config.storage.metadata_type == "sqlite":
        return CitationStoreConfig(
            storage_type="sqlite",
            sqlite_db_path=console_config.sqlite_db_path,
        )
    return CitationStoreConfig(storage_type="memory")


def create_run_log_storage(config: ConsoleConfig) -> RunLogStorage:
    return _create_run_log_storage(build_run_log_storage_config(config))


def create_trace_storage(config: ConsoleConfig) -> BaseTraceStorage:
    storage = _sdk_create_trace_storage(build_trace_storage_config(config))
    if storage is None:
        raise ValueError("Console requires trace_storage to be configured")
    return storage
