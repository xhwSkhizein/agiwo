"""
Storage manager — creates and manages storage instances from console config.

Provides storage config builders and a single access point to
RunStepStorage and BaseTraceStorage.
"""

from agiwo.agent.options import RunStepStorageConfig, TraceStorageConfig
from agiwo.agent.storage.base import RunStepStorage
from agiwo.agent.storage.factory import StorageFactory
from agiwo.observability.base import BaseTraceStorage
from agiwo.scheduler.models import AgentStateStorageConfig
from agiwo.tool.storage.citation import CitationStoreConfig

from server.config import ConsoleConfig


# ── Storage config builders ───────────────────────────────────────────


def build_run_step_storage_config(console_config: ConsoleConfig) -> RunStepStorageConfig:
    if console_config.run_step_storage_type == "sqlite":
        return RunStepStorageConfig(
            storage_type="sqlite",
            config={"db_path": console_config.sqlite_db_path},
        )
    if console_config.run_step_storage_type == "memory":
        return RunStepStorageConfig(storage_type="memory")
    return RunStepStorageConfig(
        storage_type="mongodb",
        config={
            "mongo_uri": console_config.mongodb_uri,
            "db_name": console_config.mongodb_db_name,
        },
    )


def build_trace_storage_config(console_config: ConsoleConfig) -> TraceStorageConfig:
    effective_type = console_config.effective_trace_storage_type
    if effective_type == "memory":
        return TraceStorageConfig(storage_type="memory")
    if effective_type == "sqlite":
        return TraceStorageConfig(
            storage_type="sqlite",
            config={
                "db_path": console_config.sqlite_db_path,
                "collection_name": console_config.sqlite_trace_collection,
            },
        )
    return TraceStorageConfig(
        storage_type="mongodb",
        config={
            "mongo_uri": console_config.mongodb_uri,
            "db_name": console_config.mongodb_db_name,
            "collection_name": console_config.mongodb_trace_collection,
        },
    )


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
    if console_config.metadata_storage_type == "memory":
        return CitationStoreConfig(storage_type="memory")
    return CitationStoreConfig(
        storage_type="mongodb",
        mongo_uri=console_config.mongodb_uri,
        mongo_db_name=console_config.mongodb_db_name,
        collection_name="citation_sources",
    )


# ── StorageManager ────────────────────────────────────────────────────


class StorageManager:
    """Manages lifecycle of console run-step and trace storage instances."""

    def __init__(self, config: ConsoleConfig) -> None:
        run_step_cfg = build_run_step_storage_config(config)
        trace_cfg = build_trace_storage_config(config)

        self.run_step_storage: RunStepStorage = StorageFactory.create_run_step_storage(run_step_cfg)
        self.trace_storage: BaseTraceStorage = StorageFactory.create_trace_storage(trace_cfg)

    async def close(self) -> None:
        """Close all storage connections."""
        await self.run_step_storage.close()
        if self.trace_storage is not None:
            await self.trace_storage.close()
