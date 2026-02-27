"""
Storage manager — creates and manages storage instances from console config.

Provides a single access point to RunStepStorage and BaseTraceStorage.
"""

from agiwo.agent.options import RunStepStorageConfig, TraceStorageConfig
from agiwo.agent.storage.base import RunStepStorage
from agiwo.agent.storage.factory import StorageFactory
from agiwo.observability.base import BaseTraceStorage
from agiwo.scheduler.models import AgentStateStorageConfig
from agiwo.scheduler.store import AgentStateStorage, create_agent_state_storage

from server.config import ConsoleConfig


class StorageManager:
    """Manages lifecycle of storage instances for the console server."""

    def __init__(self, config: ConsoleConfig) -> None:
        self._config = config
        run_step_cfg = self._build_run_step_config()
        trace_cfg = self._build_trace_config()

        self.run_step_storage: RunStepStorage = StorageFactory.create_run_step_storage(run_step_cfg)
        self.trace_storage: BaseTraceStorage = StorageFactory.create_trace_storage(trace_cfg)
        self.agent_state_storage: AgentStateStorage = self._build_agent_state_storage()

    async def close(self) -> None:
        """Close all storage connections."""
        await self.run_step_storage.close()
        if self.trace_storage is not None:
            await self.trace_storage.close()
        await self.agent_state_storage.close()

    def _build_run_step_config(self) -> RunStepStorageConfig:
        if self._config.storage_type == "sqlite":
            return RunStepStorageConfig(
                storage_type="sqlite",
                config={"db_path": self._config.sqlite_db_path},
            )
        return RunStepStorageConfig(
            storage_type="mongodb",
            config={
                "uri": self._config.mongodb_uri,
                "db_name": self._config.mongodb_db_name,
            },
        )

    def _build_agent_state_storage(self) -> AgentStateStorage:
        if self._config.storage_type == "sqlite":
            cfg = AgentStateStorageConfig(
                storage_type="sqlite",
                config={"db_path": self._config.sqlite_db_path},
            )
        else:
            cfg = AgentStateStorageConfig(storage_type="memory")
        return create_agent_state_storage(cfg)

    def _build_trace_config(self) -> TraceStorageConfig:
        if self._config.storage_type == "sqlite":
            return TraceStorageConfig(
                storage_type="sqlite",
                config={
                    "db_path": self._config.sqlite_db_path,
                    "collection_name": self._config.sqlite_trace_collection,
                },
            )
        return TraceStorageConfig(
            storage_type="mongodb",
            config={
                "mongo_uri": self._config.mongodb_uri,
                "db_name": self._config.mongodb_db_name,
                "collection_name": self._config.mongodb_trace_collection,
            },
        )
