"""Console storage wiring — config builders and NotifyingTraceStorage."""

import asyncio
from typing import Any

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

from server.config import ConsoleConfig


# ── NotifyingTraceStorage ────────────────────────────────────────────────────


class NotifyingTraceStorage(BaseTraceStorage):
    """Decorator that notifies subscribers whenever a trace is saved.

    SDK storage stays pure (save/get/query/close).
    Real-time notification is a Console concern for the trace SSE endpoint.
    """

    def __init__(
        self,
        inner: BaseTraceStorage,
        queue_maxsize: int = 100,
    ) -> None:
        self._inner = inner
        self._subscribers: list[asyncio.Queue[Trace]] = []
        self._queue_maxsize = queue_maxsize

    def subscribe(self) -> asyncio.Queue[Trace]:
        queue: asyncio.Queue[Trace] = asyncio.Queue(maxsize=self._queue_maxsize)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[Trace]) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    async def initialize(self) -> None:
        await self._inner.initialize()

    async def save_trace(self, trace: Trace) -> None:
        await self._inner.save_trace(trace)
        for queue in self._subscribers:
            try:
                queue.put_nowait(trace)
            except asyncio.QueueFull:
                pass

    async def get_trace(self, trace_id: str) -> Trace | None:
        return await self._inner.get_trace(trace_id)

    async def query_traces(self, query: TraceQuery | dict[str, Any]) -> list[Trace]:
        return await self._inner.query_traces(query)

    async def close(self) -> None:
        self._subscribers.clear()
        await self._inner.close()


# ── Storage config builders ──────────────────────────────────────────────────


def build_run_step_storage_config(
    console_config: ConsoleConfig,
) -> RunStepStorageConfig:
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


def create_run_step_storage(config: ConsoleConfig) -> RunStepStorage:
    return StorageFactory.create_run_step_storage(build_run_step_storage_config(config))


def create_trace_storage(config: ConsoleConfig) -> NotifyingTraceStorage:
    storage = _sdk_create_trace_storage(build_trace_storage_config(config))
    if storage is None:
        raise ValueError("Console requires trace_storage to be configured")
    return NotifyingTraceStorage(storage)
