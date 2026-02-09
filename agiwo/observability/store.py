"""
MongoDB implementation of trace storage.
"""

import asyncio
from collections import deque
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from agiwo.observability.base import BaseTraceStorage, TraceQuery
from agiwo.observability.trace import Trace
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class MongoTraceStorage(BaseTraceStorage):
    """
    MongoDB implementation of BaseTraceStorage.

    Features:
    - Async MongoDB operations
    - In-memory ring buffer for real-time access
    - SSE subscriber support
    """

    def __init__(
        self,
        mongo_uri: str | None = None,
        db_name: str = "agiwo",
        collection_name: str = "traces",
        buffer_size: int = 200,
    ) -> None:
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.buffer_size = buffer_size

        # In-memory ring buffer
        self._buffer: deque[Trace] = deque(maxlen=buffer_size)

        # SSE subscribers
        self._subscribers: list[asyncio.Queue] = []

        # MongoDB client (lazy init)
        self._client: AsyncIOMotorClient[Any] | None = None
        self._collection: AsyncIOMotorCollection[Any] | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize MongoDB connection"""
        if self._initialized:
            return

        if self.mongo_uri:
            try:
                self._client = AsyncIOMotorClient(self.mongo_uri)
                if self._client is None:
                    raise RuntimeError("Failed to create MongoDB client")
                db = self._client[self.db_name]
                self._collection = db[self.collection_name]
                if self._collection is None:
                    raise RuntimeError("Failed to get MongoDB collection")

                # Create indexes
                await self._collection.create_index("trace_id", unique=True)
                await self._collection.create_index("start_time")
                await self._collection.create_index("agent_id")
                await self._collection.create_index("session_id")
                await self._collection.create_index("status")
                await self._collection.create_index("duration_ms")

                logger.info(
                    "trace_storage_initialized",
                    db=self.db_name,
                    collection=self.collection_name,
                )
            except ImportError:
                logger.warning(
                    "motor_not_installed",
                    message="MongoDB tracing disabled. Install motor: pip install motor",
                )
            except Exception as e:
                logger.error("trace_storage_init_failed", error=str(e))

        self._initialized = True

    async def save_trace(self, trace: Trace) -> None:
        """Save trace"""
        # Add to buffer
        self._buffer.append(trace)

        # Persist to MongoDB
        if self._collection is not None:
            try:
                doc = trace.model_dump(mode="json", exclude_none=True)
                await self._collection.replace_one(
                    {"trace_id": trace.trace_id},
                    doc,
                    upsert=True,
                )
            except Exception as e:
                logger.error(
                    "trace_persist_failed",
                    trace_id=trace.trace_id,
                    error=str(e),
                )

        # Notify subscribers
        await self._notify_subscribers(trace)

    async def get_trace(self, trace_id: str) -> Trace | None:
        """Get single trace"""
        # Check buffer first
        for trace in self._buffer:
            if trace.trace_id == trace_id:
                return trace

        # Query MongoDB
        if self._collection is not None:
            try:
                doc = await self._collection.find_one({"trace_id": trace_id})
                if doc:
                    return Trace(**doc)
            except Exception as e:
                logger.error("trace_get_failed", trace_id=trace_id, error=str(e))

        return None

    async def query_traces(self, query: TraceQuery | dict[str, Any]) -> list[Trace]:
        """Query traces"""
        query = self._coerce_query(query)
        mongo_query: dict[str, Any] = {}

        if query.agent_id:
            mongo_query["agent_id"] = query.agent_id
        if query.session_id:
            mongo_query["session_id"] = query.session_id
        if query.user_id:
            mongo_query["user_id"] = query.user_id
        if query.status:
            mongo_query["status"] = query.status.value

        # Time range
        if query.start_time or query.end_time:
            mongo_query["start_time"] = {}
            if query.start_time:
                mongo_query["start_time"]["$gte"] = query.start_time.isoformat()
            if query.end_time:
                mongo_query["start_time"]["$lte"] = query.end_time.isoformat()

        # Duration range
        if query.min_duration_ms or query.max_duration_ms:
            mongo_query["duration_ms"] = {}
            if query.min_duration_ms:
                mongo_query["duration_ms"]["$gte"] = query.min_duration_ms
            if query.max_duration_ms:
                mongo_query["duration_ms"]["$lte"] = query.max_duration_ms

        # Query MongoDB
        if self._collection is not None:
            try:
                cursor = (
                    self._collection.find(mongo_query)
                    .sort("start_time", -1)
                    .skip(query.offset)
                    .limit(query.limit)
                )
                docs = await cursor.to_list(length=query.limit)
                return [Trace(**doc) for doc in docs]
            except Exception as e:
                logger.error("trace_query_failed", error=str(e))

        # Fallback to buffer
        return self._query_buffer(query)

    def _coerce_query(self, query: TraceQuery | dict[str, Any]) -> TraceQuery:
        if isinstance(query, TraceQuery):
            return query
        return TraceQuery(**query)

    def _query_buffer(self, query: TraceQuery) -> list[Trace]:
        """Query from in-memory buffer"""
        results = []
        for trace in reversed(self._buffer):
            if query.agent_id and trace.agent_id != query.agent_id:
                continue
            if query.session_id and trace.session_id != query.session_id:
                continue
            if query.status and trace.status != query.status:
                continue
            results.append(trace)

        start = query.offset
        end = start + query.limit
        return results[start:end]

    def get_recent(self, limit: int = 20) -> list[Trace]:
        """Get recent traces"""
        return list(reversed(list(self._buffer)))[:limit]

    def subscribe(self) -> asyncio.Queue:
        """Subscribe to real-time trace updates"""
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Unsubscribe from updates"""
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    async def _notify_subscribers(self, trace: Trace) -> None:
        """Notify all subscribers"""
        for queue in self._subscribers:
            try:
                queue.put_nowait(trace)
            except asyncio.QueueFull:
                pass

    async def close(self) -> None:
        """Close MongoDB connection"""
        if self._client:
            self._client.close()
            self._client = None
            self._collection = None
            self._initialized = False
            logger.info("trace_storage_closed")


__all__ = ["MongoTraceStorage"]
