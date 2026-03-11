"""
MongoDB implementation of trace storage.
"""

from typing import Any

from motor.motor_asyncio import AsyncIOMotorCollection

from agiwo.observability.base import BaseTraceStorage, TraceQuery
from agiwo.observability.trace import Trace
from agiwo.utils.storage_support.mongo_runtime import (
    MongoCollectionRuntime,
    MongoIndexSpec,
)
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
        self._initialize_runtime_state(buffer_size=buffer_size)

        # MongoDB client (lazy init)
        self._client: Any = None
        self._collection: AsyncIOMotorCollection[Any] | None = None
        self._initialized = False
        self._runtime = (
            MongoCollectionRuntime(
                uri=mongo_uri,
                db_name=db_name,
                logger=logger,
                connect_event="trace_storage_initialized",
                disconnect_event="trace_storage_closed",
            )
            if mongo_uri
            else None
        )

    async def initialize(self) -> None:
        """Initialize MongoDB connection"""
        if self._initialized:
            return

        if self.mongo_uri and self._runtime is not None:
            try:
                self._collection = await self._runtime.ensure_collection(
                    self.collection_name,
                    indexes=[
                        MongoIndexSpec("trace_id", unique=True),
                        MongoIndexSpec("start_time"),
                        MongoIndexSpec("agent_id"),
                        MongoIndexSpec("session_id"),
                        MongoIndexSpec("status"),
                        MongoIndexSpec("duration_ms"),
                    ],
                )
                self._client = self._runtime.client
            except ImportError:
                logger.warning(
                    "motor_not_installed",
                    message="MongoDB tracing disabled. Install motor: pip install motor",
                )
            except Exception as e:  # noqa: BLE001 - trace storage init boundary
                logger.error("trace_storage_init_failed", error=str(e))

        self._initialized = True

    async def disconnect(self) -> None:
        if self._client is not None and self._runtime is not None:
            await self._runtime.disconnect()
            self._client = None
            self._collection = None
            self._initialized = False

    async def save_trace(self, trace: Trace) -> None:
        """Save trace"""
        # Add to buffer
        self._append_to_buffer(trace)

        # Persist to MongoDB
        if self._collection is not None:
            try:
                doc = trace.model_dump(mode="json", exclude_none=True)
                await self._collection.replace_one(
                    {"trace_id": trace.trace_id},
                    doc,
                    upsert=True,
                )
            except Exception as e:  # noqa: BLE001 - trace persistence boundary
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
        buffered = self._get_buffered_trace(trace_id)
        if buffered is not None:
            return buffered

        # Query MongoDB
        if self._collection is not None:
            try:
                doc = await self._collection.find_one({"trace_id": trace_id})
                if doc:
                    return Trace(**doc)
            except Exception as e:  # noqa: BLE001 - trace read boundary
                logger.error("trace_get_failed", trace_id=trace_id, error=str(e))

        return None

    def _build_mongo_query(self, query: TraceQuery) -> dict[str, Any]:
        mongo_query: dict[str, Any] = {}

        for field, value in (
            ("agent_id", query.agent_id),
            ("session_id", query.session_id),
            ("user_id", query.user_id),
        ):
            if value:
                mongo_query[field] = value
        if query.status:
            mongo_query["status"] = query.status.value

        self._apply_range_filter(
            mongo_query,
            "start_time",
            query.start_time.isoformat() if query.start_time else None,
            query.end_time.isoformat() if query.end_time else None,
        )
        self._apply_range_filter(
            mongo_query,
            "duration_ms",
            query.min_duration_ms,
            query.max_duration_ms,
        )

        return mongo_query

    def _apply_range_filter(
        self,
        query: dict[str, Any],
        field: str,
        lower: str | float | None,
        upper: str | float | None,
    ) -> None:
        if lower is None and upper is None:
            return

        query[field] = {}
        if lower is not None:
            query[field]["$gte"] = lower
        if upper is not None:
            query[field]["$lte"] = upper

    async def query_traces(self, query: TraceQuery | dict[str, Any]) -> list[Trace]:
        """Query traces"""
        query = self._coerce_query(query)
        mongo_query = self._build_mongo_query(query)

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
            except Exception as e:  # noqa: BLE001 - trace query boundary
                logger.error("trace_query_failed", error=str(e))

        # Fallback to buffer
        return self._query_buffer(query)

    async def close(self) -> None:
        """Close MongoDB connection"""
        if self._client is not None:
            await self.disconnect()


__all__ = ["MongoTraceStorage"]
