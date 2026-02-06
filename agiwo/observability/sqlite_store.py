"""
SQLite implementation of TraceStore.
"""

import asyncio
import json
import os
from collections import deque
from pathlib import Path
from typing import Any

import aiosqlite

from agiwo.observability.base import BaseTraceStore, TraceQuery
from agiwo.observability.trace import Trace, Span
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class SQLiteTraceStore(BaseTraceStore):
    """
    SQLite implementation of TraceStore.

    Features:
    - Async SQLite operations
    - In-memory ring buffer for real-time access
    - SSE subscriber support
    """

    def __init__(
        self,
        db_path: str = "agiwo.db",
        collection_name: str = "agiwo_traces",
        buffer_size: int = 200,
    ) -> None:
        self.db_path = os.path.expanduser(db_path)
        self.collection_name = collection_name
        self.buffer_size = buffer_size

        # In-memory ring buffer
        self._buffer: deque[Trace] = deque(maxlen=buffer_size)

        # SSE subscribers
        self._subscribers: list[asyncio.Queue] = []

        # SQLite connection (lazy init)
        self._connection: aiosqlite.Connection | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize SQLite connection"""
        if self._initialized:
            return

        db_path = Path(self.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._connection = await aiosqlite.connect(str(db_path))
        self._connection.row_factory = aiosqlite.Row

        await self._create_tables()
        self._initialized = True

        logger.info("sqlite_trace_store_initialized", db_path=str(db_path))

    async def _create_tables(self) -> None:
        """Create database tables and indexes."""
        if not self._connection:
            raise RuntimeError("Database connection not established")

        # Create traces table
        await self._connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.collection_name} (
                trace_id TEXT PRIMARY KEY,
                agent_id TEXT,
                session_id TEXT,
                user_id TEXT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                duration_ms REAL,
                status TEXT NOT NULL,
                root_span_id TEXT,
                total_tokens INTEGER DEFAULT 0,
                total_llm_calls INTEGER DEFAULT 0,
                total_tool_calls INTEGER DEFAULT 0,
                total_cache_read_tokens INTEGER DEFAULT 0,
                total_cache_creation_tokens INTEGER DEFAULT 0,
                max_depth INTEGER DEFAULT 0,
                input_query TEXT,
                final_output TEXT,
                spans TEXT
            )
        """
        )

        # Create indexes
        await self._connection.execute(
            f"CREATE INDEX IF NOT EXISTS idx_traces_start_time ON {self.collection_name}(start_time)"
        )
        await self._connection.execute(
            f"CREATE INDEX IF NOT EXISTS idx_traces_agent_id ON {self.collection_name}(agent_id)"
        )
        await self._connection.execute(
            f"CREATE INDEX IF NOT EXISTS idx_traces_session_id ON {self.collection_name}(session_id)"
        )
        await self._connection.execute(
            f"CREATE INDEX IF NOT EXISTS idx_traces_status ON {self.collection_name}(status)"
        )
        await self._connection.execute(
            f"CREATE INDEX IF NOT EXISTS idx_traces_duration_ms ON {self.collection_name}(duration_ms)"
        )

        await self._connection.commit()

        # Migrate existing tables to add missing columns
        await self._migrate_tables()

    async def _migrate_tables(self) -> None:
        """Migrate existing tables to add missing columns if needed."""
        if not self._connection:
            return

        try:
            # Check if total_cache_read_tokens column exists
            async with self._connection.execute(
                f"PRAGMA table_info({self.collection_name})"
            ) as cursor:
                columns = [row[1] for row in await cursor.fetchall()]

            # Add missing columns if they don't exist
            if "total_cache_read_tokens" not in columns:
                await self._connection.execute(
                    f"ALTER TABLE {self.collection_name} ADD COLUMN total_cache_read_tokens INTEGER DEFAULT 0"
                )
            if "total_cache_creation_tokens" not in columns:
                await self._connection.execute(
                    f"ALTER TABLE {self.collection_name} ADD COLUMN total_cache_creation_tokens INTEGER DEFAULT 0"
                )

            await self._connection.commit()
        except Exception as e:
            # Ignore errors if columns already exist or table doesn't exist
            logger.debug("migration_skipped", error=str(e))

    def _serialize_trace(self, trace: Trace) -> dict:
        """Serialize Trace to dict for database storage."""
        data = trace.model_dump(mode="json", exclude_none=True)

        # Convert spans list to JSON string
        if "spans" in data and data["spans"]:
            data["spans"] = json.dumps(
                [span.model_dump(mode="json") for span in trace.spans]
            )

        # Convert datetime to ISO format string
        if "start_time" in data and isinstance(data["start_time"], str) is False:
            data["start_time"] = data["start_time"].isoformat()
        if "end_time" in data and isinstance(data["end_time"], str) is False:
            data["end_time"] = data["end_time"].isoformat()

        return data

    def _deserialize_trace(self, row: aiosqlite.Row) -> Trace:
        """Deserialize database row to Trace."""
        data = dict(row)

        # Parse spans JSON string
        if data.get("spans"):
            spans_data = json.loads(data["spans"])

            data["spans"] = [Span.model_validate(span) for span in spans_data]
        else:
            data["spans"] = []

        return Trace.model_validate(data)

    async def save_trace(self, trace: Trace) -> None:
        """Save trace"""
        # Add to buffer
        self._buffer.append(trace)

        # Persist to SQLite
        if self._connection is None:
            await self.initialize()

        try:
            data = self._serialize_trace(trace)

            columns = ", ".join(data.keys())
            placeholders = ", ".join(["?" for _ in data])
            values = list(data.values())

            query = f"""
                INSERT OR REPLACE INTO {self.collection_name} ({columns})
                VALUES ({placeholders})
            """

            if self._connection is None:
                raise RuntimeError("Database connection not established")
            await self._connection.execute(query, values)
            await self._connection.commit()
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

        # Query SQLite
        if self._connection is None:
            await self.initialize()

        try:
            if self._connection is None:
                raise RuntimeError("Database connection not established")
            async with self._connection.execute(
                f"SELECT * FROM {self.collection_name} WHERE trace_id = ?",
                (trace_id,),
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._deserialize_trace(row)
        except Exception as e:
            logger.error("trace_get_failed", trace_id=trace_id, error=str(e))

        return None

    async def query_traces(self, query: TraceQuery | dict[str, Any]) -> list[Trace]:
        """Query traces"""
        query = self._coerce_query(query)
        if self._connection is None:
            await self.initialize()

        try:
            sql_query = f"SELECT * FROM {self.collection_name} WHERE 1=1"
            params: list[str | int | float] = []

            if query.agent_id:
                sql_query += " AND agent_id = ?"
                params.append(query.agent_id)
            if query.session_id:
                sql_query += " AND session_id = ?"
                params.append(query.session_id)
            if query.user_id:
                sql_query += " AND user_id = ?"
                params.append(query.user_id)
            if query.status:
                sql_query += " AND status = ?"
                params.append(query.status.value)

            # Time range
            if query.start_time:
                sql_query += " AND start_time >= ?"
                params.append(query.start_time.isoformat())
            if query.end_time:
                sql_query += " AND start_time <= ?"
                params.append(query.end_time.isoformat())

            # Duration range
            if query.min_duration_ms:
                sql_query += " AND duration_ms >= ?"
                params.append(query.min_duration_ms)
            if query.max_duration_ms:
                sql_query += " AND duration_ms <= ?"
                params.append(query.max_duration_ms)

            sql_query += " ORDER BY start_time DESC LIMIT ? OFFSET ?"
            params.extend([query.limit, query.offset])

            if self._connection is None:
                raise RuntimeError("Database connection not established")
            traces = []
            async with self._connection.execute(sql_query, params) as cursor:
                async for row in cursor:
                    traces.append(self._deserialize_trace(row))
            return traces
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
        """Close SQLite connection"""
        if self._connection:
            await self._connection.close()
            self._connection = None
            self._initialized = False


__all__ = ["SQLiteTraceStore"]
