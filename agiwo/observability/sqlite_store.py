"""
SQLite implementation of TraceStorage.
"""

import json
from typing import Any

import aiosqlite

from agiwo.observability.base import BaseTraceStorage, TraceQuery
from agiwo.observability.trace import Trace, Span
from agiwo.utils.storage_support.sqlite_runtime import (
    SQLiteConnectionRuntime,
    execute_statements,
)
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class SQLiteTraceStorage(BaseTraceStorage):
    """
    SQLite implementation of TraceStorage.

    Features:
    - Async SQLite operations
    - In-memory ring buffer for real-time access
    - SSE subscriber support
    """

    def __init__(
        self,
        db_path: str,
        collection_name: str,
        buffer_size: int = 200,
    ) -> None:
        self.db_path = db_path
        self.collection_name = collection_name
        self.buffer_size = buffer_size
        self._initialize_runtime_state(buffer_size=buffer_size)

        # SQLite connection (lazy init)
        self._connection: aiosqlite.Connection | None = None
        self._runtime = SQLiteConnectionRuntime(
            db_path=db_path,
            logger=logger,
            connect_event="sqlite_trace_storage_initialized",
        )

    @property
    def _initialized(self) -> bool:
        return self._runtime.initialized

    async def initialize(self) -> None:
        """Initialize SQLite connection using shared pool."""
        self._connection = await self._runtime.ensure_connection(
            self._initialize_schema
        )

    async def disconnect(self) -> None:
        """Release connection back to pool."""
        if self._connection:
            await self._runtime.disconnect()
            self._connection = None

    async def _initialize_schema(self, connection: aiosqlite.Connection) -> None:
        """Create database tables and indexes."""
        await execute_statements(
            connection,
            [
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
                    total_input_tokens INTEGER DEFAULT 0,
                    total_output_tokens INTEGER DEFAULT 0,
                    total_llm_calls INTEGER DEFAULT 0,
                    total_tool_calls INTEGER DEFAULT 0,
                    total_cache_read_tokens INTEGER DEFAULT 0,
                    total_cache_creation_tokens INTEGER DEFAULT 0,
                    total_token_cost REAL DEFAULT 0.0,
                    max_depth INTEGER DEFAULT 0,
                    input_query TEXT,
                    final_output TEXT,
                    spans TEXT
                )
                """,
                f"CREATE INDEX IF NOT EXISTS idx_traces_start_time ON {self.collection_name}(start_time)",
                f"CREATE INDEX IF NOT EXISTS idx_traces_agent_id ON {self.collection_name}(agent_id)",
                f"CREATE INDEX IF NOT EXISTS idx_traces_session_id ON {self.collection_name}(session_id)",
                f"CREATE INDEX IF NOT EXISTS idx_traces_status ON {self.collection_name}(status)",
                f"CREATE INDEX IF NOT EXISTS idx_traces_duration_ms ON {self.collection_name}(duration_ms)",
            ],
        )
        await connection.commit()

    def _serialize_trace(self, trace: Trace) -> dict:
        """Serialize Trace to dict for database storage."""
        data = trace.model_dump(mode="json", exclude_none=True)

        # Convert spans list to JSON string (always, even if empty)
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
        self._append_to_buffer(trace)

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
        except Exception as e:  # noqa: BLE001 - trace read boundary
            logger.error("trace_get_failed", trace_id=trace_id, error=str(e))

        return None

    def _build_sql_query(
        self,
        query: TraceQuery,
    ) -> tuple[str, list[str | int | float]]:
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
        if query.start_time:
            sql_query += " AND start_time >= ?"
            params.append(query.start_time.isoformat())
        if query.end_time:
            sql_query += " AND start_time <= ?"
            params.append(query.end_time.isoformat())
        if query.min_duration_ms:
            sql_query += " AND duration_ms >= ?"
            params.append(query.min_duration_ms)
        if query.max_duration_ms:
            sql_query += " AND duration_ms <= ?"
            params.append(query.max_duration_ms)

        sql_query += " ORDER BY start_time DESC LIMIT ? OFFSET ?"
        params.extend([query.limit, query.offset])
        return sql_query, params

    async def query_traces(self, query: TraceQuery | dict[str, Any]) -> list[Trace]:
        """Query traces"""
        query = self._coerce_query(query)
        if self._connection is None:
            await self.initialize()

        try:
            sql_query, params = self._build_sql_query(query)
            if self._connection is None:
                raise RuntimeError("Database connection not established")
            traces = []
            async with self._connection.execute(sql_query, params) as cursor:
                async for row in cursor:
                    traces.append(self._deserialize_trace(row))
            return traces
        except Exception as e:  # noqa: BLE001 - trace query boundary
            logger.error("trace_query_failed", error=str(e))

        # Fallback to buffer
        return self._query_buffer(query)

    async def close(self) -> None:
        """Release connection back to pool."""
        await self.disconnect()


__all__ = ["SQLiteTraceStorage"]
