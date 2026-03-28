"""
Session Storage - Session-level data persistence (CompactMetadata).
"""

import asyncio
import json
from abc import ABC, abstractmethod
from datetime import datetime

import aiosqlite

from agiwo.agent.models.compact import CompactMetadata
from agiwo.utils.storage_support.sqlite_runtime import (
    SQLiteConnectionRuntime,
    execute_statements,
)
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class SessionStorage(ABC):
    """
    Session-level storage interface.

    Manages session-scoped data like CompactMetadata.
    Data is isolated by (session_id, agent_id) pair.
    """

    async def close(self) -> None:
        """Close storage and release resources (optional)."""
        pass

    @abstractmethod
    async def save_compact_metadata(
        self, session_id: str, agent_id: str, metadata: CompactMetadata
    ) -> None:
        """Save compact metadata (append to history)."""
        ...

    @abstractmethod
    async def get_latest_compact_metadata(
        self, session_id: str, agent_id: str
    ) -> CompactMetadata | None:
        """Get the most recent compact metadata."""
        ...

    @abstractmethod
    async def get_compact_history(
        self, session_id: str, agent_id: str
    ) -> list[CompactMetadata]:
        """Get all compact metadata history (sorted by created_at ascending)."""
        ...


class InMemorySessionStorage(SessionStorage):
    """In-memory implementation for testing and development."""

    def __init__(self) -> None:
        self._compact_history: dict[str, list[CompactMetadata]] = {}
        self._lock = asyncio.Lock()

    def _key(self, session_id: str, agent_id: str) -> str:
        return f"{session_id}:{agent_id}"

    async def save_compact_metadata(
        self, session_id: str, agent_id: str, metadata: CompactMetadata
    ) -> None:
        async with self._lock:
            key = self._key(session_id, agent_id)
            if key not in self._compact_history:
                self._compact_history[key] = []
            self._compact_history[key].append(metadata)

    async def get_latest_compact_metadata(
        self, session_id: str, agent_id: str
    ) -> CompactMetadata | None:
        key = self._key(session_id, agent_id)
        history = self._compact_history.get(key, [])
        return history[-1] if history else None

    async def get_compact_history(
        self, session_id: str, agent_id: str
    ) -> list[CompactMetadata]:
        key = self._key(session_id, agent_id)
        return list(self._compact_history.get(key, []))


class SQLiteSessionStorage(SessionStorage):
    """SQLite implementation for persistent storage."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        self._runtime = SQLiteConnectionRuntime(
            db_path=db_path,
            logger=logger,
            connect_event="sqlite_session_storage_connected",
        )

    async def _ensure_connection(self) -> aiosqlite.Connection:
        self._conn = await self._runtime.ensure_connection(self._initialize_schema)
        self._conn.row_factory = aiosqlite.Row
        return self._conn

    async def close(self) -> None:
        if self._conn is not None:
            await self._runtime.disconnect()
            self._conn = None

    async def _initialize_schema(self, connection: aiosqlite.Connection) -> None:
        await execute_statements(
            connection,
            [
                """
                CREATE TABLE IF NOT EXISTS compact_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    start_seq INTEGER NOT NULL,
                    end_seq INTEGER NOT NULL,
                    before_token_estimate INTEGER NOT NULL,
                    after_token_estimate INTEGER NOT NULL,
                    message_count INTEGER NOT NULL,
                    transcript_path TEXT NOT NULL,
                    analysis TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    compact_model TEXT NOT NULL,
                    compact_tokens INTEGER NOT NULL
                )
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_compact_session_agent
                ON compact_metadata (session_id, agent_id, created_at)
                """,
            ],
        )
        await connection.commit()

    async def save_compact_metadata(
        self, session_id: str, agent_id: str, metadata: CompactMetadata
    ) -> None:
        async with self._lock:
            conn = await self._ensure_connection()
            await conn.execute(
                """
                INSERT INTO compact_metadata (
                    session_id, agent_id, start_seq, end_seq,
                    before_token_estimate, after_token_estimate, message_count,
                    transcript_path, analysis, created_at, compact_model, compact_tokens
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata.session_id,
                    metadata.agent_id,
                    metadata.start_seq,
                    metadata.end_seq,
                    metadata.before_token_estimate,
                    metadata.after_token_estimate,
                    metadata.message_count,
                    metadata.transcript_path,
                    json.dumps(metadata.analysis, ensure_ascii=False, default=str),
                    metadata.created_at.isoformat(),
                    metadata.compact_model,
                    metadata.compact_tokens,
                ),
            )
            await conn.commit()

    async def get_latest_compact_metadata(
        self, session_id: str, agent_id: str
    ) -> CompactMetadata | None:
        conn = await self._ensure_connection()
        async with conn.execute(
            """
            SELECT * FROM compact_metadata
            WHERE session_id = ? AND agent_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (session_id, agent_id),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_metadata(row)

    async def get_compact_history(
        self, session_id: str, agent_id: str
    ) -> list[CompactMetadata]:
        conn = await self._ensure_connection()
        async with conn.execute(
            """
            SELECT * FROM compact_metadata
            WHERE session_id = ? AND agent_id = ?
            ORDER BY created_at ASC
            """,
            (session_id, agent_id),
        ) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_metadata(row) for row in rows]

    def _row_to_metadata(self, row: aiosqlite.Row) -> CompactMetadata:
        return CompactMetadata(
            session_id=row["session_id"],
            agent_id=row["agent_id"],
            start_seq=row["start_seq"],
            end_seq=row["end_seq"],
            before_token_estimate=row["before_token_estimate"],
            after_token_estimate=row["after_token_estimate"],
            message_count=row["message_count"],
            transcript_path=row["transcript_path"],
            analysis=json.loads(row["analysis"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            compact_model=row["compact_model"],
            compact_tokens=row["compact_tokens"],
        )


__all__ = ["SessionStorage", "InMemorySessionStorage", "SQLiteSessionStorage"]
