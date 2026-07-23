"""SQLite-backed SessionIntent store (I-S1).

Schema (fail-closed; wipe incompatible dev DBs, no migration):

``session_intent`` table — one row per ``session_id``:
  - session_id TEXT PRIMARY KEY
  - payload TEXT NOT NULL (JSON SessionIntent document)
  - updated_at INTEGER NOT NULL
"""

import json

import aiosqlite

from agiwo.agent.intent.base import SessionIntentStore
from agiwo.agent.intent.models import SessionIntent
from agiwo.agent.intent.serialization import (
    deserialize_session_intent,
    serialize_session_intent,
)
from agiwo.utils.logging import get_logger
from agiwo.utils.storage_support.sqlite_runtime import (
    SQLiteConnectionRuntime,
    execute_statements,
)

logger = get_logger(__name__)

_SESSION_INTENT_COLUMNS = frozenset({"session_id", "payload", "updated_at"})


class SQLiteSessionIntentStore(SessionIntentStore):
    """Persist SessionIntent beside RunLog when sqlite storage is configured."""

    def __init__(self, db_path: str = "agiwo.db") -> None:
        self.db_path = db_path
        self._connection: aiosqlite.Connection | None = None
        self._runtime = SQLiteConnectionRuntime(
            db_path=db_path,
            logger=logger,
            connect_event="sqlite_session_intent_store_connected",
        )

    async def connect(self) -> None:
        self._connection = await self._runtime.ensure_connection(
            self._initialize_schema
        )

    async def close(self) -> None:
        await self.disconnect()

    async def disconnect(self) -> None:
        if self._connection:
            await self._runtime.disconnect()
            self._connection = None

    async def _initialize_schema(self, connection: aiosqlite.Connection) -> None:
        await execute_statements(
            connection,
            [
                """
                CREATE TABLE IF NOT EXISTS session_intent (
                    session_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """,
            ],
        )
        await self._validate_schema(connection)
        await connection.commit()

    async def _validate_schema(self, connection: aiosqlite.Connection) -> None:
        async with connection.execute("PRAGMA table_info(session_intent)") as cursor:
            columns = {row[1] for row in await cursor.fetchall()}
        if columns != set(_SESSION_INTENT_COLUMNS):
            raise RuntimeError(
                "session_intent schema is incompatible with current agent version; "
                "please clear old intent data before restarting"
            )

    async def _ensure_connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            await self.connect()
        assert self._connection is not None
        return self._connection

    async def get(self, session_id: str) -> SessionIntent | None:
        conn = await self._ensure_connection()
        async with conn.execute(
            """
            SELECT payload
            FROM session_intent
            WHERE session_id = ?
            """,
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload"])
        return deserialize_session_intent(payload)

    async def upsert(self, session_id: str, intent: SessionIntent) -> None:
        conn = await self._ensure_connection()
        payload = json.dumps(serialize_session_intent(intent))
        await conn.execute(
            """
            INSERT INTO session_intent (session_id, payload, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                payload = excluded.payload,
                updated_at = excluded.updated_at
            """,
            (session_id, payload, intent.updated_at),
        )
        await conn.commit()


__all__ = ["SQLiteSessionIntentStore"]
