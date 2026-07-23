"""SQLite-backed SessionIntent store (I-S1).

Append-only schema (fail-closed; wipe incompatible dev DBs, no migration):

``session_intent_entries`` — one row per timeline entry:
  - session_id TEXT NOT NULL
  - seq INTEGER NOT NULL  (per-session monotonic)
  - kind TEXT NOT NULL
  - text TEXT NOT NULL
  - at INTEGER NOT NULL
  - run_id TEXT
  - PRIMARY KEY (session_id, seq)

``session_intent_meta`` — one row per session:
  - session_id TEXT PRIMARY KEY
  - last_run_plan TEXT  (JSON RunPlan snapshot, nullable)
  - updated_at INTEGER NOT NULL

The legacy single-table ``session_intent`` payload document is rejected at
connect time (ADR 0049 fail-closed).
"""

import json
import time

import aiosqlite

from agiwo.agent.intent.base import SessionIntentStore
from agiwo.agent.intent.models import MAX_INTENT_ENTRIES, IntentEntry, SessionIntent
from agiwo.agent.intent.serialization import (
    deserialize_intent_entry,
    deserialize_run_plan,
    serialize_run_plan,
)
from agiwo.agent.models.plan import RunPlan
from agiwo.utils.logging import get_logger
from agiwo.utils.storage_support.sqlite_runtime import (
    SQLiteConnectionRuntime,
    execute_statements,
)

logger = get_logger(__name__)

_ENTRY_COLUMNS = frozenset({"session_id", "seq", "kind", "text", "at", "run_id"})
_META_COLUMNS = frozenset({"session_id", "last_run_plan", "updated_at"})


class SQLiteSessionIntentStore(SessionIntentStore):
    """Persist SessionIntent beside RunLog when sqlite storage is configured."""

    def __init__(self, db_path: str = "agiwo.db") -> None:
        super().__init__()
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
        await self._reject_legacy_schema(connection)
        await execute_statements(
            connection,
            [
                """
                CREATE TABLE IF NOT EXISTS session_intent_entries (
                    session_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    at INTEGER NOT NULL,
                    run_id TEXT,
                    PRIMARY KEY (session_id, seq)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS session_intent_meta (
                    session_id TEXT PRIMARY KEY,
                    last_run_plan TEXT,
                    updated_at INTEGER NOT NULL
                )
                """,
            ],
        )
        await self._validate_schema(connection)
        await connection.commit()

    async def _reject_legacy_schema(self, connection: aiosqlite.Connection) -> None:
        async with connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name = 'session_intent'
            """
        ) as cursor:
            if await cursor.fetchone() is not None:
                raise RuntimeError(
                    "session_intent schema is incompatible with current agent "
                    "version; please clear old intent data before restarting"
                )

    async def _validate_schema(self, connection: aiosqlite.Connection) -> None:
        async with connection.execute(
            "PRAGMA table_info(session_intent_entries)"
        ) as cursor:
            entry_columns = {row[1] for row in await cursor.fetchall()}
        async with connection.execute(
            "PRAGMA table_info(session_intent_meta)"
        ) as cursor:
            meta_columns = {row[1] for row in await cursor.fetchall()}
        if entry_columns != set(_ENTRY_COLUMNS) or meta_columns != set(_META_COLUMNS):
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
            SELECT last_run_plan, updated_at
            FROM session_intent_meta
            WHERE session_id = ?
            """,
            (session_id,),
        ) as cursor:
            meta_row = await cursor.fetchone()
        async with conn.execute(
            """
            SELECT kind, text, at, run_id
            FROM session_intent_entries
            WHERE session_id = ?
            ORDER BY seq DESC
            LIMIT ?
            """,
            (session_id, MAX_INTENT_ENTRIES),
        ) as cursor:
            entry_rows = await cursor.fetchall()
        if meta_row is None and not entry_rows:
            return None
        entries = [
            deserialize_intent_entry(
                {
                    "kind": row["kind"],
                    "text": row["text"],
                    "at": row["at"],
                    "run_id": row["run_id"],
                }
            )
            for row in reversed(entry_rows)
        ]
        last_run_plan = None
        updated_at = 0
        if meta_row is not None:
            updated_at = int(meta_row["updated_at"])
            if meta_row["last_run_plan"]:
                last_run_plan = deserialize_run_plan(
                    json.loads(meta_row["last_run_plan"])
                )
        return SessionIntent(
            entries=entries,
            last_run_plan=last_run_plan,
            updated_at=updated_at,
        )

    async def upsert(self, session_id: str, intent: SessionIntent) -> None:
        """Replace all entries + meta for one session (test / full rewrite path)."""
        conn = await self._ensure_connection()
        await conn.execute(
            "DELETE FROM session_intent_entries WHERE session_id = ?",
            (session_id,),
        )
        for seq, entry in enumerate(intent.entries, start=1):
            await conn.execute(
                """
                INSERT INTO session_intent_entries
                    (session_id, seq, kind, text, at, run_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    seq,
                    entry.kind,
                    entry.text,
                    entry.at,
                    entry.run_id,
                ),
            )
        plan_json = (
            json.dumps(serialize_run_plan(intent.last_run_plan))
            if intent.last_run_plan is not None
            else None
        )
        await conn.execute(
            """
            INSERT INTO session_intent_meta (session_id, last_run_plan, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                last_run_plan = excluded.last_run_plan,
                updated_at = excluded.updated_at
            """,
            (session_id, plan_json, intent.updated_at),
        )
        await conn.commit()

    async def _append_entry_unlocked(
        self,
        session_id: str,
        entry: IntentEntry,
        *,
        last_run_plan: RunPlan | None = None,
    ) -> SessionIntent:
        conn = await self._ensure_connection()
        updated_at = int(time.time())
        async with conn.execute(
            """
            SELECT COALESCE(MAX(seq), 0) AS max_seq
            FROM session_intent_entries
            WHERE session_id = ?
            """,
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
        next_seq = int(row["max_seq"]) + 1
        await conn.execute(
            """
            INSERT INTO session_intent_entries
                (session_id, seq, kind, text, at, run_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                next_seq,
                entry.kind,
                entry.text,
                entry.at,
                entry.run_id,
            ),
        )
        # Preserve prior plan when append does not refresh it.
        existing_plan_json: str | None = None
        if last_run_plan is None:
            async with conn.execute(
                """
                SELECT last_run_plan FROM session_intent_meta
                WHERE session_id = ?
                """,
                (session_id,),
            ) as cursor:
                meta = await cursor.fetchone()
            if meta is not None:
                existing_plan_json = meta["last_run_plan"]
        plan_json = (
            json.dumps(serialize_run_plan(last_run_plan))
            if last_run_plan is not None
            else existing_plan_json
        )
        await conn.execute(
            """
            INSERT INTO session_intent_meta (session_id, last_run_plan, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                last_run_plan = excluded.last_run_plan,
                updated_at = excluded.updated_at
            """,
            (session_id, plan_json, updated_at),
        )
        if next_seq > MAX_INTENT_ENTRIES * 2:
            cutoff = next_seq - MAX_INTENT_ENTRIES
            await conn.execute(
                """
                DELETE FROM session_intent_entries
                WHERE session_id = ? AND seq <= ?
                """,
                (session_id, cutoff),
            )
        await conn.commit()
        loaded = await self.get(session_id)
        assert loaded is not None
        return loaded


__all__ = ["SQLiteSessionIntentStore"]
