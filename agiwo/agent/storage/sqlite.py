"""SQLite-backed run-log storage."""

import json

import aiosqlite

from agiwo.agent.models.log import (
    CompactionApplied,
    RunLogEntry,
    RunLogEntryKind,
    StepCondensedContentUpdated,
    build_compact_metadata_from_entry,
)
from agiwo.agent.models.run import CompactMetadata, RunView
from agiwo.agent.models.step import StepView
from agiwo.agent.storage.base import RunLogStorage, SessionRunStats
from agiwo.agent.storage.serialization import (
    build_run_view_from_entries,
    build_run_views_from_entries,
    build_step_views_from_entries,
    deserialize_run_log_entry_from_storage,
    serialize_run_log_entry_for_storage,
)
from agiwo.utils.storage_support.sqlite_runtime import (
    SQLiteConnectionRuntime,
    execute_statements,
)
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class SQLiteRunLogStorage(RunLogStorage):
    """SQLite implementation backed only by canonical run-log entries."""

    def __init__(self, db_path: str = "agiwo.db") -> None:
        self.db_path = db_path
        self._connection: aiosqlite.Connection | None = None
        self._runtime = SQLiteConnectionRuntime(
            db_path=db_path,
            logger=logger,
            connect_event="sqlite_run_log_storage_connected",
        )

    @property
    def _initialized(self) -> bool:
        return self._runtime.initialized

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
                CREATE TABLE IF NOT EXISTS counters (
                    session_id TEXT PRIMARY KEY,
                    sequence INTEGER NOT NULL DEFAULT 0
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS run_log_entries (
                    session_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    run_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (session_id, sequence)
                )
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_run_log_session_run_seq
                ON run_log_entries(session_id, run_id, sequence)
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_run_log_session_kind_seq
                ON run_log_entries(session_id, kind, sequence)
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_run_log_run_seq
                ON run_log_entries(run_id, sequence)
                """,
            ],
        )
        await connection.commit()

    async def _ensure_connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            await self.connect()
        assert self._connection is not None
        return self._connection

    def _deserialize_run_log_entry(self, row: aiosqlite.Row) -> RunLogEntry:
        return deserialize_run_log_entry_from_storage(self._loads(row["payload"]))

    async def append_entries(self, entries: list[RunLogEntry]) -> None:
        if not entries:
            return
        conn = await self._ensure_connection()
        rows = []
        for entry in entries:
            payload = serialize_run_log_entry_for_storage(entry)
            rows.append(
                (
                    entry.session_id,
                    entry.sequence,
                    entry.run_id,
                    entry.agent_id,
                    entry.kind.value,
                    json.dumps(payload, ensure_ascii=False, default=str),
                )
            )
        await conn.executemany(
            """
            INSERT INTO run_log_entries
            (session_id, sequence, run_id, agent_id, kind, payload)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        await conn.commit()

    async def list_entries(
        self,
        *,
        session_id: str,
        run_id: str | None = None,
        agent_id: str | None = None,
        after_sequence: int | None = None,
        limit: int = 1000,
    ) -> list[RunLogEntry]:
        conn = await self._ensure_connection()
        query = "SELECT payload FROM run_log_entries WHERE session_id = ?"
        params: list[object] = [session_id]
        if run_id is not None:
            query += " AND run_id = ?"
            params.append(run_id)
        if agent_id is not None:
            query += " AND agent_id = ?"
            params.append(agent_id)
        if after_sequence is not None:
            query += " AND sequence > ?"
            params.append(after_sequence)
        query += " ORDER BY sequence ASC LIMIT ?"
        params.append(limit)

        entries: list[RunLogEntry] = []
        async with conn.execute(query, params) as cursor:
            async for row in cursor:
                entries.append(self._deserialize_run_log_entry(row))
        return entries

    async def _load_entries_for_run_ids(
        self, run_ids: list[str]
    ) -> dict[str, list[RunLogEntry]]:
        if not run_ids:
            return {}
        conn = await self._ensure_connection()
        placeholders = ",".join("?" for _ in run_ids)
        entries_by_run_id: dict[str, list[RunLogEntry]] = {
            run_id: [] for run_id in run_ids
        }
        async with conn.execute(
            f"""
            SELECT payload FROM run_log_entries
            WHERE run_id IN ({placeholders})
            ORDER BY run_id ASC, sequence ASC
            """,
            run_ids,
        ) as cursor:
            async for row in cursor:
                entry = self._deserialize_run_log_entry(row)
                entries_by_run_id.setdefault(entry.run_id, []).append(entry)
        return entries_by_run_id

    async def get_run_view(self, run_id: str) -> RunView | None:
        entries_by_run_id = await self._load_entries_for_run_ids([run_id])
        entries = entries_by_run_id.get(run_id, [])
        if not entries:
            return None
        return build_run_view_from_entries(entries)

    async def list_run_views(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[RunView]:
        conn = await self._ensure_connection()
        query = "SELECT payload FROM run_log_entries WHERE kind = ?"
        params: list[object] = [RunLogEntryKind.RUN_STARTED.value]
        if user_id is not None:
            query += " AND json_extract(payload, '$.user_id') = ?"
            params.append(user_id)
        if session_id is not None:
            query += " AND session_id = ?"
            params.append(session_id)
        query += " ORDER BY sequence DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        started_entries: list[RunLogEntry] = []
        async with conn.execute(query, params) as cursor:
            async for row in cursor:
                started_entries.append(self._deserialize_run_log_entry(row))

        if not started_entries:
            return []

        entries_by_run_id = await self._load_entries_for_run_ids(
            [entry.run_id for entry in started_entries]
        )
        return [
            view
            for view in (
                build_run_view_from_entries(entries_by_run_id.get(entry.run_id, []))
                for entry in started_entries
            )
            if view is not None
        ]

    async def get_session_run_stats(self, session_id: str) -> SessionRunStats:
        conn = await self._ensure_connection()
        entries: list[RunLogEntry] = []
        async with conn.execute(
            """
            SELECT payload FROM run_log_entries
            WHERE session_id = ?
            ORDER BY sequence ASC
            """,
            (session_id,),
        ) as cursor:
            async for row in cursor:
                entries.append(self._deserialize_run_log_entry(row))

        run_views = build_run_views_from_entries(entries)
        step_count = len(build_step_views_from_entries(entries))
        return SessionRunStats(
            committed_step_count=step_count,
            run_views=run_views,
        )

    async def get_committed_step_count(self, session_id: str) -> int:
        conn = await self._ensure_connection()
        kinds = [
            RunLogEntryKind.USER_STEP_COMMITTED.value,
            RunLogEntryKind.ASSISTANT_STEP_COMMITTED.value,
            RunLogEntryKind.TOOL_STEP_COMMITTED.value,
            RunLogEntryKind.STEP_CONDENSED_CONTENT_UPDATED.value,
        ]
        placeholders = ",".join("?" for _ in kinds)
        async with conn.execute(
            f"""
            SELECT COUNT(*) FROM run_log_entries
            WHERE session_id = ? AND kind IN ({placeholders})
            """,
            [session_id, *kinds],
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def batch_count_run_views(self, session_ids: list[str]) -> dict[str, int]:
        if not session_ids:
            return {}
        conn = await self._ensure_connection()
        placeholders = ",".join("?" for _ in session_ids)
        result: dict[str, int] = {sid: 0 for sid in session_ids}
        async with conn.execute(
            f"""
            SELECT session_id, COUNT(*) FROM run_log_entries
            WHERE session_id IN ({placeholders}) AND kind = ?
            GROUP BY session_id
            """,
            [*session_ids, RunLogEntryKind.RUN_STARTED.value],
        ) as cursor:
            async for row in cursor:
                result[row[0]] = row[1]
        return result

    async def batch_get_committed_step_counts(
        self, session_ids: list[str]
    ) -> dict[str, int]:
        if not session_ids:
            return {}
        conn = await self._ensure_connection()
        session_placeholders = ",".join("?" for _ in session_ids)
        kinds = [
            RunLogEntryKind.USER_STEP_COMMITTED.value,
            RunLogEntryKind.ASSISTANT_STEP_COMMITTED.value,
            RunLogEntryKind.TOOL_STEP_COMMITTED.value,
        ]
        kind_placeholders = ",".join("?" for _ in kinds)
        result: dict[str, int] = {sid: 0 for sid in session_ids}
        async with conn.execute(
            f"""
            SELECT session_id, COUNT(*) FROM run_log_entries
            WHERE session_id IN ({session_placeholders})
              AND kind IN ({kind_placeholders})
            GROUP BY session_id
            """,
            [*session_ids, *kinds],
        ) as cursor:
            async for row in cursor:
                result[row[0]] = row[1]
        return result

    async def list_step_views(
        self,
        *,
        session_id: str,
        start_seq: int | None = None,
        end_seq: int | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 1000,
    ) -> list[StepView]:
        conn = await self._ensure_connection()
        kinds = [
            RunLogEntryKind.USER_STEP_COMMITTED.value,
            RunLogEntryKind.ASSISTANT_STEP_COMMITTED.value,
            RunLogEntryKind.TOOL_STEP_COMMITTED.value,
        ]
        placeholders = ",".join("?" for _ in kinds)
        query = (
            f"SELECT payload FROM run_log_entries WHERE session_id = ? "
            f"AND kind IN ({placeholders})"
        )
        params: list[object] = [session_id, *kinds]
        if start_seq is not None:
            query += " AND sequence >= ?"
            params.append(start_seq)
        if end_seq is not None:
            query += " AND sequence <= ?"
            params.append(end_seq)
        if run_id is not None:
            query += " AND run_id = ?"
            params.append(run_id)
        if agent_id is not None:
            query += " AND agent_id = ?"
            params.append(agent_id)
        query += " ORDER BY sequence ASC"

        entries: list[RunLogEntry] = []
        async with conn.execute(query, params) as cursor:
            async for row in cursor:
                entries.append(self._deserialize_run_log_entry(row))
        return build_step_views_from_entries(entries)[:limit]

    async def append_step_condensed_content(
        self,
        session_id: str,
        run_id: str,
        agent_id: str,
        step_id: str,
        condensed_content: str,
    ) -> bool:
        conn = await self._ensure_connection()
        kinds = [
            RunLogEntryKind.USER_STEP_COMMITTED.value,
            RunLogEntryKind.ASSISTANT_STEP_COMMITTED.value,
            RunLogEntryKind.TOOL_STEP_COMMITTED.value,
        ]
        placeholders = ",".join("?" for _ in kinds)
        async with conn.execute(
            f"""
            SELECT 1 FROM run_log_entries
            WHERE session_id = ? AND kind IN ({placeholders})
              AND json_extract(payload, '$.step_id') = ?
            LIMIT 1
            """,
            [session_id, *kinds, step_id],
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return False
        sequence = await self.allocate_sequence(session_id)
        await self.append_entries(
            [
                StepCondensedContentUpdated(
                    sequence=sequence,
                    session_id=session_id,
                    run_id=run_id,
                    agent_id=agent_id,
                    step_id=step_id,
                    condensed_content=condensed_content,
                )
            ]
        )
        return True

    async def get_latest_compact_metadata(
        self, session_id: str, agent_id: str
    ) -> CompactMetadata | None:
        conn = await self._ensure_connection()
        async with conn.execute(
            """
            SELECT payload FROM run_log_entries
            WHERE session_id = ? AND agent_id = ? AND kind = ?
            ORDER BY sequence DESC
            LIMIT 1
            """,
            (
                session_id,
                agent_id,
                RunLogEntryKind.COMPACTION_APPLIED.value,
            ),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        entry = self._deserialize_run_log_entry(row)
        assert isinstance(entry, CompactionApplied)
        return build_compact_metadata_from_entry(entry)

    async def get_compact_history(
        self, session_id: str, agent_id: str
    ) -> list[CompactMetadata]:
        conn = await self._ensure_connection()
        entries: list[CompactMetadata] = []
        async with conn.execute(
            """
            SELECT payload FROM run_log_entries
            WHERE session_id = ? AND agent_id = ? AND kind = ?
            ORDER BY sequence ASC
            """,
            (
                session_id,
                agent_id,
                RunLogEntryKind.COMPACTION_APPLIED.value,
            ),
        ) as cursor:
            async for row in cursor:
                entry = self._deserialize_run_log_entry(row)
                assert isinstance(entry, CompactionApplied)
                entries.append(build_compact_metadata_from_entry(entry))
        return entries

    async def get_max_sequence(self, session_id: str) -> int:
        conn = await self._ensure_connection()
        async with conn.execute(
            "SELECT MAX(sequence) FROM run_log_entries WHERE session_id = ?",
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row and row[0] is not None else 0

    async def allocate_sequence(self, session_id: str) -> int:
        conn = await self._ensure_connection()
        await conn.execute("BEGIN IMMEDIATE")
        try:
            async with conn.execute(
                "SELECT sequence FROM counters WHERE session_id = ?",
                (session_id,),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                new_seq = await self.get_max_sequence(session_id) + 1
                await conn.execute(
                    "INSERT INTO counters (session_id, sequence) VALUES (?, ?)",
                    (session_id, new_seq),
                )
            else:
                new_seq = row[0] + 1
                await conn.execute(
                    "UPDATE counters SET sequence = ? WHERE session_id = ?",
                    (new_seq, session_id),
                )
            await conn.commit()
            return new_seq
        except Exception:
            await conn.rollback()
            raise

    @staticmethod
    def _loads(value: str):
        return json.loads(value)


__all__ = ["SQLiteRunLogStorage"]
