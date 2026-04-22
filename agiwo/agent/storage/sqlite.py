"""SQLite-backed run-log storage."""

import asyncio
import json
from typing import Literal

import aiosqlite

from agiwo.agent.models.log import (
    CompactionApplied,
    RunLogEntry,
    RunLogEntryKind,
    RunRolledBack,
    StepCondensedContentUpdated,
    build_compact_metadata_from_entry,
)
from agiwo.agent.models.run import CompactMetadata, RunView
from agiwo.agent.models.runtime_decision import RuntimeDecisionState
from agiwo.agent.models.step import StepView
from agiwo.agent.storage.base import RunLogStorage, SessionRunStats
from agiwo.agent.storage.serialization import (
    build_runtime_decision_state_from_entries,
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
        self._lock = asyncio.Lock()
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
                """
                CREATE INDEX IF NOT EXISTS idx_run_log_kind_created_at
                ON run_log_entries(kind, json_extract(payload, '$.created_at') DESC)
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
        max_sequences: dict[str, int] = {}
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
            current_max = max_sequences.get(entry.session_id, 0)
            if entry.sequence > current_max:
                max_sequences[entry.session_id] = entry.sequence
        async with self._lock:
            await conn.execute("BEGIN IMMEDIATE")
            try:
                await conn.executemany(
                    """
                    INSERT INTO run_log_entries
                    (session_id, sequence, run_id, agent_id, kind, payload)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                for session_id, sequence in max_sequences.items():
                    await conn.execute(
                        """
                        INSERT INTO counters (session_id, sequence)
                        VALUES (?, ?)
                        ON CONFLICT(session_id) DO UPDATE SET
                            sequence = CASE
                                WHEN counters.sequence < excluded.sequence
                                THEN excluded.sequence
                                ELSE counters.sequence
                            END
                        """,
                        (session_id, sequence),
                    )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

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
        if session_id is not None:
            query += " ORDER BY sequence DESC"
        else:
            query += " ORDER BY json_extract(payload, '$.created_at') DESC"
        query += " LIMIT ? OFFSET ?"
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

    async def get_runtime_decision_state(
        self,
        *,
        session_id: str,
        run_id: str | None = None,
        agent_id: str | None = None,
    ) -> RuntimeDecisionState:
        decision_kinds = (
            RunLogEntryKind.TERMINATION_DECIDED,
            RunLogEntryKind.COMPACTION_APPLIED,
            RunLogEntryKind.COMPACTION_FAILED,
            RunLogEntryKind.RETROSPECT_APPLIED,
            RunLogEntryKind.RUN_ROLLED_BACK,
        )
        entries: list[RunLogEntry] = []
        for kind in decision_kinds:
            entry = await self._get_latest_entry_for_kind(
                session_id=session_id,
                kind=kind,
                run_id=run_id,
                agent_id=agent_id,
            )
            if entry is not None:
                entries.append(entry)
        entries.sort(key=lambda item: item.sequence)
        return build_runtime_decision_state_from_entries(entries)

    async def get_committed_step_count(self, session_id: str) -> int:
        conn = await self._ensure_connection()
        kinds = [
            RunLogEntryKind.USER_STEP_COMMITTED.value,
            RunLogEntryKind.ASSISTANT_STEP_COMMITTED.value,
            RunLogEntryKind.TOOL_STEP_COMMITTED.value,
            RunLogEntryKind.STEP_CONDENSED_CONTENT_UPDATED.value,
            RunLogEntryKind.RUN_ROLLED_BACK.value,
        ]
        placeholders = ",".join("?" for _ in kinds)
        entries: list[RunLogEntry] = []
        async with conn.execute(
            f"""
            SELECT payload FROM run_log_entries
            WHERE session_id = ? AND kind IN ({placeholders})
            ORDER BY sequence ASC
            """,
            [session_id, *kinds],
        ) as cursor:
            async for row in cursor:
                entries.append(self._deserialize_run_log_entry(row))
        return len(build_step_views_from_entries(entries))

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
        kinds = [
            RunLogEntryKind.USER_STEP_COMMITTED.value,
            RunLogEntryKind.ASSISTANT_STEP_COMMITTED.value,
            RunLogEntryKind.TOOL_STEP_COMMITTED.value,
            RunLogEntryKind.STEP_CONDENSED_CONTENT_UPDATED.value,
            RunLogEntryKind.RUN_ROLLED_BACK.value,
        ]
        session_placeholders = ",".join("?" for _ in session_ids)
        kind_placeholders = ",".join("?" for _ in kinds)
        entries_by_session: dict[str, list[RunLogEntry]] = {
            session_id: [] for session_id in session_ids
        }
        async with conn.execute(
            f"""
            SELECT session_id, payload FROM run_log_entries
            WHERE session_id IN ({session_placeholders})
              AND kind IN ({kind_placeholders})
            ORDER BY session_id ASC, sequence ASC
            """,
            [*session_ids, *kinds],
        ) as cursor:
            async for row in cursor:
                entries_by_session[row["session_id"]].append(
                    self._deserialize_run_log_entry(row)
                )

        result: dict[str, int] = {}
        for session_id, entries in entries_by_session.items():
            result[session_id] = len(build_step_views_from_entries(entries))
        return result

    async def list_step_views(
        self,
        *,
        session_id: str,
        start_seq: int | None = None,
        end_seq: int | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        include_rolled_back: bool = False,
        limit: int = 1000,
        order: Literal["asc", "desc"] = "asc",
    ) -> list[StepView]:
        conn = await self._ensure_connection()
        kinds = [
            RunLogEntryKind.USER_STEP_COMMITTED.value,
            RunLogEntryKind.ASSISTANT_STEP_COMMITTED.value,
            RunLogEntryKind.TOOL_STEP_COMMITTED.value,
            RunLogEntryKind.STEP_CONDENSED_CONTENT_UPDATED.value,
            RunLogEntryKind.RUN_ROLLED_BACK.value,
        ]
        placeholders = ",".join("?" for _ in kinds)
        query = (
            f"SELECT payload FROM run_log_entries WHERE session_id = ? "
            f"AND kind IN ({placeholders})"
        )
        params: list[object] = [session_id, *kinds]
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
        step_views = build_step_views_from_entries(
            entries,
            include_rolled_back=include_rolled_back,
        )
        if start_seq is not None:
            step_views = [step for step in step_views if step.sequence >= start_seq]
        if end_seq is not None:
            step_views = [step for step in step_views if step.sequence <= end_seq]
        if order == "desc":
            step_views = list(reversed(step_views))
        return step_views[:limit]

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

    async def _get_latest_entry_for_kind(
        self,
        *,
        session_id: str,
        kind: RunLogEntryKind,
        run_id: str | None = None,
        agent_id: str | None = None,
    ) -> RunLogEntry | None:
        conn = await self._ensure_connection()
        query = "SELECT payload FROM run_log_entries WHERE session_id = ? AND kind = ?"
        params: list[object] = [session_id, kind.value]
        if run_id is not None:
            query += " AND run_id = ?"
            params.append(run_id)
        if agent_id is not None:
            query += " AND agent_id = ?"
            params.append(agent_id)
        query += " ORDER BY sequence DESC LIMIT 1"
        async with conn.execute(query, params) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._deserialize_run_log_entry(row)

    async def append_run_rollback(
        self,
        *,
        session_id: str,
        run_id: str,
        agent_id: str,
        start_sequence: int,
        end_sequence: int,
        reason: str,
    ) -> None:
        sequence = await self.allocate_sequence(session_id)
        await self.append_entries(
            [
                RunRolledBack(
                    sequence=sequence,
                    session_id=session_id,
                    run_id=run_id,
                    agent_id=agent_id,
                    start_sequence=start_sequence,
                    end_sequence=end_sequence,
                    reason=reason,
                )
            ]
        )

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
        async with self._lock:
            await conn.execute("BEGIN IMMEDIATE")
            try:
                async with conn.execute(
                    "SELECT sequence FROM counters WHERE session_id = ?",
                    (session_id,),
                ) as cursor:
                    row = await cursor.fetchone()
                if row is None:
                    async with conn.execute(
                        "SELECT COALESCE(MAX(sequence), 0) FROM run_log_entries WHERE session_id = ?",
                        (session_id,),
                    ) as cursor:
                        max_row = await cursor.fetchone()
                    new_seq = (
                        max_row[0] if max_row and max_row[0] is not None else 0
                    ) + 1
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
