"""
SQLite implementation of RunStepStorage.
"""

import json
from datetime import datetime, timezone

import aiosqlite

from agiwo.agent.models.log import RunLogEntry, RunLogEntryKind
from agiwo.agent.models.run import CompactMetadata, Run, RunView
from agiwo.agent.models.step import StepRecord, StepView
from agiwo.agent.storage.base import RunStepStorage
from agiwo.agent.storage.serialization import (
    build_run_view_from_entries,
    build_step_views_from_entries,
    deserialize_run_log_entry_from_storage,
    deserialize_run_from_storage,
    deserialize_step_from_storage,
    serialize_run_log_entry_for_storage,
    serialize_run_for_storage,
    serialize_step_for_storage,
)
from agiwo.utils.storage_support.sqlite_runtime import (
    SQLiteConnectionRuntime,
    execute_statements,
)
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class SQLiteRunStepStorage(RunStepStorage):
    """
    SQLite implementation of RunStepStorage.
    """

    def __init__(self, db_path: str = "agiwo.db") -> None:
        self.db_path = db_path
        self._connection: aiosqlite.Connection | None = None
        self._runtime = SQLiteConnectionRuntime(
            db_path=db_path,
            logger=logger,
            connect_event="sqlite_run_step_storage_connected",
        )

    @property
    def _initialized(self) -> bool:
        return self._runtime.initialized

    async def connect(self) -> None:
        """Initialize database connection and create tables using shared pool."""
        self._connection = await self._runtime.ensure_connection(
            self._initialize_schema
        )

    async def close(self) -> None:
        """Close database connection."""
        await self.disconnect()

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
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    user_id TEXT,
                    user_input TEXT NOT NULL,
                    status TEXT NOT NULL,
                    response_content TEXT,
                    metrics TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    parent_run_id TEXT,
                    trace_id TEXT
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS steps (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    agent_id TEXT,
                    runnable_type TEXT,
                    role TEXT NOT NULL,
                    content TEXT,
                    content_for_user TEXT,
                    reasoning_content TEXT,
                    user_input TEXT,
                    tool_calls TEXT,
                    tool_call_id TEXT,
                    name TEXT,
                    is_error INTEGER DEFAULT 0,
                    metrics TEXT,
                    created_at TEXT NOT NULL,
                    parent_run_id TEXT,
                    depth INTEGER DEFAULT 0,
                    condensed_content TEXT,
                    UNIQUE(session_id, sequence)
                )
                """,
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
                "CREATE INDEX IF NOT EXISTS idx_runs_agent_id ON runs(agent_id)",
                "CREATE INDEX IF NOT EXISTS idx_runs_user_id ON runs(user_id)",
                "CREATE INDEX IF NOT EXISTS idx_runs_session_id ON runs(session_id)",
                "CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at)",
                "CREATE INDEX IF NOT EXISTS idx_steps_session_seq ON steps(session_id, sequence)",
                "CREATE INDEX IF NOT EXISTS idx_steps_session_run_seq ON steps(session_id, run_id, sequence)",
                "CREATE INDEX IF NOT EXISTS idx_steps_session_tool_call_id ON steps(session_id, tool_call_id)",
                "CREATE INDEX IF NOT EXISTS idx_steps_created_at ON steps(created_at)",
                "CREATE INDEX IF NOT EXISTS idx_run_log_session_run_seq ON run_log_entries(session_id, run_id, sequence)",
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

    async def _ensure_connection(self) -> aiosqlite.Connection:
        """Ensure database connection is established and return it."""
        if self._connection is None:
            await self.connect()
        assert self._connection is not None
        return self._connection

    def _serialize_model(self, model: Run | StepRecord) -> dict:
        data = (
            serialize_run_for_storage(model)
            if isinstance(model, Run)
            else serialize_step_for_storage(model)
        )
        if "metrics" in data:
            data["metrics"] = self._dumps(data["metrics"])
        if "tool_calls" in data:
            data["tool_calls"] = self._dumps(data["tool_calls"])
        return data

    def _deserialize_run(self, row: aiosqlite.Row) -> Run:
        """Deserialize database row to Run model."""
        data = dict(row)

        if data.get("metrics"):
            data["metrics"] = self._loads(data["metrics"])
        return deserialize_run_from_storage(data)

    def _deserialize_step(self, row: aiosqlite.Row) -> StepRecord:
        """Deserialize database row to StepRecord model."""
        data = dict(row)

        if data.get("metrics"):
            data["metrics"] = self._loads(data["metrics"])
        if data.get("tool_calls"):
            data["tool_calls"] = self._loads(data["tool_calls"])
        return deserialize_step_from_storage(data)

    def _deserialize_run_log_entry(self, row: aiosqlite.Row) -> RunLogEntry:
        data = self._loads(dict(row)["payload"])
        return deserialize_run_log_entry_from_storage(data)

    async def append_entries(self, entries: list[RunLogEntry]) -> None:
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
            INSERT OR REPLACE INTO run_log_entries
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
            ORDER BY session_id ASC, sequence ASC
            """,
            run_ids,
        ) as cursor:
            async for row in cursor:
                entry = self._deserialize_run_log_entry(row)
                entries_by_run_id.setdefault(entry.run_id, []).append(entry)
        return entries_by_run_id

    # --- Run Operations ---

    async def save_run(self, run: Run) -> None:
        """Save or update a run."""
        conn = await self._ensure_connection()
        data = self._serialize_model(run)
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data])
        values = list(data.values())
        query = f"""
            INSERT OR REPLACE INTO runs ({columns})
            VALUES ({placeholders})
        """
        await conn.execute(query, values)
        await conn.commit()

    async def get_run(self, run_id: str) -> Run | None:
        """Get a run by ID."""
        conn = await self._ensure_connection()
        async with conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)) as cursor:
            row = await cursor.fetchone()
            if row:
                return self._deserialize_run(row)
            return None

    async def get_run_view(self, run_id: str) -> RunView | None:
        entries_by_run_id = await self._load_entries_for_run_ids([run_id])
        entries = entries_by_run_id.get(run_id, [])
        if entries:
            return build_run_view_from_entries(entries)
        return await super().get_run_view(run_id)

    async def list_runs(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Run]:
        """List runs with filtering and pagination."""
        conn = await self._ensure_connection()
        query = "SELECT * FROM runs WHERE agent_id IS NOT NULL"
        params = []

        if user_id is not None:
            query += " AND user_id = ?"
            params.append(user_id)
        if session_id is not None:
            query += " AND session_id = ?"
            params.append(session_id)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        runs = []
        async with conn.execute(query, params) as cursor:
            async for row in cursor:
                runs.append(self._deserialize_run(row))
        return runs

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
        if session_id is not None:
            query += " AND session_id = ?"
            params.append(session_id)
        query += " ORDER BY session_id ASC, sequence ASC"

        started_entries: list[RunLogEntry] = []
        async with conn.execute(query, params) as cursor:
            async for row in cursor:
                started_entries.append(self._deserialize_run_log_entry(row))

        run_views = [
            view
            for view in (
                build_run_view_from_entries([entry]) for entry in started_entries
            )
            if view is not None
        ]
        if user_id is not None:
            run_views = [view for view in run_views if view.user_id == user_id]
        earliest = datetime.min.replace(tzinfo=timezone.utc)
        run_views.sort(
            key=lambda view: view.created_at or earliest,
            reverse=True,
        )
        selected = run_views[offset : offset + limit]
        if not selected:
            return []

        entries_by_run_id = await self._load_entries_for_run_ids(
            [view.run_id for view in selected]
        )
        return [
            view
            for view in (
                build_run_view_from_entries(entries_by_run_id.get(run_view.run_id, []))
                for run_view in selected
            )
            if view is not None
        ]

    async def count_runs(self, session_id: str) -> int:
        conn = await self._ensure_connection()
        async with conn.execute(
            "SELECT COUNT(*) FROM runs WHERE session_id = ? AND agent_id IS NOT NULL",
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def delete_run(self, run_id: str) -> None:
        """Delete a run and its associated steps."""
        conn = await self._ensure_connection()
        await conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        await conn.execute("DELETE FROM steps WHERE run_id = ?", (run_id,))
        await conn.commit()

    # --- Step Operations ---

    async def save_step(self, step: StepRecord) -> None:
        """Save or update a step."""
        conn = await self._ensure_connection()
        data = self._serialize_model(step)
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data])
        values = list(data.values())
        query = f"""
            INSERT OR REPLACE INTO steps ({columns})
            VALUES ({placeholders})
        """
        await conn.execute(query, values)
        await conn.commit()

    async def save_steps_batch(self, steps: list[StepRecord]) -> None:
        """Batch save steps."""
        if not steps:
            return
        conn = await self._ensure_connection()
        serialized = [self._serialize_model(step) for step in steps]
        all_keys = list(dict.fromkeys(k for s in serialized for k in s))
        columns = ", ".join(all_keys)
        placeholders = ", ".join(["?" for _ in all_keys])
        query = f"""
            INSERT OR REPLACE INTO steps ({columns})
            VALUES ({placeholders})
        """
        await conn.executemany(
            query,
            [tuple(item.get(k) for k in all_keys) for item in serialized],
        )
        await conn.commit()

    async def get_steps(
        self,
        session_id: str,
        start_seq: int | None = None,
        end_seq: int | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 1000,
    ) -> list[StepRecord]:
        """Get steps for a session with optional filtering."""
        conn = await self._ensure_connection()
        query = "SELECT * FROM steps WHERE session_id = ?"
        params: list[str | int | None] = [session_id]

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

        query += " ORDER BY sequence ASC LIMIT ?"
        params.append(limit)

        steps = []
        async with conn.execute(query, params) as cursor:
            async for row in cursor:
                steps.append(self._deserialize_step(row))
        return steps

    async def get_last_step(self, session_id: str) -> StepRecord | None:
        """Get the last step of a session."""
        conn = await self._ensure_connection()
        async with conn.execute(
            "SELECT * FROM steps WHERE session_id = ? ORDER BY sequence DESC LIMIT 1",
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return self._deserialize_step(row)
            return None

    async def delete_steps(self, session_id: str, start_seq: int) -> int:
        """Delete steps from a sequence number onwards."""
        conn = await self._ensure_connection()
        cursor = await conn.execute(
            "DELETE FROM steps WHERE session_id = ? AND sequence >= ?",
            (session_id, start_seq),
        )
        await conn.commit()
        return cursor.rowcount

    async def update_step_condensed_content(
        self,
        session_id: str,
        step_id: str,
        condensed_content: str,
    ) -> bool:
        conn = await self._ensure_connection()
        cursor = await conn.execute(
            "UPDATE steps SET condensed_content = ? WHERE session_id = ? AND id = ?",
            (condensed_content, session_id, step_id),
        )
        await conn.commit()
        return cursor.rowcount > 0

    async def get_step_count(self, session_id: str) -> int:
        """Get total number of steps for a session."""
        conn = await self._ensure_connection()
        async with conn.execute(
            "SELECT COUNT(*) FROM steps WHERE session_id = ?", (session_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def get_committed_step_count(self, session_id: str) -> int:
        conn = await self._ensure_connection()
        kinds = [
            RunLogEntryKind.USER_STEP_COMMITTED.value,
            RunLogEntryKind.ASSISTANT_STEP_COMMITTED.value,
            RunLogEntryKind.TOOL_STEP_COMMITTED.value,
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
        kind_placeholders = ",".join("?" for _ in kinds)
        query = (
            f"SELECT payload FROM run_log_entries WHERE session_id = ? "
            f"AND kind IN ({kind_placeholders})"
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
        query += " ORDER BY sequence ASC LIMIT ?"
        params.append(limit)

        entries: list[RunLogEntry] = []
        async with conn.execute(query, params) as cursor:
            async for row in cursor:
                entries.append(self._deserialize_run_log_entry(row))
        return build_step_views_from_entries(entries)

    async def get_max_sequence(self, session_id: str) -> int:
        """Get the maximum sequence number in the session.

        Returns:
            Maximum sequence number, or 0 if no steps exist
        """
        conn = await self._ensure_connection()
        async with conn.execute(
            "SELECT MAX(sequence) FROM steps WHERE session_id = ?", (session_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row and row[0] is not None else 0

    async def allocate_sequence(self, session_id: str) -> int:
        """Atomically allocate next sequence number using SQLite transactions.
        Thread-safe and concurrent-safe operation.

        Args:
            session_id: Session ID

        Returns:
            Next sequence number (starting from 1)
        """
        conn = await self._ensure_connection()
        await conn.execute("BEGIN IMMEDIATE")
        try:
            async with conn.execute(
                "SELECT sequence FROM counters WHERE session_id = ?", (session_id,)
            ) as cursor:
                row = await cursor.fetchone()

                if row:
                    new_seq = row[0] + 1
                    await conn.execute(
                        "UPDATE counters SET sequence = ? WHERE session_id = ?",
                        (new_seq, session_id),
                    )
                else:
                    max_seq = await self.get_max_sequence(session_id)
                    new_seq = max_seq + 1
                    await conn.execute(
                        "INSERT INTO counters (session_id, sequence) VALUES (?, ?)",
                        (session_id, new_seq),
                    )

                await conn.commit()
                return new_seq
        except Exception:
            await conn.rollback()
            raise

    # --- Batch Query Operations ---

    async def batch_count_runs(self, session_ids: list[str]) -> dict[str, int]:
        if not session_ids:
            return {}
        conn = await self._ensure_connection()
        placeholders = ",".join("?" for _ in session_ids)
        query = (
            f"SELECT session_id, COUNT(*) FROM runs "
            f"WHERE session_id IN ({placeholders}) AND agent_id IS NOT NULL "
            f"GROUP BY session_id"
        )
        result: dict[str, int] = {sid: 0 for sid in session_ids}
        async with conn.execute(query, session_ids) as cursor:
            async for row in cursor:
                result[row[0]] = row[1]
        return result

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

    async def batch_get_step_counts(self, session_ids: list[str]) -> dict[str, int]:
        if not session_ids:
            return {}
        conn = await self._ensure_connection()
        placeholders = ",".join("?" for _ in session_ids)
        query = (
            f"SELECT session_id, COUNT(*) FROM steps "
            f"WHERE session_id IN ({placeholders}) "
            f"GROUP BY session_id"
        )
        result: dict[str, int] = {sid: 0 for sid in session_ids}
        async with conn.execute(query, session_ids) as cursor:
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

    async def batch_get_latest_runs(
        self, session_ids: list[str]
    ) -> dict[str, Run | None]:
        if not session_ids:
            return {}
        conn = await self._ensure_connection()
        placeholders = ",".join("?" for _ in session_ids)
        query = (
            f"SELECT r.* FROM runs r "
            f"INNER JOIN ("
            f"  SELECT session_id, MAX(created_at) AS max_created "
            f"  FROM runs WHERE session_id IN ({placeholders}) AND agent_id IS NOT NULL "
            f"  GROUP BY session_id"
            f") latest ON r.session_id = latest.session_id AND r.created_at = latest.max_created "
            f"WHERE r.agent_id IS NOT NULL"
        )
        result: dict[str, Run | None] = {sid: None for sid in session_ids}
        async with conn.execute(query, session_ids) as cursor:
            async for row in cursor:
                run = self._deserialize_run(row)
                result[run.session_id] = run
        return result

    async def get_latest_run_view(self, session_id: str) -> RunView | None:
        conn = await self._ensure_connection()
        async with conn.execute(
            """
            SELECT run_id FROM run_log_entries
            WHERE session_id = ?
            ORDER BY sequence DESC
            LIMIT 1
            """,
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return await self.get_run_view(row[0])

    async def batch_get_latest_run_views(
        self, session_ids: list[str]
    ) -> dict[str, RunView | None]:
        if not session_ids:
            return {}
        result: dict[str, RunView | None] = {}
        for session_id in session_ids:
            result[session_id] = await self.get_latest_run_view(session_id)
        return result

    async def get_step_by_tool_call_id(
        self,
        session_id: str,
        tool_call_id: str,
    ) -> StepRecord | None:
        """Get a Tool Step by tool_call_id."""
        conn = await self._ensure_connection()
        async with conn.execute(
            "SELECT * FROM steps WHERE session_id = ? AND tool_call_id = ?",
            (session_id, tool_call_id),
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return self._deserialize_step(row)
            return None

    # --- Compact Metadata ---

    async def save_compact_metadata(
        self, session_id: str, agent_id: str, metadata: CompactMetadata
    ) -> None:
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
                self._dumps(metadata.analysis),
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
            ORDER BY created_at DESC LIMIT 1
            """,
            (session_id, agent_id),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_compact_metadata(row)

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
            return [self._row_to_compact_metadata(row) for row in rows]

    def _row_to_compact_metadata(self, row: aiosqlite.Row) -> CompactMetadata:
        return CompactMetadata(
            session_id=row["session_id"],
            agent_id=row["agent_id"],
            start_seq=row["start_seq"],
            end_seq=row["end_seq"],
            before_token_estimate=row["before_token_estimate"],
            after_token_estimate=row["after_token_estimate"],
            message_count=row["message_count"],
            transcript_path=row["transcript_path"],
            analysis=self._loads(row["analysis"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            compact_model=row["compact_model"],
            compact_tokens=row["compact_tokens"],
        )

    @staticmethod
    def _dumps(value) -> str:
        return json.dumps(value)

    @staticmethod
    def _loads(value: str):
        return json.loads(value)


__all__ = ["SQLiteRunStepStorage"]
