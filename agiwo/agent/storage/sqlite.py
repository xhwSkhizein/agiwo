"""
SQLite implementation of RunStepStorage.
"""

import json

import aiosqlite

from agiwo.agent.models.run import Run
from agiwo.agent.models.step import StepRecord
from agiwo.agent.storage.base import RunStepStorage
from agiwo.agent.storage.serialization import (
    deserialize_run_from_storage,
    deserialize_step_from_storage,
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
                    metrics TEXT,
                    created_at TEXT NOT NULL,
                    parent_run_id TEXT,
                    depth INTEGER DEFAULT 0,
                    UNIQUE(session_id, sequence)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS counters (
                    session_id TEXT PRIMARY KEY,
                    sequence INTEGER NOT NULL DEFAULT 0
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
            ],
        )
        await connection.commit()

    async def _ensure_connection(self) -> None:
        """Ensure database connection is established."""
        if self._connection is None:
            await self.connect()

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

    # --- Run Operations ---

    async def save_run(self, run: Run) -> None:
        """Save or update a run."""
        await self._ensure_connection()

        try:
            data = self._serialize_model(run)

            # Build INSERT OR REPLACE query
            columns = ", ".join(data.keys())
            placeholders = ", ".join(["?" for _ in data])
            values = list(data.values())

            query = f"""
                INSERT OR REPLACE INTO runs ({columns})
                VALUES ({placeholders})
            """

            await self._connection.execute(query, values)
            await self._connection.commit()
        except Exception as e:
            logger.error("save_run_failed", error=str(e), run_id=run.id)
            raise

    async def get_run(self, run_id: str) -> Run | None:
        """Get a run by ID."""
        await self._ensure_connection()

        try:
            if self._connection is None:
                raise RuntimeError("Database connection not established")
            async with self._connection.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._deserialize_run(row)
                return None
        except Exception as e:
            logger.error("get_run_failed", error=str(e), run_id=run_id)
            raise

    async def list_runs(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Run]:
        """List runs with filtering and pagination."""
        await self._ensure_connection()

        try:
            query = "SELECT * FROM runs WHERE agent_id IS NOT NULL"
            params = []

            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            if self._connection is None:
                raise RuntimeError("Database connection not established")
            runs = []
            async with self._connection.execute(query, params) as cursor:
                async for row in cursor:
                    runs.append(self._deserialize_run(row))
            return runs
        except Exception as e:
            logger.error("list_runs_failed", error=str(e))
            raise

    async def delete_run(self, run_id: str) -> None:
        """Delete a run and its associated steps."""
        await self._ensure_connection()

        try:
            if self._connection is None:
                raise RuntimeError("Database connection not established")
            await self._connection.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            await self._connection.execute(
                "DELETE FROM steps WHERE run_id = ?", (run_id,)
            )

            await self._connection.commit()
        except Exception as e:
            logger.error("delete_run_failed", error=str(e), run_id=run_id)
            raise

    # --- Step Operations ---

    async def save_step(self, step: StepRecord) -> None:
        """Save or update a step."""
        await self._ensure_connection()

        try:
            data = self._serialize_model(step)

            # Build INSERT OR REPLACE query
            columns = ", ".join(data.keys())
            placeholders = ", ".join(["?" for _ in data])
            values = list(data.values())

            query = f"""
                INSERT OR REPLACE INTO steps ({columns})
                VALUES ({placeholders})
            """

            if self._connection is None:
                raise RuntimeError("Database connection not established")
            await self._connection.execute(query, values)
            await self._connection.commit()
        except Exception as e:
            logger.error(
                "save_step_failed",
                error=str(e),
                step_id=step.id,
                session_id=step.session_id,
                sequence=step.sequence,
            )
            raise

    async def save_steps_batch(self, steps: list[StepRecord]) -> None:
        """Batch save steps."""
        if not steps:
            return

        await self._ensure_connection()

        try:
            serialized = [self._serialize_model(step) for step in steps]
            all_keys = list(dict.fromkeys(k for s in serialized for k in s))
            columns = ", ".join(all_keys)
            placeholders = ", ".join(["?" for _ in all_keys])
            query = f"""
                INSERT OR REPLACE INTO steps ({columns})
                VALUES ({placeholders})
            """
            await self._connection.executemany(
                query,
                [tuple(item.get(k) for k in all_keys) for item in serialized],
            )
            await self._connection.commit()
        except Exception as e:
            logger.error("save_steps_batch_failed", error=str(e), count=len(steps))
            raise

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
        await self._ensure_connection()

        try:
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

            if self._connection is None:
                raise RuntimeError("Database connection not established")
            steps = []
            async with self._connection.execute(query, params) as cursor:
                async for row in cursor:
                    steps.append(self._deserialize_step(row))
            return steps
        except Exception as e:
            logger.error("get_steps_failed", error=str(e), session_id=session_id)
            raise

    async def get_last_step(self, session_id: str) -> StepRecord | None:
        """Get the last step of a session."""
        await self._ensure_connection()

        try:
            if self._connection is None:
                raise RuntimeError("Database connection not established")
            async with self._connection.execute(
                "SELECT * FROM steps WHERE session_id = ? ORDER BY sequence DESC LIMIT 1",
                (session_id,),
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._deserialize_step(row)
                return None
        except Exception as e:
            logger.error("get_last_step_failed", error=str(e), session_id=session_id)
            raise

    async def delete_steps(self, session_id: str, start_seq: int) -> int:
        """Delete steps from a sequence number onwards."""
        await self._ensure_connection()

        try:
            if self._connection is None:
                raise RuntimeError("Database connection not established")
            cursor = await self._connection.execute(
                "DELETE FROM steps WHERE session_id = ? AND sequence >= ?",
                (session_id, start_seq),
            )
            await self._connection.commit()
            return cursor.rowcount
        except Exception as e:
            logger.error("delete_steps_failed", error=str(e), session_id=session_id)
            raise

    async def get_step_count(self, session_id: str) -> int:
        """Get total number of steps for a session."""
        await self._ensure_connection()

        try:
            if self._connection is None:
                raise RuntimeError("Database connection not established")
            async with self._connection.execute(
                "SELECT COUNT(*) FROM steps WHERE session_id = ?", (session_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return row[0] if row else 0
        except Exception as e:
            logger.error("get_step_count_failed", error=str(e), session_id=session_id)
            raise

    async def get_max_sequence(self, session_id: str) -> int:
        """
        Get the maximum sequence number in the session.

        Returns:
            Maximum sequence number, or 0 if no steps exist
        """
        await self._ensure_connection()

        try:
            if self._connection is None:
                raise RuntimeError("Database connection not established")
            async with self._connection.execute(
                "SELECT MAX(sequence) FROM steps WHERE session_id = ?", (session_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return row[0] if row and row[0] is not None else 0
        except Exception as e:
            logger.error("get_max_sequence_failed", error=str(e), session_id=session_id)
            raise

    async def allocate_sequence(self, session_id: str) -> int:
        """
        Atomically allocate next sequence number using SQLite transactions.
        Thread-safe and concurrent-safe operation.

        Args:
            session_id: Session ID

        Returns:
            Next sequence number (starting from 1)
        """
        await self._ensure_connection()

        try:
            if self._connection is None:
                raise RuntimeError("Database connection not established")
            # Use BEGIN IMMEDIATE to acquire write lock immediately
            await self._connection.execute("BEGIN IMMEDIATE")

            try:
                # Try to get existing counter
                async with self._connection.execute(
                    "SELECT sequence FROM counters WHERE session_id = ?", (session_id,)
                ) as cursor:
                    row = await cursor.fetchone()

                    if row:
                        # Increment existing counter
                        new_seq = row[0] + 1
                        await self._connection.execute(
                            "UPDATE counters SET sequence = ? WHERE session_id = ?",
                            (new_seq, session_id),
                        )
                    else:
                        # Initialize counter from steps
                        max_seq = await self.get_max_sequence(session_id)
                        new_seq = max_seq + 1
                        await self._connection.execute(
                            "INSERT INTO counters (session_id, sequence) VALUES (?, ?)",
                            (session_id, new_seq),
                        )

                    await self._connection.commit()
                    return new_seq
            except Exception:
                await self._connection.rollback()
                raise
        except Exception as e:
            logger.error(
                "allocate_sequence_failed", error=str(e), session_id=session_id
            )
            raise

    async def get_step_by_tool_call_id(
        self,
        session_id: str,
        tool_call_id: str,
    ) -> StepRecord | None:
        """Get a Tool Step by tool_call_id."""
        await self._ensure_connection()

        try:
            if self._connection is None:
                raise RuntimeError("Database connection not established")
            async with self._connection.execute(
                "SELECT * FROM steps WHERE session_id = ? AND tool_call_id = ?",
                (session_id, tool_call_id),
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._deserialize_step(row)
                return None
        except Exception as e:
            logger.error(
                "get_step_by_tool_call_id_failed",
                error=str(e),
                tool_call_id=tool_call_id,
            )
            raise

    @staticmethod
    def _dumps(value) -> str:
        return json.dumps(value)

    @staticmethod
    def _loads(value: str):
        return json.loads(value)


__all__ = ["SQLiteRunStepStorage"]
