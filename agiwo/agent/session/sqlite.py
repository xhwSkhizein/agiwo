"""
SQLite implementation of SessionStore.
"""

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any

import aiosqlite

from agiwo.agent.schema import Run, Step
from agiwo.agent.session.base import SessionStore
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class SQLiteSessionStore(SessionStore):
    """
    SQLite implementation of SessionStore.
    """

    def __init__(self, db_path: str = "agiwo.db") -> None:
        self.db_path = db_path
        self._connection: aiosqlite.Connection | None = None
        self._initialized = False

    async def connect(self) -> None:
        """Initialize database connection and create tables."""
        if self._initialized:
            return

        self._connection = await aiosqlite.connect(self.db_path)
        self._connection.row_factory = aiosqlite.Row

        await self._create_tables()
        self._initialized = True

        logger.info("sqlite_connected", db_path=self.db_path)

    async def initialize(self) -> None:
        """Backward-compatible alias for connect()."""
        await self.connect()

    async def disconnect(self) -> None:
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            self._initialized = False

    async def _create_tables(self) -> None:
        """Create database tables and indexes."""
        if not self._connection:
            raise RuntimeError("Database connection not established")

        # Create runs table
        await self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                runnable_type TEXT NOT NULL DEFAULT 'agent',
                session_id TEXT NOT NULL,
                user_id TEXT,
                input_query TEXT NOT NULL,
                status TEXT NOT NULL,
                response_content TEXT,
                metrics TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                parent_run_id TEXT,
                trace_id TEXT
            )
        """
        )

        # Create steps table
        await self._connection.execute(
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
                reasoning_content TEXT,
                tool_calls TEXT,
                tool_call_id TEXT,
                name TEXT,
                metrics TEXT,
                created_at TEXT NOT NULL,
                parent_run_id TEXT,
                trace_id TEXT,
                span_id TEXT,
                parent_span_id TEXT,
                depth INTEGER DEFAULT 0,
                llm_messages TEXT,
                llm_tools TEXT,
                llm_request_params TEXT,
                UNIQUE(session_id, sequence)
            )
        """
        )

        # Create counters table for atomic sequence allocation
        await self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS counters (
                session_id TEXT PRIMARY KEY,
                sequence INTEGER NOT NULL DEFAULT 0
            )
        """
        )

        # Create indexes for runs
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_runs_agent_id ON runs(agent_id)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_runs_user_id ON runs(user_id)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_runs_session_id ON runs(session_id)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at)"
        )

        # Create indexes for steps
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_steps_session_seq ON steps(session_id, sequence)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_steps_session_run_seq "
            "ON steps(session_id, run_id, sequence)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_steps_session_tool_call_id "
            "ON steps(session_id, tool_call_id)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_steps_created_at ON steps(created_at)"
        )

        await self._connection.commit()

        # Migrate existing tables to add missing columns if needed
        await self._migrate_tables()

    async def _migrate_tables(self) -> None:
        """Migrate existing tables to add missing columns if needed."""
        if not self._connection:
            return

        try:
            runs_columns = await self._get_table_columns("runs")
            steps_columns = await self._get_table_columns("steps")

            # --- runs table ---
            await self._ensure_column("runs", "agent_id", "TEXT", runs_columns)
            await self._ensure_column(
                "runs",
                "runnable_type",
                "TEXT NOT NULL DEFAULT 'agent'",
                runs_columns,
            )
            await self._ensure_column("runs", "response_content", "TEXT", runs_columns)
            await self._ensure_column("runs", "metrics", "TEXT", runs_columns)
            await self._ensure_column("runs", "updated_at", "TEXT", runs_columns)
            await self._ensure_column("runs", "parent_run_id", "TEXT", runs_columns)
            await self._ensure_column("runs", "trace_id", "TEXT", runs_columns)

            # Backfill agent_id from runnable_id if present in old schema
            if "runnable_id" in runs_columns and "agent_id" in runs_columns:
                await self._connection.execute(
                    "UPDATE runs SET agent_id = runnable_id WHERE agent_id IS NULL"
                )

            # --- steps table ---
            await self._ensure_column("steps", "agent_id", "TEXT", steps_columns)
            await self._ensure_column(
                "steps",
                "runnable_type",
                "TEXT NOT NULL DEFAULT 'agent'",
                steps_columns,
            )
            await self._ensure_column("steps", "content_for_user", "TEXT", steps_columns)
            await self._ensure_column("steps", "reasoning_content", "TEXT", steps_columns)
            await self._ensure_column("steps", "tool_calls", "TEXT", steps_columns)
            await self._ensure_column("steps", "tool_call_id", "TEXT", steps_columns)
            await self._ensure_column("steps", "name", "TEXT", steps_columns)
            await self._ensure_column("steps", "metrics", "TEXT", steps_columns)
            await self._ensure_column("steps", "parent_run_id", "TEXT", steps_columns)
            await self._ensure_column("steps", "trace_id", "TEXT", steps_columns)
            await self._ensure_column("steps", "span_id", "TEXT", steps_columns)
            await self._ensure_column("steps", "parent_span_id", "TEXT", steps_columns)
            await self._ensure_column("steps", "depth", "INTEGER DEFAULT 0", steps_columns)
            await self._ensure_column("steps", "llm_messages", "TEXT", steps_columns)
            await self._ensure_column("steps", "llm_tools", "TEXT", steps_columns)
            await self._ensure_column(
                "steps", "llm_request_params", "TEXT", steps_columns
            )

            if "runnable_id" in steps_columns and "agent_id" in steps_columns:
                await self._connection.execute(
                    "UPDATE steps SET agent_id = runnable_id WHERE agent_id IS NULL"
                )

            # Ensure indexes exist for new columns
            await self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_agent_id ON runs(agent_id)"
            )
            await self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_steps_session_run_seq "
                "ON steps(session_id, run_id, sequence)"
            )

            await self._connection.commit()
        except Exception as e:
            logger.debug("migration_skipped", error=str(e))

    async def _get_table_columns(self, table_name: str) -> set[str]:
        if not self._connection:
            return set()
        async with self._connection.execute(f"PRAGMA table_info({table_name})") as cursor:
            rows = await cursor.fetchall()
        return {row[1] for row in rows}

    async def _ensure_column(
        self,
        table_name: str,
        column_name: str,
        column_def: str,
        existing: set[str],
    ) -> None:
        if column_name in existing:
            return
        await self._connection.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}"
        )
        existing.add(column_name)

    async def _ensure_connection(self) -> None:
        """Ensure database connection is established."""
        if not self._initialized:
            await self.connect()

    def _serialize_model(self, model: Run | Step) -> dict:
        """Serialize Pydantic model or dataclass to dict, handling nested models."""
        data = self._model_to_dict(model)
        # Convert nested models to JSON strings
        if isinstance(model, Run) and model.metrics:
            data["metrics"] = self._serialize_metrics(model.metrics)
        elif isinstance(model, Step) and model.metrics:
            data["metrics"] = self._serialize_metrics(model.metrics)

        # Convert list/dict fields to JSON strings
        if isinstance(model, Step):
            if model.tool_calls:
                data["tool_calls"] = json.dumps(model.tool_calls)
            if model.llm_messages:
                data["llm_messages"] = json.dumps(model.llm_messages)
            if model.llm_tools:
                data["llm_tools"] = json.dumps(model.llm_tools)
            if model.llm_request_params:
                data["llm_request_params"] = json.dumps(model.llm_request_params)

        # Convert datetime to ISO format string
        for field_name in ("created_at", "updated_at"):
            if field_name in data and not isinstance(data[field_name], str):
                data[field_name] = data[field_name].isoformat()

        # Map Run model fields to database columns
        if isinstance(model, Run):
            # Map agent_id to agent_id for database compatibility
            if "agent_id" in data:
                data["agent_id"] = data.pop("agent_id")
                data["runnable_type"] = "agent"
        elif isinstance(model, Step):
            # Map agent_id to agent_id for database compatibility
            if "agent_id" in data:
                data["agent_id"] = data.pop("agent_id")
                data["runnable_type"] = "agent"

        return data

    def _model_to_dict(self, model: Run | Step) -> dict:
        if is_dataclass(model):
            data = asdict(model)
            return {k: v for k, v in data.items() if v is not None}
        return model.model_dump(mode="json", exclude_none=True)

    def _serialize_metrics(self, metrics: Any) -> str:
        if is_dataclass(metrics):
            payload = asdict(metrics)
        else:
            payload = metrics.model_dump(mode="json")
        return json.dumps(self._normalize_datetimes(payload))

    def _normalize_datetimes(self, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return {k: self._normalize_datetimes(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._normalize_datetimes(v) for v in value]
        if isinstance(value, tuple):
            return [self._normalize_datetimes(v) for v in value]
        return value

    def _deserialize_run(self, row: aiosqlite.Row) -> Run:
        """Deserialize database row to Run model."""
        data = dict(row)

        # Parse JSON fields
        if data.get("metrics"):
            metrics_data = json.loads(data["metrics"])
            from agiwo.agent.schema import RunMetrics

            # RunMetrics is a dataclass
            data["metrics"] = RunMetrics(**metrics_data)

        # Map database columns to Run model fields
        if "agent_id" in data:
            data["agent_id"] = data.pop("agent_id")
        # Remove runnable_type as it's not in the Run model
        data.pop("runnable_type", None)

        # Run is a dataclass, so use direct instantiation
        return Run(**data)

    def _deserialize_step(self, row: aiosqlite.Row) -> Step:
        """Deserialize database row to Step model."""
        data = dict(row)

        # Parse JSON fields
        if data.get("metrics"):
            metrics_data = json.loads(data["metrics"])
            from agiwo.agent.schema import StepMetrics

            # StepMetrics is a dataclass
            data["metrics"] = StepMetrics(**metrics_data)
        if data.get("tool_calls"):
            data["tool_calls"] = json.loads(data["tool_calls"])
        if data.get("llm_messages"):
            data["llm_messages"] = json.loads(data["llm_messages"])
        if data.get("llm_tools"):
            data["llm_tools"] = json.loads(data["llm_tools"])
        if data.get("llm_request_params"):
            data["llm_request_params"] = json.loads(data["llm_request_params"])

        # Map database columns to Step model fields
        if "agent_id" in data:
            data["agent_id"] = data.pop("agent_id")
        # Remove runnable_type as it's not in the Step model
        data.pop("runnable_type", None)

        # Convert role string to MessageRole enum
        if "role" in data and isinstance(data["role"], str):
            from agiwo.agent.schema import MessageRole

            data["role"] = MessageRole(data["role"])

        # Step is a dataclass, so use direct instantiation
        return Step(**data)

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
            run = await self.get_run(run_id)
            await self._connection.execute("DELETE FROM runs WHERE id = ?", (run_id,))

            if run and run.session_id:
                await self._connection.execute(
                    "DELETE FROM steps WHERE session_id = ?", (run.session_id,)
                )

            await self._connection.commit()
        except Exception as e:
            logger.error("delete_run_failed", error=str(e), run_id=run_id)
            raise

    # --- Step Operations ---

    async def save_step(self, step: Step) -> None:
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

    async def save_steps_batch(self, steps: list[Step]) -> None:
        """Batch save steps."""
        if not steps:
            return

        await self._ensure_connection()

        try:
            for step in steps:
                await self.save_step(step)
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
    ) -> list[Step]:
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

    async def get_last_step(self, session_id: str) -> Step | None:
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
    ) -> Step | None:
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


__all__ = ["SQLiteSessionStore"]
