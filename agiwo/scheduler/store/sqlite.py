"""SQLite-backed scheduler state storage."""

import json
from collections.abc import Collection
from datetime import datetime

import aiosqlite

from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    PendingEvent,
    SchedulerEventType,
    normalize_statuses,
    thaw_value,
)
from agiwo.scheduler.store.base import AgentStateStorage
from agiwo.scheduler.store.codec import (
    deserialize_scheduler_run_result_for_store,
    deserialize_user_input_for_store,
    deserialize_wake_condition_for_store,
    serialize_scheduler_run_result_for_store,
    serialize_user_input_for_store,
    serialize_wake_condition_for_store,
)
from agiwo.utils.logging import get_logger
from agiwo.utils.storage_support.sqlite_runtime import (
    SQLiteConnectionRuntime,
    execute_statements,
)

logger = get_logger(__name__)


class SQLiteAgentStateStorage(AgentStateStorage):
    """SQLite-backed agent state storage."""

    def __init__(self, db_path: str = "scheduler.db") -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None
        self._runtime = SQLiteConnectionRuntime(
            db_path=db_path,
            logger=logger,
            connect_event="sqlite_agent_state_storage_connected",
        )

    async def _get_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            self._conn = await self._runtime.ensure_connection(self._initialize_schema)
        return self._conn

    async def _initialize_schema(self, conn: aiosqlite.Connection) -> None:
        await execute_statements(
            conn,
            [
                """
                CREATE TABLE IF NOT EXISTS agent_states (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    parent_id TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    task TEXT NOT NULL,
                    pending_input TEXT,
                    config_overrides TEXT DEFAULT '{}',
                    wake_type TEXT,
                    wake_time_value REAL,
                    wake_time_unit TEXT,
                    wake_wait_for TEXT DEFAULT '[]',
                    wake_wait_mode TEXT DEFAULT 'all',
                    wake_completed_ids TEXT DEFAULT '[]',
                    wakeup_at TEXT,
                    wake_timeout_at TEXT,
                    result_summary TEXT,
                    last_run_id TEXT,
                    last_run_termination_reason TEXT,
                    last_run_summary TEXT,
                    last_run_error TEXT,
                    last_run_completed_at TEXT,
                    signal_propagated INTEGER DEFAULT 0,
                    is_persistent INTEGER DEFAULT 0,
                    depth INTEGER DEFAULT 0,
                    wake_count INTEGER DEFAULT 0,
                    rollback_count INTEGER DEFAULT 0,
                    agent_config_id TEXT,
                    explain TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """,
                "CREATE INDEX IF NOT EXISTS idx_agent_states_status ON agent_states(status)",
                "CREATE INDEX IF NOT EXISTS idx_agent_states_parent ON agent_states(parent_id)",
                "CREATE INDEX IF NOT EXISTS idx_agent_states_session ON agent_states(session_id)",
                """
                CREATE TABLE IF NOT EXISTS pending_events (
                    id TEXT PRIMARY KEY,
                    target_agent_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    source_agent_id TEXT,
                    created_at TEXT NOT NULL
                )
                """,
                "CREATE INDEX IF NOT EXISTS idx_pending_events_target ON pending_events(target_agent_id, session_id)",
                "CREATE INDEX IF NOT EXISTS idx_pending_events_created ON pending_events(created_at)",
            ],
        )
        await conn.commit()

    def _row_to_state(self, row: dict) -> AgentState:
        wake_condition = None
        if row["wake_type"]:
            wake_condition = deserialize_wake_condition_for_store(
                {
                    "type": row["wake_type"],
                    "wait_for": json.loads(row["wake_wait_for"]),
                    "wait_mode": row["wake_wait_mode"],
                    "completed_ids": json.loads(row["wake_completed_ids"]),
                    "time_value": row["wake_time_value"],
                    "time_unit": row["wake_time_unit"],
                    "wakeup_at": row["wakeup_at"],
                    "timeout_at": row["wake_timeout_at"],
                }
            )

        config_overrides = {}
        if row["config_overrides"]:
            config_overrides = json.loads(row["config_overrides"])

        last_run_result = deserialize_scheduler_run_result_for_store(
            {
                "run_id": row["last_run_id"],
                "termination_reason": row["last_run_termination_reason"],
                "summary": row["last_run_summary"],
                "error": row["last_run_error"],
                "completed_at": row["last_run_completed_at"],
            }
            if row["last_run_termination_reason"]
            else None
        )

        return AgentState(
            id=row["id"],
            session_id=row["session_id"],
            status=AgentStateStatus(row["status"]),
            task=deserialize_user_input_for_store(row["task"]),
            parent_id=row.get("parent_id"),
            pending_input=(
                deserialize_user_input_for_store(row["pending_input"])
                if row.get("pending_input")
                else None
            ),
            config_overrides=config_overrides,
            wake_condition=wake_condition,
            result_summary=row.get("result_summary"),
            signal_propagated=bool(row.get("signal_propagated", 0)),
            agent_config_id=row.get("agent_config_id"),
            is_persistent=bool(row.get("is_persistent", 0)),
            depth=row.get("depth", 0) or 0,
            wake_count=row.get("wake_count", 0) or 0,
            rollback_count=row.get("rollback_count", 0) or 0,
            explain=row.get("explain"),
            last_run_result=last_run_result,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def _row_to_event(self, row: dict) -> PendingEvent:
        return PendingEvent(
            id=row["id"],
            target_agent_id=row["target_agent_id"],
            session_id=row["session_id"],
            event_type=SchedulerEventType(row["event_type"]),
            payload=json.loads(row["payload_json"]),
            source_agent_id=row.get("source_agent_id"),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _wake_condition_columns(self) -> list[str]:
        return [
            "wake_type",
            "wake_time_value",
            "wake_time_unit",
            "wake_wait_for",
            "wake_wait_mode",
            "wake_completed_ids",
            "wakeup_at",
            "wake_timeout_at",
        ]

    def _wake_condition_values(self, wake_condition) -> list[object]:
        payload = serialize_wake_condition_for_store(wake_condition)
        if payload is None:
            return [None, None, None, "[]", "all", "[]", None, None]
        return [
            payload["type"],
            payload.get("time_value"),
            payload.get("time_unit"),
            json.dumps(payload.get("wait_for", [])),
            payload.get("wait_mode", "all"),
            json.dumps(payload.get("completed_ids", [])),
            payload.get("wakeup_at"),
            payload.get("timeout_at"),
        ]

    async def save_state(self, state: AgentState) -> None:
        conn = await self._get_conn()
        wake_values = self._wake_condition_values(state.wake_condition)
        last_run_values = self._last_run_result_values(state.last_run_result)
        await conn.execute(
            """
            INSERT OR REPLACE INTO agent_states
                (id, session_id, parent_id, status, task, pending_input, config_overrides,
                 wake_type, wake_time_value, wake_time_unit, wake_wait_for, wake_wait_mode,
                 wake_completed_ids, wakeup_at, wake_timeout_at, result_summary,
                 last_run_id, last_run_termination_reason, last_run_summary, last_run_error,
                 last_run_completed_at,
                 signal_propagated, is_persistent, depth, wake_count, rollback_count,
                 agent_config_id, explain, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                state.id,
                state.session_id,
                state.parent_id,
                state.status.value,
                serialize_user_input_for_store(state.task),
                (
                    serialize_user_input_for_store(state.pending_input)
                    if state.pending_input is not None
                    else None
                ),
                json.dumps(thaw_value(state.config_overrides)),
                *wake_values,
                state.result_summary,
                *last_run_values,
                1 if state.signal_propagated else 0,
                1 if state.is_persistent else 0,
                state.depth,
                state.wake_count,
                state.rollback_count,
                state.agent_config_id,
                state.explain,
                state.created_at.isoformat(),
                state.updated_at.isoformat(),
            ),
        )
        await conn.commit()

    def _last_run_result_values(self, last_run_result) -> list[object]:
        payload = serialize_scheduler_run_result_for_store(last_run_result)
        if payload is None:
            return [None, None, None, None, None]
        return [
            payload.get("run_id"),
            payload["termination_reason"],
            payload.get("summary"),
            payload.get("error"),
            payload["completed_at"],
        ]

    async def get_state(self, state_id: str) -> AgentState | None:
        conn = await self._get_conn()
        cursor = await conn.execute(
            "SELECT * FROM agent_states WHERE id = ?", (state_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_state(dict(row))

    async def list_states(
        self,
        *,
        statuses: Collection[AgentStateStatus] | None = None,
        parent_id: str | None = None,
        session_id: str | None = None,
        signal_propagated: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AgentState]:
        conn = await self._get_conn()
        clauses: list[str] = []
        params: list[object] = []

        status_filter = normalize_statuses(statuses)
        if status_filter:
            placeholders = ",".join("?" for _ in status_filter)
            clauses.append(f"status IN ({placeholders})")
            params.extend(status.value for status in status_filter)
        if parent_id is not None:
            clauses.append("parent_id = ?")
            params.append(parent_id)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if signal_propagated is not None:
            clauses.append("signal_propagated = ?")
            params.append(1 if signal_propagated else 0)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, offset])
        cursor = await conn.execute(
            f"""
            SELECT * FROM agent_states
            {where_sql}
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
            """,
            params,
        )
        rows = await cursor.fetchall()
        return [self._row_to_state(dict(row)) for row in rows]

    async def save_event(self, event: PendingEvent) -> None:
        conn = await self._get_conn()
        await conn.execute(
            """
            INSERT OR REPLACE INTO pending_events
                (id, target_agent_id, session_id, event_type, payload_json, source_agent_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                event.target_agent_id,
                event.session_id,
                event.event_type.value,
                json.dumps(thaw_value(event.payload)),
                event.source_agent_id,
                event.created_at.isoformat(),
            ),
        )
        await conn.commit()

    async def list_events(
        self,
        *,
        target_agent_id: str | None = None,
        session_id: str | None = None,
    ) -> list[PendingEvent]:
        conn = await self._get_conn()
        clauses: list[str] = []
        params: list[object] = []
        if target_agent_id is not None:
            clauses.append("target_agent_id = ?")
            params.append(target_agent_id)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cursor = await conn.execute(
            f"""
            SELECT * FROM pending_events
            {where_sql}
            ORDER BY created_at ASC
            """,
            params,
        )
        rows = await cursor.fetchall()
        return [self._row_to_event(dict(row)) for row in rows]

    async def delete_events(self, event_ids: list[str]) -> None:
        if not event_ids:
            return
        conn = await self._get_conn()
        placeholders = ",".join("?" * len(event_ids))
        await conn.execute(
            f"DELETE FROM pending_events WHERE id IN ({placeholders})",
            event_ids,
        )
        await conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._runtime.disconnect()
            self._conn = None
