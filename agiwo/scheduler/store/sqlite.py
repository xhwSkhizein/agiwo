"""SQLite-backed scheduler state storage."""

import json
from datetime import datetime, timedelta, timezone

import aiosqlite

from agiwo.agent.input_codec import deserialize_user_input, serialize_user_input
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    PendingEvent,
    SchedulerEventType,
    TimeUnit,
    WaitMode,
    WakeCondition,
    WakeType,
)
from agiwo.scheduler.store.base import AgentStateStorage
from agiwo.scheduler.store.semantics import RECENT_STEPS_MAX, is_wakeable_state
from agiwo.utils.storage_support.sqlite_runtime import (
    SQLiteConnectionRuntime,
    execute_statements,
)
from agiwo.utils.logging import get_logger

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
                    config_overrides TEXT DEFAULT '{}',
                    wake_type TEXT,
                    wake_time_value REAL,
                    wake_time_unit TEXT,
                    wake_wait_for TEXT DEFAULT '[]',
                    wake_wait_mode TEXT DEFAULT 'all',
                    wake_completed_ids TEXT DEFAULT '[]',
                    wake_submitted_task TEXT,
                    wakeup_at TEXT,
                    wake_timeout_at TEXT,
                    result_summary TEXT,
                    signal_propagated INTEGER DEFAULT 0,
                    is_persistent INTEGER DEFAULT 0,
                    depth INTEGER DEFAULT 0,
                    wake_count INTEGER DEFAULT 0,
                    agent_config_id TEXT,
                    explain TEXT,
                    last_activity_at TEXT,
                    recent_steps_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """,
                "CREATE INDEX IF NOT EXISTS idx_agent_states_status ON agent_states(status)",
                "CREATE INDEX IF NOT EXISTS idx_agent_states_parent ON agent_states(parent_id)",
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
            time_unit = None
            if row["wake_time_unit"]:
                time_unit = TimeUnit(row["wake_time_unit"])
            wake_condition = WakeCondition(
                type=WakeType(row["wake_type"]),
                wait_for=json.loads(row["wake_wait_for"]),
                wait_mode=WaitMode(row["wake_wait_mode"]),
                completed_ids=json.loads(row["wake_completed_ids"]),
                time_value=row["wake_time_value"],
                time_unit=time_unit,
                wakeup_at=datetime.fromisoformat(row["wakeup_at"])
                if row["wakeup_at"]
                else None,
                submitted_task=deserialize_user_input(row["wake_submitted_task"])
                if row["wake_submitted_task"]
                else None,
                timeout_at=datetime.fromisoformat(row["wake_timeout_at"])
                if row["wake_timeout_at"]
                else None,
            )

        config_overrides = {}
        if row["config_overrides"]:
            config_overrides = json.loads(row["config_overrides"])

        recent_steps = None
        if row.get("recent_steps_json"):
            try:
                recent_steps = json.loads(row["recent_steps_json"])
            except (json.JSONDecodeError, TypeError):
                recent_steps = None

        last_activity_at = None
        if row.get("last_activity_at"):
            last_activity_at = datetime.fromisoformat(row["last_activity_at"])

        return AgentState(
            id=row["id"],
            session_id=row["session_id"],
            status=AgentStateStatus(row["status"]),
            task=deserialize_user_input(row["task"]),
            parent_id=row.get("parent_id"),
            config_overrides=config_overrides,
            wake_condition=wake_condition,
            result_summary=row.get("result_summary"),
            signal_propagated=bool(row.get("signal_propagated", 0)),
            agent_config_id=row.get("agent_config_id"),
            is_persistent=bool(row.get("is_persistent", 0)),
            depth=row.get("depth", 0) or 0,
            wake_count=row.get("wake_count", 0) or 0,
            explain=row.get("explain"),
            last_activity_at=last_activity_at,
            recent_steps=recent_steps,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def _wake_condition_columns(self) -> list[str]:
        return [
            "wake_type",
            "wake_time_value",
            "wake_time_unit",
            "wake_wait_for",
            "wake_wait_mode",
            "wake_completed_ids",
            "wake_submitted_task",
            "wakeup_at",
            "wake_timeout_at",
        ]

    def _wake_condition_values(self, wc: WakeCondition | None) -> list:
        if wc is None:
            return [None, None, None, "[]", "all", "[]", None, None, None]
        return [
            wc.type.value,
            wc.time_value,
            wc.time_unit.value if wc.time_unit else None,
            json.dumps(wc.wait_for),
            wc.wait_mode.value,
            json.dumps(wc.completed_ids),
            serialize_user_input(wc.submitted_task)
            if wc.submitted_task is not None
            else None,
            wc.wakeup_at.isoformat() if wc.wakeup_at else None,
            wc.timeout_at.isoformat() if wc.timeout_at else None,
        ]

    async def save_state(self, state: AgentState) -> None:
        conn = await self._get_conn()
        wc_vals = self._wake_condition_values(state.wake_condition)
        await conn.execute(
            """
            INSERT OR REPLACE INTO agent_states
                (id, session_id, parent_id,
                 status, task, config_overrides,
                 wake_type, wake_time_value, wake_time_unit,
                 wake_wait_for, wake_wait_mode, wake_completed_ids,
                 wake_submitted_task, wakeup_at, wake_timeout_at,
                 result_summary, signal_propagated,
                 agent_config_id, is_persistent, depth, wake_count,
                 explain, last_activity_at, recent_steps_json,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                state.id,
                state.session_id,
                state.parent_id,
                state.status.value,
                serialize_user_input(state.task),
                json.dumps(state.config_overrides),
                *wc_vals,
                state.result_summary,
                1 if state.signal_propagated else 0,
                state.agent_config_id,
                1 if state.is_persistent else 0,
                state.depth,
                state.wake_count,
                state.explain,
                state.last_activity_at.isoformat() if state.last_activity_at else None,
                json.dumps(state.recent_steps)
                if state.recent_steps is not None
                else None,
                state.created_at.isoformat(),
                state.updated_at.isoformat(),
            ),
        )
        await conn.commit()

    async def get_state(self, state_id: str) -> AgentState | None:
        conn = await self._get_conn()
        cursor = await conn.execute("SELECT * FROM agent_states WHERE id = ?", (state_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_state(dict(row))

    async def update_status(
        self,
        state_id: str,
        status: AgentStateStatus,
        *,
        wake_condition: WakeCondition | None = ...,
        result_summary: str | None = ...,
        explain: str | None = ...,
        last_activity_at: datetime | None = ...,
        recent_steps: list[dict] | None = ...,
    ) -> None:
        conn = await self._get_conn()
        now = datetime.now(timezone.utc).isoformat()

        sets = ["status = ?", "updated_at = ?"]
        params: list = [status.value, now]

        if wake_condition is not ...:
            wc_cols = self._wake_condition_columns()
            wc_vals = self._wake_condition_values(wake_condition)
            for col, val in zip(wc_cols, wc_vals):
                sets.append(f"{col} = ?")
                params.append(val)

        if result_summary is not ...:
            sets.append("result_summary = ?")
            params.append(result_summary)

        if explain is not ...:
            sets.append("explain = ?")
            params.append(explain)

        if last_activity_at is not ...:
            sets.append("last_activity_at = ?")
            params.append(last_activity_at.isoformat() if last_activity_at else None)

        if recent_steps is not ...:
            sets.append("recent_steps_json = ?")
            params.append(json.dumps(recent_steps) if recent_steps is not None else None)

        params.append(state_id)
        sql = f"UPDATE agent_states SET {', '.join(sets)} WHERE id = ?"
        await conn.execute(sql, params)
        await conn.commit()

    async def get_states_by_parent(self, parent_id: str) -> list[AgentState]:
        conn = await self._get_conn()
        cursor = await conn.execute(
            "SELECT * FROM agent_states WHERE parent_id = ?",
            (parent_id,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_state(dict(r)) for r in rows]

    async def find_pending(self) -> list[AgentState]:
        conn = await self._get_conn()
        cursor = await conn.execute(
            "SELECT * FROM agent_states WHERE status = ?",
            (AgentStateStatus.PENDING.value,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_state(dict(r)) for r in rows]

    async def find_wakeable(self, now: datetime) -> list[AgentState]:
        conn = await self._get_conn()
        cursor = await conn.execute(
            "SELECT * FROM agent_states WHERE status = ?",
            (AgentStateStatus.SLEEPING.value,),
        )
        rows = await cursor.fetchall()
        results: list[AgentState] = []
        for row in rows:
            state = self._row_to_state(dict(row))
            if is_wakeable_state(state, now):
                results.append(state)
        return results

    async def find_unpropagated_completed(self) -> list[AgentState]:
        conn = await self._get_conn()
        cursor = await conn.execute(
            """
            SELECT * FROM agent_states
            WHERE status IN (?, ?)
              AND signal_propagated = 0
              AND parent_id IS NOT NULL
            """,
            (AgentStateStatus.COMPLETED.value, AgentStateStatus.FAILED.value),
        )
        rows = await cursor.fetchall()
        return [self._row_to_state(dict(r)) for r in rows]

    async def mark_child_completed(self, parent_id: str, child_id: str) -> None:
        conn = await self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        cursor = await conn.execute(
            "SELECT wake_completed_ids FROM agent_states WHERE id = ?",
            (parent_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return

        completed_ids: list[str] = json.loads(row["wake_completed_ids"] or "[]")
        if child_id not in completed_ids:
            completed_ids.append(child_id)

        await conn.execute(
            "UPDATE agent_states SET wake_completed_ids = ?, updated_at = ? WHERE id = ?",
            (json.dumps(completed_ids), now, parent_id),
        )
        await conn.commit()

    async def mark_propagated(self, state_id: str) -> None:
        conn = await self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        await conn.execute(
            "UPDATE agent_states SET signal_propagated = 1, updated_at = ? WHERE id = ?",
            (now, state_id),
        )
        await conn.commit()

    async def find_timed_out(self, now: datetime) -> list[AgentState]:
        conn = await self._get_conn()
        now_iso = now.isoformat()
        cursor = await conn.execute(
            """
            SELECT * FROM agent_states
            WHERE status = ?
              AND wake_timeout_at IS NOT NULL
              AND wake_timeout_at <= ?
            """,
            (AgentStateStatus.SLEEPING.value, now_iso),
        )
        rows = await cursor.fetchall()
        return [self._row_to_state(dict(r)) for r in rows]

    async def increment_wake_count(self, state_id: str) -> None:
        conn = await self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        await conn.execute(
            "UPDATE agent_states SET wake_count = wake_count + 1, updated_at = ? WHERE id = ?",
            (now, state_id),
        )
        await conn.commit()

    async def list_all(
        self,
        *,
        status: AgentStateStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AgentState]:
        conn = await self._get_conn()
        if status is not None:
            cursor = await conn.execute(
                "SELECT * FROM agent_states WHERE status = ? ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (status.value, limit, offset),
            )
        else:
            cursor = await conn.execute(
                "SELECT * FROM agent_states ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        rows = await cursor.fetchall()
        return [self._row_to_state(dict(r)) for r in rows]

    async def find_running(self) -> list[AgentState]:
        conn = await self._get_conn()
        cursor = await conn.execute(
            "SELECT * FROM agent_states WHERE status = ?",
            (AgentStateStatus.RUNNING.value,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_state(dict(r)) for r in rows]

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
                json.dumps(event.payload),
                event.source_agent_id,
                event.created_at.isoformat(),
            ),
        )
        await conn.commit()

    async def get_pending_events(
        self, target_agent_id: str, session_id: str
    ) -> list[PendingEvent]:
        conn = await self._get_conn()
        cursor = await conn.execute(
            """
            SELECT * FROM pending_events
            WHERE target_agent_id = ? AND session_id = ?
            ORDER BY created_at ASC
            """,
            (target_agent_id, session_id),
        )
        rows = await cursor.fetchall()
        return [self._row_to_event(dict(r)) for r in rows]

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

    async def delete_events_by_agent(self, target_agent_id: str) -> None:
        conn = await self._get_conn()
        await conn.execute(
            "DELETE FROM pending_events WHERE target_agent_id = ?",
            (target_agent_id,),
        )
        await conn.commit()

    async def append_recent_step(self, state_id: str, step: dict) -> None:
        conn = await self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        step_json = json.dumps(step)
        await conn.execute(
            f"""
            UPDATE agent_states
            SET recent_steps_json = (
                SELECT json(
                    CASE
                        WHEN json_array_length(COALESCE(recent_steps_json, '[]')) >= {RECENT_STEPS_MAX}
                        THEN json_remove(
                            json_insert(COALESCE(recent_steps_json, '[]'), '$[#]', json(?)),
                            '$[0]'
                        )
                        ELSE json_insert(COALESCE(recent_steps_json, '[]'), '$[#]', json(?))
                    END
                )
            ),
            last_activity_at = ?,
            updated_at = ?
            WHERE id = ?
            """,
            (step_json, step_json, now, now, state_id),
        )
        await conn.commit()

    async def find_agents_with_debounced_events(
        self,
        min_count: int,
        max_wait_seconds: float,
        now: datetime,
    ) -> list[tuple[str, str]]:
        conn = await self._get_conn()
        cutoff = (now - timedelta(seconds=max_wait_seconds)).isoformat()
        cursor = await conn.execute(
            """
            SELECT target_agent_id, session_id
            FROM pending_events
            GROUP BY target_agent_id, session_id
            HAVING COUNT(*) >= ? OR MIN(created_at) <= ?
            """,
            (min_count, cutoff),
        )
        rows = await cursor.fetchall()
        return [(row[0], row[1]) for row in rows]

    async def has_recent_health_warning(
        self,
        target_agent_id: str,
        source_agent_id: str,
        within_seconds: float,
        now: datetime,
    ) -> bool:
        conn = await self._get_conn()
        cutoff = (now - timedelta(seconds=within_seconds)).isoformat()
        cursor = await conn.execute(
            """
            SELECT COUNT(*) FROM pending_events
            WHERE target_agent_id = ?
              AND source_agent_id = ?
              AND event_type = ?
              AND created_at >= ?
            """,
            (
                target_agent_id,
                source_agent_id,
                SchedulerEventType.HEALTH_WARNING.value,
                cutoff,
            ),
        )
        row = await cursor.fetchone()
        return bool(row and row[0] > 0)

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

    async def close(self) -> None:
        if self._conn is not None:
            await self._runtime.disconnect()
            self._conn = None
