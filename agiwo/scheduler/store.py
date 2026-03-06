"""
Agent State Storage — ABC and implementations.

Provides persistence for AgentState records and PendingEvent records used by the Scheduler.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta

import aiosqlite

from agiwo.agent.schema import deserialize_user_input, serialize_user_input
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    AgentStateStorageConfig,
    PendingEvent,
    SchedulerEventType,
    TimeUnit,
    WaitMode,
    WakeCondition,
    WakeType,
)
from agiwo.utils.logging import get_logger
from agiwo.utils.sqlite_pool import get_shared_connection, release_shared_connection

logger = get_logger(__name__)

_RECENT_STEPS_MAX = 10


class AgentStateStorage(ABC):
    """Abstract base for agent state persistence."""

    @abstractmethod
    async def save_state(self, state: AgentState) -> None:
        """Save or upsert an AgentState."""
        ...

    @abstractmethod
    async def get_state(self, state_id: str) -> AgentState | None:
        """Get an AgentState by id."""
        ...

    @abstractmethod
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
        """Update status and optional fields. Sentinel `...` means 'do not change'."""
        ...

    @abstractmethod
    async def get_states_by_parent(self, parent_id: str) -> list[AgentState]:
        """Get all child states for a given parent id."""
        ...

    @abstractmethod
    async def find_pending(self) -> list[AgentState]:
        """Find all PENDING states."""
        ...

    @abstractmethod
    async def find_wakeable(self, now: datetime) -> list[AgentState]:
        """Find SLEEPING states whose wake condition is satisfied."""
        ...

    @abstractmethod
    async def find_unpropagated_completed(self) -> list[AgentState]:
        """Find COMPLETED/FAILED states with signal_propagated=False and parent_id is not None."""
        ...

    @abstractmethod
    async def mark_child_completed(self, parent_id: str, child_id: str) -> None:
        """Append child_id to parent's wake_condition.completed_ids."""
        ...

    @abstractmethod
    async def mark_propagated(self, state_id: str) -> None:
        """Mark a state's signal as propagated."""
        ...

    @abstractmethod
    async def find_timed_out(self, now: datetime) -> list[AgentState]:
        """Find SLEEPING states whose wake_condition has timed out."""
        ...

    @abstractmethod
    async def increment_wake_count(self, state_id: str) -> None:
        """Atomically increment wake_count for the given state."""
        ...

    @abstractmethod
    async def list_all(
        self,
        *,
        status: AgentStateStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AgentState]:
        """List all states, optionally filtered by status, ordered by updated_at desc."""
        ...

    @abstractmethod
    async def find_running(self) -> list[AgentState]:
        """Find all RUNNING states."""
        ...

    # -- PendingEvent methods ------------------------------------------------

    @abstractmethod
    async def save_event(self, event: PendingEvent) -> None:
        """Persist a PendingEvent."""
        ...

    @abstractmethod
    async def get_pending_events(
        self, target_agent_id: str, session_id: str
    ) -> list[PendingEvent]:
        """Get all pending events for a given agent+session, ordered by created_at asc."""
        ...

    @abstractmethod
    async def delete_events(self, event_ids: list[str]) -> None:
        """Delete events by their ids."""
        ...

    @abstractmethod
    async def find_agents_with_debounced_events(
        self,
        min_count: int,
        max_wait_seconds: float,
        now: datetime,
    ) -> list[tuple[str, str]]:
        """Return (agent_id, session_id) pairs whose pending events meet the debounce threshold.

        Threshold is met when:
            COUNT(events) >= min_count
            OR MIN(created_at) <= now - max_wait_seconds
        """
        ...

    @abstractmethod
    async def append_recent_step(self, state_id: str, step: dict) -> None:
        """Atomically append a step summary to recent_steps (capped at _RECENT_STEPS_MAX).

        Updates last_activity_at at the same time.
        """
        ...

    @abstractmethod
    async def delete_events_by_agent(self, target_agent_id: str) -> None:
        """Delete all pending events for a given agent_id regardless of session."""
        ...

    @abstractmethod
    async def has_recent_health_warning(
        self,
        target_agent_id: str,
        source_agent_id: str,
        within_seconds: float,
        now: datetime,
    ) -> bool:
        """Check if there is already a recent HEALTH_WARNING event from source to target."""
        ...

    async def close(self) -> None:
        """Close storage and release resources."""
        pass


# ═══════════════════════════════════════════════════════════════════════════
# In-Memory Implementation (for testing)
# ═══════════════════════════════════════════════════════════════════════════


class InMemoryAgentStateStorage(AgentStateStorage):
    """In-memory implementation for testing."""

    def __init__(self) -> None:
        self._states: dict[str, AgentState] = {}
        self._events: dict[str, PendingEvent] = {}
        self._lock = asyncio.Lock()

    async def save_state(self, state: AgentState) -> None:
        async with self._lock:
            self._states[state.id] = state

    async def get_state(self, state_id: str) -> AgentState | None:
        return self._states.get(state_id)

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
        async with self._lock:
            state = self._states.get(state_id)
            if state is None:
                return
            state.status = status
            state.updated_at = datetime.now(timezone.utc)
            if wake_condition is not ...:
                state.wake_condition = wake_condition
            if result_summary is not ...:
                state.result_summary = result_summary
            if explain is not ...:
                state.explain = explain
            if last_activity_at is not ...:
                state.last_activity_at = last_activity_at
            if recent_steps is not ...:
                state.recent_steps = recent_steps

    async def get_states_by_parent(self, parent_id: str) -> list[AgentState]:
        return [
            s for s in self._states.values() if s.parent_id == parent_id
        ]

    async def find_pending(self) -> list[AgentState]:
        return [
            s
            for s in self._states.values()
            if s.status == AgentStateStatus.PENDING
        ]

    async def find_wakeable(self, now: datetime) -> list[AgentState]:
        results: list[AgentState] = []
        for s in self._states.values():
            if s.status != AgentStateStatus.SLEEPING:
                continue
            if s.wake_condition is not None and s.wake_condition.is_satisfied(now):
                results.append(s)
        return results

    async def find_unpropagated_completed(self) -> list[AgentState]:
        return [
            s
            for s in self._states.values()
            if s.status in (AgentStateStatus.COMPLETED, AgentStateStatus.FAILED)
            and not s.signal_propagated
            and s.parent_id is not None
        ]

    async def mark_child_completed(self, parent_id: str, child_id: str) -> None:
        async with self._lock:
            state = self._states.get(parent_id)
            if state is None or state.wake_condition is None:
                return
            if child_id not in state.wake_condition.completed_ids:
                state.wake_condition.completed_ids.append(child_id)
            state.updated_at = datetime.now(timezone.utc)

    async def mark_propagated(self, state_id: str) -> None:
        async with self._lock:
            state = self._states.get(state_id)
            if state is None:
                return
            state.signal_propagated = True
            state.updated_at = datetime.now(timezone.utc)

    async def find_timed_out(self, now: datetime) -> list[AgentState]:
        results: list[AgentState] = []
        for s in self._states.values():
            if s.status != AgentStateStatus.SLEEPING:
                continue
            if s.wake_condition is not None and s.wake_condition.is_timed_out(now):
                results.append(s)
        return results

    async def increment_wake_count(self, state_id: str) -> None:
        async with self._lock:
            state = self._states.get(state_id)
            if state is None:
                return
            state.wake_count += 1
            state.updated_at = datetime.now(timezone.utc)

    async def list_all(
        self,
        *,
        status: AgentStateStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AgentState]:
        states = list(self._states.values())
        if status is not None:
            states = [s for s in states if s.status == status]
        states.sort(key=lambda s: s.updated_at, reverse=True)
        return states[offset : offset + limit]

    async def find_running(self) -> list[AgentState]:
        return [
            s for s in self._states.values() if s.status == AgentStateStatus.RUNNING
        ]

    async def save_event(self, event: PendingEvent) -> None:
        async with self._lock:
            self._events[event.id] = event

    async def get_pending_events(
        self, target_agent_id: str, session_id: str
    ) -> list[PendingEvent]:
        events = [
            e
            for e in self._events.values()
            if e.target_agent_id == target_agent_id and e.session_id == session_id
        ]
        events.sort(key=lambda e: e.created_at)
        return events

    async def delete_events(self, event_ids: list[str]) -> None:
        async with self._lock:
            for eid in event_ids:
                self._events.pop(eid, None)

    async def append_recent_step(self, state_id: str, step: dict) -> None:
        async with self._lock:
            state = self._states.get(state_id)
            if state is None:
                return
            existing = state.recent_steps or []
            state.recent_steps = (existing + [step])[-_RECENT_STEPS_MAX:]
            state.last_activity_at = datetime.now(timezone.utc)
            state.updated_at = datetime.now(timezone.utc)

    async def delete_events_by_agent(self, target_agent_id: str) -> None:
        async with self._lock:
            to_delete = [
                eid
                for eid, e in self._events.items()
                if e.target_agent_id == target_agent_id
            ]
            for eid in to_delete:
                del self._events[eid]

    async def find_agents_with_debounced_events(
        self,
        min_count: int,
        max_wait_seconds: float,
        now: datetime,
    ) -> list[tuple[str, str]]:
        cutoff = now - timedelta(seconds=max_wait_seconds)
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        groups: dict[tuple[str, str], list[PendingEvent]] = {}
        for event in self._events.values():
            key = (event.target_agent_id, event.session_id)
            groups.setdefault(key, []).append(event)

        result: list[tuple[str, str]] = []
        for (agent_id, session_id), events in groups.items():
            if len(events) >= min_count:
                result.append((agent_id, session_id))
                continue
            oldest = min(events, key=lambda e: e.created_at)
            oldest_ts = oldest.created_at
            if oldest_ts.tzinfo is None:
                oldest_ts = oldest_ts.replace(tzinfo=timezone.utc)
            if oldest_ts <= cutoff:
                result.append((agent_id, session_id))
        return result

    async def has_recent_health_warning(
        self,
        target_agent_id: str,
        source_agent_id: str,
        within_seconds: float,
        now: datetime,
    ) -> bool:
        cutoff = now - timedelta(seconds=within_seconds)
        for event in self._events.values():
            if (
                event.target_agent_id == target_agent_id
                and event.source_agent_id == source_agent_id
                and event.event_type == SchedulerEventType.HEALTH_WARNING
            ):
                ts = event.created_at
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                cutoff_ts = cutoff
                if cutoff_ts.tzinfo is None:
                    cutoff_ts = cutoff_ts.replace(tzinfo=timezone.utc)
                if ts >= cutoff_ts:
                    return True
        return False


# ═══════════════════════════════════════════════════════════════════════════
# SQLite Implementation
# ═══════════════════════════════════════════════════════════════════════════


class SQLiteAgentStateStorage(AgentStateStorage):
    """SQLite-backed agent state storage."""

    def __init__(self, db_path: str = "scheduler.db") -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None
        self._initialized = False

    async def _get_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            # Use shared connection pool
            self._conn = await get_shared_connection(self._db_path)
            await self._create_tables()
            self._initialized = True
            logger.info("sqlite_agent_state_storage_connected", db_path=self._db_path)
        elif not self._initialized:
            # Check if tables exist (database may have been deleted externally)
            try:
                await self._conn.execute("SELECT 1 FROM agent_states LIMIT 1")
            except Exception:
                # Table doesn't exist, recreate
                await self._create_tables()
            self._initialized = True
        return self._conn

    async def _create_tables(self) -> None:
        conn = self._conn
        assert conn is not None
        await conn.execute(
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
                explain TEXT,
                last_activity_at TEXT,
                recent_steps_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_states_status ON agent_states(status)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_states_parent ON agent_states(parent_id)"
        )
        # Add new columns if they don't exist (for existing databases)
        for col_def in [
            ("explain", "TEXT"),
            ("last_activity_at", "TEXT"),
            ("recent_steps_json", "TEXT"),
        ]:
            try:
                await conn.execute(
                    f"ALTER TABLE agent_states ADD COLUMN {col_def[0]} {col_def[1]}"
                )
            except Exception:
                pass  # Column already exists

        await conn.execute(
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
            """
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pending_events_target ON pending_events(target_agent_id, session_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pending_events_created ON pending_events(created_at)"
        )
        await conn.commit()

    def _row_to_state(self, row: dict) -> AgentState:
        """Convert a database row to an AgentState."""
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
            "wake_type", "wake_time_value", "wake_time_unit",
            "wake_wait_for", "wake_wait_mode", "wake_completed_ids",
            "wake_submitted_task", "wakeup_at", "wake_timeout_at",
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
                 is_persistent, depth, wake_count,
                 explain, last_activity_at, recent_steps_json,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        cursor = await conn.execute(
            "SELECT * FROM agent_states WHERE id = ?", (state_id,)
        )
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
            params.append(
                json.dumps(recent_steps) if recent_steps is not None else None
            )

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
        for r in rows:
            state = self._row_to_state(dict(r))
            if state.wake_condition is not None and state.wake_condition.is_satisfied(now):
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

    # -- PendingEvent methods ------------------------------------------------

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
                        WHEN json_array_length(COALESCE(recent_steps_json, '[]')) >= {_RECENT_STEPS_MAX}
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
            await release_shared_connection(self._db_path)
            self._conn = None
            self._initialized = False


def create_agent_state_storage(config: AgentStateStorageConfig) -> AgentStateStorage:
    """Factory: create AgentStateStorage from configuration."""
    storage_type = config.storage_type
    cfg = config.config

    if storage_type == "memory":
        return InMemoryAgentStateStorage()
    elif storage_type == "sqlite":
        db_path = cfg.get("db_path", "scheduler.db")
        return SQLiteAgentStateStorage(db_path=db_path)
    else:
        raise ValueError(f"Unknown agent_state_storage_type: {storage_type}")
