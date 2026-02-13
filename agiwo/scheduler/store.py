"""
Agent State Storage — ABC and implementations.

Provides persistence for AgentState records used by the Scheduler.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    AgentStateStorageConfig,
    TimeUnit,
    WaitMode,
    WakeCondition,
    WakeType,
)


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
    ) -> None:
        """Update status and optional fields. Sentinel `...` means 'do not change'."""
        ...

    @abstractmethod
    async def get_states_by_parent(self, parent_state_id: str) -> list[AgentState]:
        """Get all child states for a given parent state id."""
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
        """Find COMPLETED/FAILED states with signal_propagated=False and parent_state_id is not None."""
        ...

    @abstractmethod
    async def mark_child_completed(self, parent_state_id: str, child_id: str) -> None:
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

    async def get_states_by_parent(self, parent_state_id: str) -> list[AgentState]:
        return [
            s for s in self._states.values() if s.parent_state_id == parent_state_id
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
            and s.parent_state_id is not None
        ]

    async def mark_child_completed(self, parent_state_id: str, child_id: str) -> None:
        async with self._lock:
            state = self._states.get(parent_state_id)
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


# ═══════════════════════════════════════════════════════════════════════════
# SQLite Implementation
# ═══════════════════════════════════════════════════════════════════════════


class SQLiteAgentStateStorage(AgentStateStorage):
    """SQLite-backed agent state storage."""

    def __init__(self, db_path: str = "scheduler.db") -> None:
        self._db_path = db_path
        self._conn: "aiosqlite.Connection | None" = None

    async def _get_conn(self) -> "aiosqlite.Connection":
        if self._conn is None:
            import aiosqlite

            self._conn = await aiosqlite.connect(self._db_path)
            self._conn.row_factory = aiosqlite.Row
            await self._conn.execute("PRAGMA journal_mode=WAL")
            await self._create_table()
        return self._conn

    async def _create_table(self) -> None:
        conn = self._conn
        assert conn is not None
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_states (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                parent_agent_id TEXT NOT NULL,
                parent_state_id TEXT,
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
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_states_status ON agent_states(status)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_states_parent ON agent_states(parent_state_id)"
        )
        await conn.commit()

    def _row_to_state(self, row: dict) -> AgentState:
        """Convert a database row to an AgentState."""
        wake_condition = None
        if row["wake_type"]:
            wakeup_at = None
            if row.get("wakeup_at"):
                wakeup_at = datetime.fromisoformat(row["wakeup_at"])
            time_unit = None
            if row.get("wake_time_unit"):
                time_unit = TimeUnit(row["wake_time_unit"])
            timeout_at = None
            if row.get("wake_timeout_at"):
                timeout_at = datetime.fromisoformat(row["wake_timeout_at"])
            wait_for = json.loads(row.get("wake_wait_for") or "[]")
            wait_mode = WaitMode(row.get("wake_wait_mode") or "all")
            completed_ids = json.loads(row.get("wake_completed_ids") or "[]")
            wake_condition = WakeCondition(
                type=WakeType(row["wake_type"]),
                wait_for=wait_for,
                wait_mode=wait_mode,
                completed_ids=completed_ids,
                time_value=row.get("wake_time_value"),
                time_unit=time_unit,
                wakeup_at=wakeup_at,
                submitted_task=row.get("wake_submitted_task"),
                timeout_at=timeout_at,
            )

        config_overrides = {}
        if row["config_overrides"]:
            config_overrides = json.loads(row["config_overrides"])

        return AgentState(
            id=row["id"],
            session_id=row["session_id"],
            agent_id=row["agent_id"],
            parent_agent_id=row["parent_agent_id"],
            parent_state_id=row["parent_state_id"],
            status=AgentStateStatus(row["status"]),
            task=row["task"],
            config_overrides=config_overrides,
            wake_condition=wake_condition,
            result_summary=row.get("result_summary"),
            signal_propagated=bool(row.get("signal_propagated", 0)),
            is_persistent=bool(row.get("is_persistent", 0)),
            depth=row.get("depth", 0) or 0,
            wake_count=row.get("wake_count", 0) or 0,
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
            wc.submitted_task,
            wc.wakeup_at.isoformat() if wc.wakeup_at else None,
            wc.timeout_at.isoformat() if wc.timeout_at else None,
        ]

    async def save_state(self, state: AgentState) -> None:
        conn = await self._get_conn()
        wc_vals = self._wake_condition_values(state.wake_condition)
        await conn.execute(
            """
            INSERT OR REPLACE INTO agent_states
                (id, session_id, agent_id, parent_agent_id, parent_state_id,
                 status, task, config_overrides,
                 wake_type, wake_time_value, wake_time_unit,
                 wake_wait_for, wake_wait_mode, wake_completed_ids,
                 wake_submitted_task, wakeup_at, wake_timeout_at,
                 result_summary, signal_propagated,
                 is_persistent, depth, wake_count,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                state.id,
                state.session_id,
                state.agent_id,
                state.parent_agent_id,
                state.parent_state_id,
                state.status.value,
                state.task,
                json.dumps(state.config_overrides),
                *wc_vals,
                state.result_summary,
                1 if state.signal_propagated else 0,
                1 if state.is_persistent else 0,
                state.depth,
                state.wake_count,
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

        params.append(state_id)
        sql = f"UPDATE agent_states SET {', '.join(sets)} WHERE id = ?"
        await conn.execute(sql, params)
        await conn.commit()

    async def get_states_by_parent(self, parent_state_id: str) -> list[AgentState]:
        conn = await self._get_conn()
        cursor = await conn.execute(
            "SELECT * FROM agent_states WHERE parent_state_id = ?",
            (parent_state_id,),
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
              AND parent_state_id IS NOT NULL
            """,
            (AgentStateStatus.COMPLETED.value, AgentStateStatus.FAILED.value),
        )
        rows = await cursor.fetchall()
        return [self._row_to_state(dict(r)) for r in rows]

    async def mark_child_completed(self, parent_state_id: str, child_id: str) -> None:
        conn = await self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        cursor = await conn.execute(
            "SELECT wake_completed_ids FROM agent_states WHERE id = ?",
            (parent_state_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return
        completed_ids: list[str] = json.loads(row["wake_completed_ids"] or "[]")
        if child_id not in completed_ids:
            completed_ids.append(child_id)
        await conn.execute(
            "UPDATE agent_states SET wake_completed_ids = ?, updated_at = ? WHERE id = ?",
            (json.dumps(completed_ids), now, parent_state_id),
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

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None


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
