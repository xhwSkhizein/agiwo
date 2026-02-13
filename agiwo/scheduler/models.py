"""
Scheduler data models.

Defines the core data structures for agent scheduling:
AgentState, AgentStateStatus, WakeCondition, WakeType, WaitMode, TimeUnit, TaskLimits.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class AgentStateStatus(str, Enum):
    """Status of a scheduled agent."""

    PENDING = "pending"
    RUNNING = "running"
    SLEEPING = "sleeping"
    COMPLETED = "completed"
    FAILED = "failed"


class WakeType(str, Enum):
    """Type of wake condition."""

    WAITSET = "waitset"
    TIMER = "timer"
    PERIODIC = "periodic"
    TASK_SUBMITTED = "task_submitted"


class WaitMode(str, Enum):
    """How to evaluate a waitset: wait for ALL or ANY."""

    ALL = "all"
    ANY = "any"


class TimeUnit(str, Enum):
    """Time unit for timer/periodic wake conditions."""

    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURS = "hours"


def to_seconds(value: float, unit: TimeUnit) -> float:
    """Convert a time value with unit to seconds."""
    if unit == TimeUnit.SECONDS:
        return value
    if unit == TimeUnit.MINUTES:
        return value * 60
    if unit == TimeUnit.HOURS:
        return value * 3600
    raise ValueError(f"Unknown time unit: {unit}")


@dataclass
class WakeCondition:
    """Condition under which a sleeping agent should be woken."""

    type: WakeType
    # WAITSET fields
    wait_for: list[str] = field(default_factory=list)
    wait_mode: WaitMode = WaitMode.ALL
    completed_ids: list[str] = field(default_factory=list)
    # TIMER / PERIODIC fields
    time_value: float | None = None
    time_unit: TimeUnit | None = None
    wakeup_at: datetime | None = None
    # TASK_SUBMITTED fields
    submitted_task: str | None = None
    # Timeout (WAITSET / PERIODIC — prevents permanent sleep)
    timeout_at: datetime | None = None

    def is_satisfied(self, now: datetime) -> bool:
        """Check if this wake condition is currently satisfied."""
        if self.type == WakeType.WAITSET:
            if not self.wait_for:
                return False
            if self.wait_mode == WaitMode.ALL:
                return set(self.wait_for) <= set(self.completed_ids)
            return bool(set(self.wait_for) & set(self.completed_ids))
        if self.type in (WakeType.TIMER, WakeType.PERIODIC):
            return self.wakeup_at is not None and now >= self.wakeup_at
        if self.type == WakeType.TASK_SUBMITTED:
            return self.submitted_task is not None
        return False

    def is_timed_out(self, now: datetime) -> bool:
        """Check if this wake condition has timed out."""
        return self.timeout_at is not None and now >= self.timeout_at

    def to_seconds(self) -> float | None:
        """Convert time_value + time_unit to seconds."""
        if self.time_value is None or self.time_unit is None:
            return None
        return to_seconds(self.time_value, self.time_unit)

    def to_dict(self) -> dict:
        """Serialize to dict for storage."""
        result: dict = {"type": self.type.value}
        if self.wait_for:
            result["wait_for"] = self.wait_for
        result["wait_mode"] = self.wait_mode.value
        if self.completed_ids:
            result["completed_ids"] = self.completed_ids
        if self.time_value is not None:
            result["time_value"] = self.time_value
        if self.time_unit is not None:
            result["time_unit"] = self.time_unit.value
        if self.wakeup_at is not None:
            result["wakeup_at"] = self.wakeup_at.isoformat()
        if self.submitted_task is not None:
            result["submitted_task"] = self.submitted_task
        if self.timeout_at is not None:
            result["timeout_at"] = self.timeout_at.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "WakeCondition":
        """Deserialize from dict."""
        wakeup_at = None
        if data.get("wakeup_at"):
            wakeup_at = datetime.fromisoformat(data["wakeup_at"])
        time_unit = None
        if data.get("time_unit"):
            time_unit = TimeUnit(data["time_unit"])
        timeout_at = None
        if data.get("timeout_at"):
            timeout_at = datetime.fromisoformat(data["timeout_at"])
        wait_mode = WaitMode.ALL
        if data.get("wait_mode"):
            wait_mode = WaitMode(data["wait_mode"])
        return cls(
            type=WakeType(data["type"]),
            wait_for=data.get("wait_for", []),
            wait_mode=wait_mode,
            completed_ids=data.get("completed_ids", []),
            time_value=data.get("time_value"),
            time_unit=time_unit,
            wakeup_at=wakeup_at,
            submitted_task=data.get("submitted_task"),
            timeout_at=timeout_at,
        )


@dataclass
class AgentState:
    """
    Persistent state of a scheduled agent.

    The `id` field is the agent_id and serves as the primary key.
    """

    id: str
    session_id: str
    agent_id: str
    parent_agent_id: str
    status: AgentStateStatus
    task: str
    parent_state_id: str | None = None
    config_overrides: dict = field(default_factory=dict)
    wake_condition: WakeCondition | None = None
    result_summary: str | None = None
    signal_propagated: bool = False
    is_persistent: bool = False
    depth: int = 0
    wake_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AgentStateStorageConfig:
    """Configuration for AgentStateStorage.

    storage_type: "memory" | "sqlite"
        - memory: In-memory storage (default, no persistence)
        - sqlite: SQLite database storage
    config: storage-specific configuration
        - sqlite: {"db_path": str}
    """

    storage_type: str = "memory"
    config: dict = field(default_factory=dict)


@dataclass
class TaskLimits:
    """Runtime limits for scheduled tasks — enforced by TaskGuard."""

    max_depth: int = 5
    max_children_per_agent: int = 10
    default_wait_timeout: float = 600.0
    max_wake_count: int = 20


@dataclass
class SchedulerConfig:
    """Configuration for the Scheduler.

    All fields are pure configuration — no side effects in construction.
    """

    state_storage: AgentStateStorageConfig = field(default_factory=AgentStateStorageConfig)
    check_interval: float = 5.0
    max_concurrent: int = 10
    graceful_shutdown_wait_seconds: int = 30
    task_limits: TaskLimits = field(default_factory=TaskLimits)
