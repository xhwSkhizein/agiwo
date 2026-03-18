"""
Scheduler data models.

Defines the core data structures for agent scheduling:
AgentState, AgentStateStatus, WakeCondition, WakeType, WaitMode, TimeUnit,
TaskLimits, PendingEvent, SchedulerEventType.
"""

from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from agiwo.agent.input import UserInput


class AgentStateStatus(str, Enum):
    """Status of a scheduled agent."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    IDLE = "idle"
    QUEUED = "queued"
    COMPLETED = "completed"
    FAILED = "failed"


ACTIVE_AGENT_STATUSES = frozenset(
    {
        AgentStateStatus.PENDING,
        AgentStateStatus.RUNNING,
        AgentStateStatus.WAITING,
        AgentStateStatus.IDLE,
        AgentStateStatus.QUEUED,
    }
)
TERMINAL_AGENT_STATUSES = frozenset(
    {
        AgentStateStatus.COMPLETED,
        AgentStateStatus.FAILED,
    }
)


class WakeType(str, Enum):
    """Type of wake condition."""

    WAITSET = "waitset"
    TIMER = "timer"
    PERIODIC = "periodic"
    PENDING_EVENTS = "pending_events"


class WaitMode(str, Enum):
    """How to evaluate a waitset: wait for ALL or ANY."""

    ALL = "all"
    ANY = "any"


class TimeUnit(str, Enum):
    """Time unit for timer/periodic wake conditions."""

    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURS = "hours"


class SchedulerEventType(str, Enum):
    """Type of pending event in the scheduler event queue."""

    CHILD_SLEEP_RESULT = "child_sleep_result"
    CHILD_COMPLETED = "child_completed"
    CHILD_FAILED = "child_failed"
    USER_HINT = "user_hint"


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
class ChildAgentConfigOverrides:
    """Typed overrides used when deriving a scheduler child agent."""

    instruction: str | None = None
    system_prompt: str | None = None


@dataclass
class WakeCondition:
    """Condition under which a waiting agent should be woken."""

    type: WakeType
    wait_for: list[str] = field(default_factory=list)
    wait_mode: WaitMode = WaitMode.ALL
    completed_ids: list[str] = field(default_factory=list)
    time_value: float | None = None
    time_unit: TimeUnit | None = None
    wakeup_at: datetime | None = None
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
        return False

    def is_timed_out(self, now: datetime) -> bool:
        """Check if this wake condition has timed out."""
        return self.timeout_at is not None and now >= self.timeout_at

    def to_seconds(self) -> float | None:
        """Convert time_value + time_unit to seconds."""
        if self.time_value is None or self.time_unit is None:
            return None
        return to_seconds(self.time_value, self.time_unit)


@dataclass
class AgentState:
    """
    Persistent state of a scheduled agent.

    ID Design:
    - `id`: Primary key, equals Agent.id (1:1 relationship)
    - `parent_id`: Parent state's id (None for root agent)
    """

    id: str
    session_id: str
    status: AgentStateStatus
    task: UserInput
    parent_id: str | None = None
    pending_input: UserInput | None = None
    config_overrides: dict = field(default_factory=dict)
    wake_condition: WakeCondition | None = None
    result_summary: str | None = None
    signal_propagated: bool = False
    agent_config_id: str | None = None
    is_persistent: bool = False
    depth: int = 0
    wake_count: int = 0
    explain: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_root(self) -> bool:
        return self.parent_id is None

    @property
    def is_child(self) -> bool:
        return self.parent_id is not None

    def is_active(self) -> bool:
        return self.status in ACTIVE_AGENT_STATUSES

    def is_terminal(self) -> bool:
        return self.status in TERMINAL_AGENT_STATUSES

    def is_waiting(self) -> bool:
        return self.status == AgentStateStatus.WAITING

    def is_idle_root(self) -> bool:
        return (
            self.is_root and self.is_persistent and self.status == AgentStateStatus.IDLE
        )

    def is_queued_root(self) -> bool:
        return (
            self.is_root
            and self.is_persistent
            and self.status == AgentStateStatus.QUEUED
        )

    def can_accept_enqueue_input(self) -> bool:
        return (
            self.is_root
            and self.is_persistent
            and self.status
            in (
                AgentStateStatus.IDLE,
                AgentStateStatus.FAILED,
            )
        )

    def resolve_runtime_session_id(self) -> str:
        """Return the session_id that the agent runtime should use."""
        if self.is_child:
            return self.id
        return self.session_id


@dataclass
class PendingEvent:
    """An event in an agent's pending event queue."""

    id: str
    target_agent_id: str
    session_id: str
    event_type: SchedulerEventType
    payload: dict
    created_at: datetime
    source_agent_id: str | None = None


@dataclass
class AgentStateStorageConfig:
    """Configuration for AgentStateStorage."""

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
    """Configuration for the Scheduler."""

    state_storage: AgentStateStorageConfig = field(
        default_factory=AgentStateStorageConfig
    )
    check_interval: float = 1.0
    max_concurrent: int = 20
    graceful_shutdown_wait_seconds: float = 10.0
    task_limits: TaskLimits = field(default_factory=TaskLimits)
    event_debounce_min_count: int = 3
    event_debounce_max_wait_seconds: float = 10.0


def normalize_statuses(
    statuses: Collection[AgentStateStatus] | None,
) -> frozenset[AgentStateStatus] | None:
    if statuses is None:
        return None
    return frozenset(statuses)


__all__ = [
    "ACTIVE_AGENT_STATUSES",
    "AgentState",
    "AgentStateStatus",
    "AgentStateStorageConfig",
    "ChildAgentConfigOverrides",
    "PendingEvent",
    "SchedulerConfig",
    "SchedulerEventType",
    "TERMINAL_AGENT_STATUSES",
    "TaskLimits",
    "TimeUnit",
    "WaitMode",
    "WakeCondition",
    "WakeType",
    "normalize_statuses",
    "to_seconds",
]
