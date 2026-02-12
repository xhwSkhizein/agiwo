"""
Scheduler data models.

Defines the core data structures for agent scheduling:
AgentState, AgentStateStatus, WakeCondition, WakeType, TimeUnit.
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

    CHILDREN_COMPLETE = "children_complete"
    DELAY = "delay"
    INTERVAL = "interval"


class TimeUnit(str, Enum):
    """Time unit for delay/interval wake conditions."""

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
    # DELAY / INTERVAL fields
    time_value: float | None = None
    time_unit: TimeUnit | None = None
    # CHILDREN_COMPLETE fields
    total_children: int = 0
    completed_children: int = 0
    # Computed absolute wakeup time (for DELAY / INTERVAL)
    wakeup_at: datetime | None = None

    def is_satisfied(self, now: datetime) -> bool:
        """Check if this wake condition is currently satisfied."""
        if self.type == WakeType.CHILDREN_COMPLETE:
            return (
                self.total_children > 0
                and self.completed_children >= self.total_children
            )
        if self.type in (WakeType.DELAY, WakeType.INTERVAL):
            return self.wakeup_at is not None and now >= self.wakeup_at
        return False

    def to_seconds(self) -> float | None:
        """Convert time_value + time_unit to seconds."""
        if self.time_value is None or self.time_unit is None:
            return None
        return to_seconds(self.time_value, self.time_unit)

    def to_dict(self) -> dict:
        """Serialize to dict for storage."""
        result: dict = {"type": self.type.value}
        if self.time_value is not None:
            result["time_value"] = self.time_value
        if self.time_unit is not None:
            result["time_unit"] = self.time_unit.value
        result["total_children"] = self.total_children
        result["completed_children"] = self.completed_children
        if self.wakeup_at is not None:
            result["wakeup_at"] = self.wakeup_at.isoformat()
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
        return cls(
            type=WakeType(data["type"]),
            time_value=data.get("time_value"),
            time_unit=time_unit,
            total_children=data.get("total_children", 0),
            completed_children=data.get("completed_children", 0),
            wakeup_at=wakeup_at,
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
class SchedulerConfig:
    """Configuration for the Scheduler.

    All fields are pure configuration — no side effects in construction.
    """

    state_storage: AgentStateStorageConfig = field(default_factory=AgentStateStorageConfig)
    check_interval: float = 5.0
    max_concurrent: int = 10
