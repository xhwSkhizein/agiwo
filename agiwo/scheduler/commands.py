"""Scheduler domain commands and dispatch actions."""

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Literal

from agiwo.agent import AgentStreamItem, UserInput
from agiwo.scheduler.models import (
    AgentState,
    PendingEvent,
    TimeUnit,
    WaitMode,
    WakeCondition,
    WakeType,
    _freeze_value,
)

CancelChildOutcome = Literal[
    "missing",
    "already_terminal",
    "requires_force",
    "cancelled",
]


class RouteStreamMode(str, Enum):
    RUN_END = "run_end"
    UNTIL_SETTLED = "until_settled"


class DispatchReason(str, Enum):
    ROOT_SUBMIT = "root_submit"
    ROOT_QUEUED_INPUT = "root_queued_input"
    CHILD_PENDING = "child_pending"
    WAKE_READY = "wake_ready"
    WAKE_EVENTS = "wake_events"
    WAKE_TIMEOUT = "wake_timeout"


@dataclass(frozen=True, slots=True)
class DispatchAction:
    state: AgentState
    reason: DispatchReason
    input_override: UserInput | None = None
    events: tuple[PendingEvent, ...] = ()


@dataclass(frozen=True, slots=True)
class SpawnChildRequest:
    parent_agent_id: str
    session_id: str
    task: str
    instruction: str | None = None
    system_prompt: str | None = None
    custom_child_id: str | None = None
    allowed_skills: list[str] | None = None
    allowed_tools: list[str] | None = None
    fork: bool = False


@dataclass(frozen=True, slots=True)
class SleepRequest:
    agent_id: str
    session_id: str
    wake_type: WakeType
    wait_mode: WaitMode = WaitMode.ALL
    wait_for: list[str] | None = None
    timeout: float | None = None
    delay_seconds: float | int | None = None
    time_unit: TimeUnit = TimeUnit.SECONDS
    explain: str | None = None
    no_progress: bool = False


@dataclass(frozen=True, slots=True)
class SleepResult:
    wake_condition: WakeCondition
    summary: str


@dataclass(frozen=True, slots=True)
class CancelChildRequest:
    target_id: str
    reason: str
    caller_id: str | None = None
    force: bool = False


@dataclass(frozen=True, slots=True)
class CancelChildResult:
    outcome: CancelChildOutcome
    state: AgentState | None = None
    running_processes: tuple[Mapping[str, object], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "running_processes",
            tuple(
                proc if isinstance(proc, MappingProxyType) else _freeze_value(proc)
                for proc in self.running_processes
            ),
        )


@dataclass(frozen=True, slots=True)
class RouteResult:
    action: Literal["submitted", "enqueued", "steered"]
    state_id: str
    stream: AsyncIterator[AgentStreamItem] | None = None


__all__ = [
    "CancelChildRequest",
    "CancelChildResult",
    "DispatchAction",
    "DispatchReason",
    "RouteStreamMode",
    "RouteResult",
    "SleepRequest",
    "SleepResult",
    "SpawnChildRequest",
]
