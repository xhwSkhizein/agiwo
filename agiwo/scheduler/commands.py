"""Scheduler domain commands and dispatch actions."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from agiwo.agent.input import UserInput
from agiwo.agent.runtime import AgentStreamItem
from agiwo.scheduler.models import (
    AgentState,
    PendingEvent,
    TimeUnit,
    WaitMode,
    WakeCondition,
    WakeType,
)

CancelChildOutcome = Literal[
    "missing",
    "already_terminal",
    "requires_force",
    "cancelled",
]


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
    running_processes: list[dict[str, object]] = field(default_factory=list)


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
    "RouteResult",
    "SleepRequest",
    "SleepResult",
    "SpawnChildRequest",
]
