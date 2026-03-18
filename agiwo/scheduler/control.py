"""Narrow control surface exposed to scheduler tools."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol

from agiwo.scheduler.models import (
    AgentState,
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


@dataclass(frozen=True)
class SpawnChildRequest:
    parent_agent_id: str
    session_id: str
    task: str
    instruction: str | None = None
    system_prompt: str | None = None
    custom_child_id: str | None = None


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class SleepResult:
    wake_condition: WakeCondition
    summary: str


@dataclass(frozen=True)
class CancelChildRequest:
    target_id: str
    reason: str
    caller_id: str | None = None
    force: bool = False


@dataclass(frozen=True)
class CancelChildResult:
    outcome: CancelChildOutcome
    state: AgentState | None = None
    running_processes: list[dict[str, object]] = field(default_factory=list)


class SchedulerControl(Protocol):
    """Tool-facing scheduler control interface."""

    async def spawn_child(self, request: SpawnChildRequest) -> AgentState: ...

    async def sleep_current_agent(self, request: SleepRequest) -> SleepResult: ...

    async def get_child_state(self, target_id: str) -> AgentState | None: ...

    async def list_child_states(
        self,
        *,
        caller_id: str | None,
        session_id: str,
    ) -> list[AgentState]: ...

    async def inspect_child_processes(
        self,
        target_id: str,
    ) -> list[dict[str, object]]: ...

    async def cancel_child(self, request: CancelChildRequest) -> CancelChildResult: ...

    def age_seconds(
        self, timestamp: datetime, *, now: datetime | None = None
    ) -> int: ...


__all__ = [
    "CancelChildRequest",
    "CancelChildResult",
    "SchedulerControl",
    "SleepRequest",
    "SleepResult",
    "SpawnChildRequest",
]
