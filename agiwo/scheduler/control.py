"""Narrow control surface exposed to scheduler tools."""

from datetime import datetime
from typing import Protocol

from agiwo.scheduler.models import AgentState, WakeCondition, WakeType


class SchedulerControl(Protocol):
    """Tool-facing scheduler control interface."""

    async def spawn_child(
        self,
        *,
        parent_agent_id: str,
        session_id: str,
        task: str,
        instruction: str | None,
        system_prompt: str | None,
        custom_child_id: str | None,
    ) -> AgentState: ...

    async def sleep_current_agent(
        self,
        *,
        agent_id: str,
        session_id: str,
        wake_type: WakeType,
        wake_type_str: str,
        wait_mode_str: str,
        explicit_wait_for: list[str] | None,
        timeout: float | None,
        delay_seconds: float | int | None,
        time_unit_str: str,
        explain: str | None,
    ) -> tuple[WakeCondition, str]: ...

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

    async def cancel_child(
        self,
        *,
        caller_id: str | None,
        target_id: str,
        force: bool,
        reason: str,
    ) -> tuple[str, AgentState | None, list[dict[str, object]]]: ...

    def age_seconds(
        self, timestamp: datetime, *, now: datetime | None = None
    ) -> int: ...


__all__ = ["SchedulerControl"]
