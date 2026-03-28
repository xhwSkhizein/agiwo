"""Internal helper for scheduler tool-facing control operations."""

from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import Protocol
from uuid import uuid4

from agiwo.scheduler.commands import (
    CancelChildRequest,
    CancelChildResult,
    SleepRequest,
    SleepResult,
    SpawnChildRequest,
)
from agiwo.scheduler.guard import TaskGuard
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    ChildAgentConfigOverrides,
    WakeCondition,
    WakeType,
    to_seconds,
)
from agiwo.scheduler.runtime_state import RuntimeState
from agiwo.scheduler.store.base import AgentStateStorage
from agiwo.scheduler.store.codec import serialize_child_agent_config_overrides
from agiwo.tool.process import AgentProcessRegistry


class SchedulerToolPort(Protocol):
    """Minimal interface that scheduling tools need from the scheduler."""

    async def spawn_child(self, request: SpawnChildRequest) -> AgentState: ...
    async def sleep_current_agent(self, request: SleepRequest) -> SleepResult: ...
    async def get_child_state(self, target_id: str) -> AgentState | None: ...
    async def list_child_states(
        self, *, caller_id: str | None, session_id: str
    ) -> list[AgentState]: ...
    async def inspect_child_processes(
        self, target_id: str
    ) -> list[dict[str, object]]: ...
    async def cancel_child(self, request: CancelChildRequest) -> CancelChildResult: ...
    def age_seconds(
        self, timestamp: datetime, *, now: datetime | None = None
    ) -> int: ...


class SchedulerToolControl:
    def __init__(
        self,
        *,
        store: AgentStateStorage,
        guard: TaskGuard,
        rt: RuntimeState,
        save_state: Callable[[AgentState], Awaitable[None]],
        cancel_subtree: Callable[[str, str], Awaitable[None]],
    ) -> None:
        self._store = store
        self._guard = guard
        self._rt = rt
        self._save_state = save_state
        self._cancel_subtree = cancel_subtree

    async def spawn_child(self, request: SpawnChildRequest) -> AgentState:
        parent_state = await self._store.get_state(request.parent_agent_id)
        if parent_state is None:
            raise ValueError(
                f"Parent agent state '{request.parent_agent_id}' not found"
            )

        rejection = await self._guard.check_spawn(parent_state)
        if rejection is not None:
            raise ValueError(f"Spawn rejected: {rejection}")

        child_id = (
            request.custom_child_id or f"{request.parent_agent_id}_{uuid4().hex[:5]}"
        )
        if await self._store.get_state(child_id) is not None:
            raise ValueError(f"Child state '{child_id}' already exists")

        state = AgentState(
            id=child_id,
            session_id=request.session_id,
            status=AgentStateStatus.PENDING,
            task=request.task,
            parent_id=request.parent_agent_id,
            config_overrides=serialize_child_agent_config_overrides(
                ChildAgentConfigOverrides(
                    instruction=request.instruction,
                    system_prompt=request.system_prompt,
                )
            ),
            depth=parent_state.depth + 1,
        )
        await self._save_state(state)
        return state

    async def sleep_current_agent(self, request: SleepRequest) -> SleepResult:
        wake_condition = await self._build_sleep_condition(request)
        state = await self._store.get_state(request.agent_id)
        if state is None:
            raise ValueError(f"Agent state '{request.agent_id}' not found")

        await self._save_state(
            state.with_waiting(
                wake_condition=wake_condition,
                explain=request.explain,
            )
        )
        return SleepResult(
            wake_condition=wake_condition,
            summary=self._build_sleep_summary(request, wake_condition),
        )

    async def get_child_state(self, target_id: str) -> AgentState | None:
        return await self._store.get_state(target_id)

    async def list_child_states(
        self,
        *,
        caller_id: str | None,
        session_id: str,
    ) -> list[AgentState]:
        return await self._store.list_states(
            parent_id=caller_id,
            session_id=session_id,
            limit=1000,
        )

    async def inspect_child_processes(
        self,
        target_id: str,
    ) -> list[dict[str, object]]:
        agent = self._rt.agents.get(target_id)
        if agent is None:
            return []

        for tool in agent.tools:
            if not isinstance(tool, AgentProcessRegistry):
                continue
            try:
                return await tool.list_agent_processes(target_id, state="running")
            except Exception:  # noqa: BLE001
                return []
        return []

    async def cancel_child(self, request: CancelChildRequest) -> CancelChildResult:
        target_state = await self._store.get_state(request.target_id)
        if target_state is None:
            return CancelChildResult(outcome="missing")

        if target_state.parent_id != request.caller_id:
            raise PermissionError(
                f"agent '{request.target_id}' is not a direct child of '{request.caller_id}'"
            )

        if not target_state.is_active():
            return CancelChildResult(
                outcome="already_terminal",
                state=target_state,
            )

        if not request.force and target_state.status == AgentStateStatus.RUNNING:
            return CancelChildResult(
                outcome="requires_force",
                state=target_state,
                running_processes=await self.inspect_child_processes(request.target_id),
            )

        await self._cancel_subtree(request.target_id, request.reason)
        return CancelChildResult(
            outcome="cancelled",
            state=target_state,
        )

    def age_seconds(self, timestamp: datetime, *, now: datetime | None = None) -> int:
        current = now or datetime.now(timezone.utc)
        normalized = (
            timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
        )
        return int((current - normalized).total_seconds())

    async def _build_sleep_condition(self, request: SleepRequest) -> WakeCondition:
        now = datetime.now(timezone.utc)
        if request.wake_type == WakeType.WAITSET:
            wait_for = await self._resolve_waitset_targets(request)
            return WakeCondition(
                type=request.wake_type,
                wait_for=wait_for,
                wait_mode=request.wait_mode,
                completed_ids=await self._collect_completed_child_ids(wait_for),
                timeout_at=now
                + timedelta(
                    seconds=request.timeout or self._guard.limits.default_wait_timeout
                ),
            )

        if request.delay_seconds is None:
            raise ValueError("delay_seconds is required for timer/periodic wake type")

        wake_condition = WakeCondition(
            type=request.wake_type,
            time_value=request.delay_seconds,
            time_unit=request.time_unit,
            wakeup_at=now
            + timedelta(seconds=to_seconds(request.delay_seconds, request.time_unit)),
        )
        if request.wake_type == WakeType.PERIODIC and request.timeout is not None:
            return wake_condition.with_timeout_at(
                now + timedelta(seconds=request.timeout)
            )
        return wake_condition

    def _build_sleep_summary(
        self,
        request: SleepRequest,
        wake_condition: WakeCondition,
    ) -> str:
        summary = (
            f"Agent '{request.agent_id}' entering sleep. "
            f"Wake condition: {request.wake_type.value}"
        )
        if request.wake_type == WakeType.WAITSET:
            summary += (
                " "
                f"(waiting_for={len(wake_condition.wait_for)}, "
                f"mode={wake_condition.wait_mode.value}, "
                f"already_done={len(wake_condition.completed_ids)})"
            )
        elif request.delay_seconds is not None:
            summary += f" (delay={request.delay_seconds} {request.time_unit.value})"
        if request.explain:
            summary += f" | reason: {request.explain}"
        return summary

    async def _resolve_waitset_targets(self, request: SleepRequest) -> list[str]:
        if request.wait_for is not None:
            return request.wait_for
        children = await self._store.list_states(
            parent_id=request.agent_id,
            session_id=request.session_id,
            limit=1000,
        )
        return [child.id for child in children]

    async def _collect_completed_child_ids(self, child_ids: list[str]) -> list[str]:
        completed_ids: list[str] = []
        for child_id in child_ids:
            child_state = await self._store.get_state(child_id)
            if child_state is not None and child_state.is_terminal():
                completed_ids.append(child_id)
        return completed_ids


__all__ = ["SchedulerToolControl", "SchedulerToolPort"]
