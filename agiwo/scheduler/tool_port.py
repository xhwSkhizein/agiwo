"""Public tool-facing scheduler port built on top of runtime/store/guard."""

from datetime import datetime, timezone
from uuid import uuid4

from agiwo.scheduler.guard import TaskGuard
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    ChildAgentConfigOverrides,
    WakeCondition,
    WakeType,
)
from agiwo.scheduler.runtime import SchedulerRuntime
from agiwo.scheduler.store import AgentStateStorage
from agiwo.scheduler.tool_support import (
    build_sleep_condition,
    build_sleep_summary,
    list_running_agent_processes,
)


class SchedulerToolPort:
    """Expose the narrow scheduler capability surface needed by scheduler tools."""

    def __init__(
        self,
        store: AgentStateStorage,
        guard: TaskGuard,
        runtime: SchedulerRuntime,
    ) -> None:
        self._store = store
        self._guard = guard
        self._runtime = runtime

    async def spawn_child(
        self,
        *,
        parent_agent_id: str,
        session_id: str,
        task: str,
        instruction: str | None,
        system_prompt: str | None,
        custom_child_id: str | None,
    ) -> AgentState:
        parent_state = await self._store.get_state(parent_agent_id)
        if parent_state is None:
            raise ValueError(f"Parent agent state '{parent_agent_id}' not found")

        rejection = await self._guard.check_spawn(parent_state)
        if rejection is not None:
            raise ValueError(f"Spawn rejected: {rejection}")

        child_id = custom_child_id or f"{parent_agent_id}_{uuid4().hex[:5]}"
        config_overrides = ChildAgentConfigOverrides(
            instruction=instruction,
            system_prompt=system_prompt,
        ).to_dict()
        state = AgentState(
            id=child_id,
            session_id=session_id,
            status=AgentStateStatus.PENDING,
            task=task,
            parent_id=parent_agent_id,
            config_overrides=config_overrides,
            depth=parent_state.depth + 1,
        )
        await self._store.save_state(state)
        return state

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
    ) -> tuple[WakeCondition, str]:
        wake_condition = await build_sleep_condition(
            self._store,
            self._guard,
            agent_id=agent_id,
            session_id=session_id,
            wake_type=wake_type,
            wait_mode_str=wait_mode_str,
            explicit_wait_for=explicit_wait_for,
            timeout=timeout,
            delay_seconds=delay_seconds,
            time_unit_str=time_unit_str,
        )
        await self._store.update_status(
            agent_id,
            AgentStateStatus.SLEEPING,
            wake_condition=wake_condition,
            explain=explain,
        )
        summary = build_sleep_summary(
            agent_id=agent_id,
            wake_type=wake_type,
            wake_type_str=wake_type_str,
            wake_condition=wake_condition,
            delay_seconds=delay_seconds,
            time_unit_str=time_unit_str,
            explain=explain,
        )
        return wake_condition, summary

    async def get_child_state(self, target_id: str) -> AgentState | None:
        return await self._store.get_state(target_id)

    async def list_child_states(
        self,
        *,
        caller_id: str | None,
        session_id: str,
    ) -> list[AgentState]:
        children = await self._store.get_states_by_parent(caller_id)
        return [child for child in children if child.session_id == session_id]

    async def inspect_child_processes(
        self,
        target_id: str,
    ) -> list[dict[str, object]]:
        return await list_running_agent_processes(
            self._runtime.get_registered_agent(target_id),
            target_id,
        )

    async def cancel_child(
        self,
        *,
        caller_id: str | None,
        target_id: str,
        force: bool,
        reason: str,
    ) -> tuple[str, AgentState | None, list[dict[str, object]]]:
        target_state = await self._store.get_state(target_id)
        if target_state is None:
            return "missing", None, []

        if target_state.parent_id != caller_id:
            raise PermissionError(
                f"agent '{target_id}' is not a direct child of '{caller_id}'"
            )

        if target_state.status not in (
            AgentStateStatus.RUNNING,
            AgentStateStatus.SLEEPING,
            AgentStateStatus.PENDING,
        ):
            return "already_terminal", target_state, []

        if not force and target_state.status == AgentStateStatus.RUNNING:
            return (
                "requires_force",
                target_state,
                await self.inspect_child_processes(target_id),
            )

        await self._runtime.recursive_cancel(target_id, reason)
        return "cancelled", target_state, []

    def age_seconds(self, timestamp: datetime, *, now: datetime | None = None) -> int:
        current = now or datetime.now(timezone.utc)
        normalized = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
        return int((current - normalized).total_seconds())


__all__ = ["SchedulerToolPort"]
