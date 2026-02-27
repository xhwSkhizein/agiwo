"""
Scheduling tools — SpawnAgentTool, SleepAndWaitTool, QuerySpawnedAgentTool.

These tools are injected into agents by the Scheduler. They read `context.agent_id`
directly (no SchedulingMeta needed) and interact with AgentStateStorage.
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from agiwo.agent.execution_context import ExecutionContext
from agiwo.agent.schema import TerminationReason
from agiwo.scheduler.guard import TaskGuard
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    TimeUnit,
    WaitMode,
    WakeCondition,
    WakeType,
    to_seconds,
)
from agiwo.scheduler.store import AgentStateStorage
from agiwo.tool.base import BaseTool, ToolResult
from agiwo.utils.abort_signal import AbortSignal


class SpawnAgentTool(BaseTool):
    """Spawn a child agent to handle a sub-task."""

    def __init__(self, store: AgentStateStorage, guard: TaskGuard) -> None:
        self._store = store
        self._guard = guard
        super().__init__()

    def get_name(self) -> str:
        return "spawn_agent"

    def get_description(self) -> str:
        return (
            "Spawn a child agent to handle a truly independent sub-task asynchronously. "
            "ONLY use this when the task genuinely requires parallel execution or delegation "
            "(e.g., concurrent data fetching, parallel analysis). "
            "Do NOT spawn a child agent just to perform a simple action you can do directly. "
            "The spawned child agent will NOT be able to spawn further child agents. "
            "After spawning, call sleep_and_wait to wait for the child to complete if needed."
        )

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "The task for the child agent to complete directly. Keep it brief but thorough, covering necessary context (e.g., background, dependencies, goal, expected outcome) depending on task needs."
                        "Describe what outcome you need (not what process to follow). "
                        "Do NOT instruct the child to spawn more agents — it will complete the task itself."
                    ),
                },
                "instruction": {
                    "type": "string",
                    "description": "Optional instruction: guide the child agent to finish the task in a more efficient, elegant, and effective way.",
                },
            },
            "required": ["task"],
        }

    def is_concurrency_safe(self) -> bool:
        return True

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        start_time = time.time()
        parent_agent_id = context.agent_id
        if not parent_agent_id:
            return ToolResult.error(
                tool_name=self.get_name(),
                error="Cannot spawn agent: no agent_id in execution context",
                tool_call_id=parameters.get("tool_call_id", ""),
                input_args=parameters,
                start_time=start_time,
            )

        parent_state = await self._store.get_state(parent_agent_id)
        if parent_state is None:
            return ToolResult.error(
                tool_name=self.get_name(),
                error=f"Parent agent state '{parent_agent_id}' not found",
                tool_call_id=parameters.get("tool_call_id", ""),
                input_args=parameters,
                start_time=start_time,
            )

        rejection = await self._guard.check_spawn(parent_state)
        if rejection is not None:
            return ToolResult.error(
                tool_name=self.get_name(),
                error=f"Spawn rejected: {rejection}",
                tool_call_id=parameters.get("tool_call_id", ""),
                input_args=parameters,
                start_time=start_time,
            )

        task = parameters.get("task", "")
        instruction = parameters.get("instruction")
        custom_child_id = parameters.get("child_id")
        child_id = custom_child_id or f"{parent_agent_id}_{uuid4().hex[:5]}"

        config_overrides: dict[str, Any] = {}
        if instruction:
            config_overrides["instruction"] = instruction
        system_prompt = parameters.get("system_prompt")
        if system_prompt:
            config_overrides["system_prompt"] = system_prompt

        state = AgentState(
            id=child_id,
            session_id=context.session_id,
            status=AgentStateStatus.PENDING,
            task=task,
            parent_id=parent_agent_id,
            config_overrides=config_overrides,
            depth=parent_state.depth + 1,
        )
        await self._store.save_state(state)

        end_time = time.time()
        return ToolResult(
            tool_name=self.get_name(),
            tool_call_id=parameters.get("tool_call_id", ""),
            input_args={"task": task, "child_id": child_id},
            content=f"Spawned child agent '{child_id}' for task: {task}",
            output={"child_id": child_id, "status": "pending"},
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
        )


class SleepAndWaitTool(BaseTool):
    """Put the current agent to sleep until a wake condition is met."""

    def __init__(self, store: AgentStateStorage, guard: TaskGuard) -> None:
        self._store = store
        self._guard = guard
        super().__init__()

    def get_name(self) -> str:
        return "sleep_and_wait"

    def get_description(self) -> str:
        return (
            "Put the current agent to sleep and wait for a condition. "
            "Use 'waitset' to wait for spawned child agents to finish. "
            "Use 'timer' to sleep for a fixed duration. "
            "Use 'periodic' to periodically wake up and check."
        )

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "wake_type": {
                    "type": "string",
                    "enum": ["waitset", "timer", "periodic"],
                    "description": "Type of wake condition",
                },
                "wait_mode": {
                    "type": "string",
                    "enum": ["all", "any"],
                    "description": "For waitset: wait for ALL or ANY children (default: all)",
                },
                "wait_for": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of specific child agent IDs to wait for. If omitted, waits for all children.",
                },
                "timeout": {
                    "type": "number",
                    "description": "Optional timeout in seconds for the wait",
                },
                "delay_seconds": {
                    "type": "number",
                    "description": "Delay/interval value in time_unit (used with timer or periodic)",
                },
                "time_unit": {
                    "type": "string",
                    "enum": ["seconds", "minutes", "hours"],
                    "description": "Time unit for delay_seconds (default: seconds)",
                },
            },
            "required": ["wake_type"],
        }

    def is_concurrency_safe(self) -> bool:
        return False

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        start_time = time.time()
        agent_id = context.agent_id
        if not agent_id:
            return ToolResult.error(
                tool_name=self.get_name(),
                error="Cannot sleep: no agent_id in execution context",
                tool_call_id=parameters.get("tool_call_id", ""),
                input_args=parameters,
                start_time=start_time,
            )

        wake_type_str = parameters.get("wake_type", "waitset")
        delay_seconds = parameters.get("delay_seconds")
        time_unit_str = parameters.get("time_unit", "seconds")
        wait_mode_str = parameters.get("wait_mode", "all")
        explicit_wait_for: list[str] | None = parameters.get("wait_for")
        timeout = parameters.get("timeout")

        try:
            wake_type = WakeType(wake_type_str)
        except ValueError:
            return ToolResult.error(
                tool_name=self.get_name(),
                error=f"Invalid wake_type: {wake_type_str}",
                tool_call_id=parameters.get("tool_call_id", ""),
                input_args=parameters,
                start_time=start_time,
            )

        now = datetime.now(timezone.utc)
        wc = WakeCondition(type=wake_type)

        if wake_type == WakeType.WAITSET:
            if explicit_wait_for is not None:
                wc.wait_for = explicit_wait_for
            else:
                children = await self._store.get_states_by_parent(agent_id)
                wc.wait_for = [c.id for c in children if c.session_id == context.session_id]
            try:
                wc.wait_mode = WaitMode(wait_mode_str)
            except ValueError:
                wc.wait_mode = WaitMode.ALL
            already_done = []
            for cid in wc.wait_for:
                child_state = await self._store.get_state(cid)
                if child_state is not None and child_state.status in (
                    AgentStateStatus.COMPLETED,
                    AgentStateStatus.FAILED,
                ):
                    already_done.append(cid)
            wc.completed_ids = already_done
            effective_timeout = timeout or self._guard.limits.default_wait_timeout
            wc.timeout_at = now + timedelta(seconds=effective_timeout)

        elif wake_type in (WakeType.TIMER, WakeType.PERIODIC):
            if delay_seconds is None:
                return ToolResult.error(
                    tool_name=self.get_name(),
                    error="delay_seconds is required for timer/periodic wake type",
                    tool_call_id=parameters.get("tool_call_id", ""),
                    input_args=parameters,
                    start_time=start_time,
                )
            try:
                time_unit = TimeUnit(time_unit_str)
            except ValueError:
                time_unit = TimeUnit.SECONDS
            wc.time_value = delay_seconds
            wc.time_unit = time_unit
            seconds = to_seconds(delay_seconds, time_unit)
            wc.wakeup_at = now + timedelta(seconds=seconds)
            if wake_type == WakeType.PERIODIC and timeout is not None:
                wc.timeout_at = now + timedelta(seconds=timeout)

        await self._store.update_status(
            agent_id,
            AgentStateStatus.SLEEPING,
            wake_condition=wc,
        )

        end_time = time.time()
        summary = f"Agent '{agent_id}' entering sleep. Wake condition: {wake_type_str}"
        if wake_type == WakeType.WAITSET:
            summary += f" (waiting_for={len(wc.wait_for)}, mode={wc.wait_mode.value}, already_done={len(wc.completed_ids)})"
        elif delay_seconds is not None:
            summary += f" (delay={delay_seconds} {time_unit_str})"

        return ToolResult(
            tool_name=self.get_name(),
            tool_call_id=parameters.get("tool_call_id", ""),
            input_args=parameters,
            content=summary,
            output={"wake_type": wake_type_str, "agent_id": agent_id},
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            termination_reason=TerminationReason.SLEEPING,
        )


class QuerySpawnedAgentTool(BaseTool):
    """Query the status and result of a spawned child agent."""

    def __init__(self, store: AgentStateStorage) -> None:
        self._store = store
        super().__init__()

    def get_name(self) -> str:
        return "query_spawned_agent"

    def get_description(self) -> str:
        return (
            "Query the current status and result of a spawned child agent. "
            "Use this after waking up to check what the child agents have accomplished."
        )

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "The ID of the child agent to query",
                },
            },
            "required": ["agent_id"],
        }

    def is_concurrency_safe(self) -> bool:
        return True

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        start_time = time.time()
        target_id = parameters.get("agent_id", "")

        state = await self._store.get_state(target_id)
        if state is None:
            end_time = time.time()
            return ToolResult(
                tool_name=self.get_name(),
                tool_call_id=parameters.get("tool_call_id", ""),
                input_args=parameters,
                content=f"Agent '{target_id}' not found.",
                output=None,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                is_success=False,
                error=f"Agent '{target_id}' not found",
            )

        result_info = {
            "agent_id": state.id,
            "status": state.status.value,
            "task": state.task,
            "result_summary": state.result_summary,
        }

        content_parts = [
            f"Agent: {state.id}",
            f"Status: {state.status.value}",
            f"Task: {state.task}",
        ]
        if state.result_summary:
            content_parts.append(f"Result: {state.result_summary}")

        end_time = time.time()
        return ToolResult(
            tool_name=self.get_name(),
            tool_call_id=parameters.get("tool_call_id", ""),
            input_args=parameters,
            content="\n".join(content_parts),
            output=result_info,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
        )
