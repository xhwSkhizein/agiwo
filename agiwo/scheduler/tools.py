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
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    TimeUnit,
    WakeCondition,
    WakeType,
    to_seconds,
)
from agiwo.scheduler.store import AgentStateStorage
from agiwo.tool.base import BaseTool, ToolResult
from agiwo.utils.abort_signal import AbortSignal


class SpawnAgentTool(BaseTool):
    """Spawn a child agent to handle a sub-task."""

    def __init__(self, store: AgentStateStorage) -> None:
        self._store = store
        super().__init__()

    def get_name(self) -> str:
        return "spawn_agent"

    def get_description(self) -> str:
        return (
            "Spawn a child agent to handle a sub-task asynchronously. "
            "The child agent will be scheduled and executed by the scheduler. "
            "After spawning all children, call sleep_and_wait to wait for them."
        )

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task to delegate to the child agent",
                },
                "system_prompt": {
                    "type": "string",
                    "description": "Optional system prompt override for the child agent",
                },
                "child_id": {
                    "type": "string",
                    "description": "Optional explicit child agent ID. Auto-generated if not provided.",
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

        task = parameters.get("task", "")
        system_prompt = parameters.get("system_prompt")
        child_id = parameters.get("child_id") or f"{parent_agent_id}_{uuid4().hex[:8]}"

        config_overrides: dict[str, Any] = {}
        if system_prompt:
            config_overrides["system_prompt"] = system_prompt

        state = AgentState(
            id=child_id,
            session_id=context.session_id,
            agent_id=child_id,
            parent_agent_id=parent_agent_id,
            parent_state_id=parent_agent_id,
            status=AgentStateStatus.PENDING,
            task=task,
            config_overrides=config_overrides,
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

    def __init__(self, store: AgentStateStorage) -> None:
        self._store = store
        super().__init__()

    def get_name(self) -> str:
        return "sleep_and_wait"

    def get_description(self) -> str:
        return (
            "Put the current agent to sleep and wait for a condition. "
            "Use 'children_complete' to wait for all spawned child agents to finish. "
            "Use 'delay' to sleep for a fixed duration. "
            "Use 'interval' to periodically wake up and check."
        )

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "wake_type": {
                    "type": "string",
                    "enum": ["children_complete", "delay", "interval"],
                    "description": "Type of wake condition",
                },
                "delay_seconds": {
                    "type": "number",
                    "description": "Delay/interval value (used with delay or interval wake_type)",
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

        wake_type_str = parameters.get("wake_type", "children_complete")
        delay_seconds = parameters.get("delay_seconds")
        time_unit_str = parameters.get("time_unit", "seconds")

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

        if wake_type == WakeType.CHILDREN_COMPLETE:
            children = await self._store.get_states_by_parent(agent_id)
            wc.total_children = len(children)
            already_completed = sum(
                1 for c in children if c.status == AgentStateStatus.COMPLETED
            )
            wc.completed_children = already_completed

        elif wake_type in (WakeType.DELAY, WakeType.INTERVAL):
            if delay_seconds is None:
                return ToolResult.error(
                    tool_name=self.get_name(),
                    error="delay_seconds is required for delay/interval wake type",
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

        await self._store.update_status(
            agent_id,
            AgentStateStatus.SLEEPING,
            wake_condition=wc,
        )

        end_time = time.time()
        summary = f"Agent '{agent_id}' entering sleep. Wake condition: {wake_type_str}"
        if wake_type == WakeType.CHILDREN_COMPLETE:
            summary += f" (total_children={wc.total_children}, already_completed={wc.completed_children})"
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
            "agent_id": state.agent_id,
            "status": state.status.value,
            "task": state.task,
            "result_summary": state.result_summary,
        }

        content_parts = [
            f"Agent: {state.agent_id}",
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
