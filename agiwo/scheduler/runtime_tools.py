"""Scheduling runtime tools injected by the Scheduler."""

import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from agiwo.agent.execution_context import ExecutionContext
from agiwo.agent.runtime_tools import RuntimeToolOutcome
from agiwo.agent.runtime import TerminationReason
from agiwo.scheduler.control import SchedulerControl
from agiwo.scheduler.models import WakeType
from agiwo.scheduler.formatting import (
    build_child_result_detail_lines,
    summarize_text,
)
from agiwo.tool.base import ToolDefinition, ToolResult
from agiwo.utils.abort_signal import AbortSignal


class SchedulerRuntimeTool(ABC):
    cacheable = False
    timeout_seconds = 30

    @abstractmethod
    def get_name(self) -> str: ...

    @abstractmethod
    def get_description(self) -> str: ...

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]: ...

    @abstractmethod
    def is_concurrency_safe(self) -> bool: ...

    @abstractmethod
    async def execute_for_agent(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
        abort_signal: AbortSignal | None = None,
    ) -> RuntimeToolOutcome: ...

    def get_short_description(self) -> str:
        return self.get_description()

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.get_name(),
            description=self.get_description(),
            parameters=self.get_parameters(),
            is_concurrency_safe=self.is_concurrency_safe(),
            timeout_seconds=self.timeout_seconds,
            cacheable=self.cacheable,
        )


class SpawnAgentTool(SchedulerRuntimeTool):
    """Spawn a child agent to handle a sub-task."""

    def __init__(self, port: SchedulerControl) -> None:
        self._port = port

    def get_name(self) -> str:
        return "spawn_agent"

    def get_description(self) -> str:
        return (
            "Spawn a child agent to handle a truly independent sub-task asynchronously. "
            "ONLY use this when the task genuinely requires parallel execution or delegation "
            "(e.g., concurrent data fetching, parallel analysis). "
            "Do NOT spawn a child agent just to perform a simple action you can do directly. "
            "**IMPORTANT: The spawned child agent will NOT be able to spawn further child agents.**"
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

    async def execute_for_agent(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
        abort_signal: AbortSignal | None = None,
    ) -> RuntimeToolOutcome:
        start_time = time.time()
        parent_agent_id = context.agent_id
        if not parent_agent_id:
            return RuntimeToolOutcome(
                result=ToolResult.failed(
                    tool_name=self.get_name(),
                    error="Cannot spawn agent: no agent_id in execution context",
                    tool_call_id=parameters.get("tool_call_id", ""),
                    input_args=parameters,
                    start_time=start_time,
                )
            )

        task = parameters.get("task", "")
        instruction = parameters.get("instruction")
        custom_child_id = parameters.get("child_id")
        system_prompt = parameters.get("system_prompt")
        try:
            state = await self._port.spawn_child(
                parent_agent_id=parent_agent_id,
                session_id=context.session_id,
                task=task,
                instruction=instruction,
                system_prompt=system_prompt,
                custom_child_id=custom_child_id,
            )
        except ValueError as exc:
            return RuntimeToolOutcome(
                result=ToolResult.failed(
                    tool_name=self.get_name(),
                    error=str(exc),
                    tool_call_id=parameters.get("tool_call_id", ""),
                    input_args=parameters,
                    start_time=start_time,
                )
            )

        return RuntimeToolOutcome(
            result=ToolResult.success(
                tool_name=self.get_name(),
                tool_call_id=parameters.get("tool_call_id", ""),
                input_args={"task": task, "child_id": state.id},
                content=f"Spawned child agent '{state.id}' for task: {task}",
                output={"child_id": state.id, "status": "pending"},
                start_time=start_time,
            )
        )


class SleepAndWaitTool(SchedulerRuntimeTool):
    """Put the current agent to sleep until a wake condition is met."""

    def __init__(self, port: SchedulerControl) -> None:
        self._port = port

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
                "explain": {
                    "type": "string",
                    "description": "Optional brief reason for sleeping — visible to the parent agent and operators for observability.",
                },
            },
            "required": ["wake_type"],
        }

    def is_concurrency_safe(self) -> bool:
        return False

    async def execute_for_agent(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
        abort_signal: AbortSignal | None = None,
    ) -> RuntimeToolOutcome:
        start_time = time.time()
        agent_id = context.agent_id
        if not agent_id:
            return RuntimeToolOutcome(
                result=ToolResult.failed(
                    tool_name=self.get_name(),
                    error="Cannot sleep: no agent_id in execution context",
                    tool_call_id=parameters.get("tool_call_id", ""),
                    input_args=parameters,
                    start_time=start_time,
                )
            )

        wake_type_str = parameters.get("wake_type", "waitset")
        delay_seconds = parameters.get("delay_seconds")
        time_unit_str = parameters.get("time_unit", "seconds")
        wait_mode_str = parameters.get("wait_mode", "all")
        explicit_wait_for: list[str] | None = parameters.get("wait_for")
        timeout = parameters.get("timeout")
        explain = parameters.get("explain")

        try:
            wake_type = WakeType(wake_type_str)
        except ValueError:
            return RuntimeToolOutcome(
                result=ToolResult.failed(
                    tool_name=self.get_name(),
                    error=f"Invalid wake_type: {wake_type_str}",
                    tool_call_id=parameters.get("tool_call_id", ""),
                    input_args=parameters,
                    start_time=start_time,
                )
            )

        try:
            wake_condition, summary = await self._port.sleep_current_agent(
                agent_id=agent_id,
                session_id=context.session_id,
                wake_type=wake_type,
                wake_type_str=wake_type_str,
                wait_mode_str=wait_mode_str,
                explicit_wait_for=explicit_wait_for,
                timeout=timeout,
                delay_seconds=delay_seconds,
                time_unit_str=time_unit_str,
                explain=explain,
            )
        except ValueError as exc:
            return RuntimeToolOutcome(
                result=ToolResult.failed(
                    tool_name=self.get_name(),
                    error=str(exc),
                    tool_call_id=parameters.get("tool_call_id", ""),
                    input_args=parameters,
                    start_time=start_time,
                )
            )

        return RuntimeToolOutcome(
            result=ToolResult.success(
                tool_name=self.get_name(),
                tool_call_id=parameters.get("tool_call_id", ""),
                input_args=parameters,
                content=summary,
                output={"wake_type": wake_type_str, "agent_id": agent_id},
                start_time=start_time,
            ),
            termination_reason=TerminationReason.SLEEPING,
        )


class QuerySpawnedAgentTool(SchedulerRuntimeTool):
    """Query the status and result of a spawned child agent."""

    def __init__(self, port: SchedulerControl) -> None:
        self._port = port

    def get_name(self) -> str:
        return "query_spawned_agent"

    def get_description(self) -> str:
        return (
            "Query the current status, result, and recent activity of a spawned child agent. "
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

    async def execute_for_agent(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
        abort_signal: AbortSignal | None = None,
    ) -> RuntimeToolOutcome:
        start_time = time.time()
        target_id = parameters.get("agent_id", "")

        state = await self._port.get_child_state(target_id)
        if state is None:
            return RuntimeToolOutcome(
                result=ToolResult.failed(
                    tool_name=self.get_name(),
                    error=f"Agent '{target_id}' not found",
                    tool_call_id=parameters.get("tool_call_id", ""),
                    input_args=parameters,
                    content=f"Agent '{target_id}' not found.",
                    output=None,
                    start_time=start_time,
                )
            )

        result_info: dict[str, Any] = {
            "agent_id": state.id,
            "status": state.status.value,
            "task": state.task if isinstance(state.task, str) else str(state.task),
            "result_summary": state.result_summary,
            "explain": state.explain,
            "wake_count": state.wake_count,
            "last_activity_at": state.last_activity_at.isoformat()
            if state.last_activity_at
            else None,
        }
        if state.recent_steps:
            result_info["recent_steps"] = state.recent_steps

        content_parts = [
            f"Agent: {state.id}",
            f"Status: {state.status.value}",
            f"Task: {state.task if isinstance(state.task, str) else str(state.task)}",
        ]
        content_parts.extend(
            build_child_result_detail_lines(
                result=state.result_summary,
                explain=state.explain,
            )
        )
        if state.last_activity_at:
            content_parts.append(f"Last activity: {state.last_activity_at.isoformat()}")
        if state.recent_steps:
            steps_text = []
            for step in state.recent_steps[-3:]:
                role = step.get("role", "?")
                ts = step.get("timestamp", "")
                tools = step.get("tool_calls", [])
                if tools:
                    steps_text.append(
                        f"  [{ts[:19]}] {role}: called {', '.join(tools)}"
                    )
                else:
                    steps_text.append(f"  [{ts[:19]}] {role}")
            content_parts.append("Recent steps:\n" + "\n".join(steps_text))

        return RuntimeToolOutcome(
            result=ToolResult.success(
                tool_name=self.get_name(),
                tool_call_id=parameters.get("tool_call_id", ""),
                input_args=parameters,
                content="\n".join(content_parts),
                output=result_info,
                start_time=start_time,
            )
        )


class CancelAgentTool(SchedulerRuntimeTool):
    """Cancel a child agent, optionally forcing termination even with running processes."""

    def __init__(self, port: SchedulerControl) -> None:
        self._port = port

    def get_name(self) -> str:
        return "cancel_agent"

    def get_description(self) -> str:
        return (
            "Cancel a spawned child agent and its entire subtree. "
            "Use force=false first to check for running processes; "
            "if confirmed safe, use force=true to terminate."
        )

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "The ID of the child agent to cancel",
                },
                "force": {
                    "type": "boolean",
                    "description": "If false (default), check for running processes first. If true, cancel regardless.",
                    "default": False,
                },
                "reason": {
                    "type": "string",
                    "description": "Optional reason for cancellation (for logging/audit)",
                },
            },
            "required": ["agent_id"],
        }

    def is_concurrency_safe(self) -> bool:
        return True

    async def execute_for_agent(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
        abort_signal: AbortSignal | None = None,
    ) -> RuntimeToolOutcome:
        start_time = time.time()
        caller_id = context.agent_id
        target_id = parameters.get("agent_id", "")
        force = parameters.get("force", False)
        reason = parameters.get("reason", "Cancelled by parent agent")

        try:
            outcome, target_state, running_procs = await self._port.cancel_child(
                caller_id=caller_id,
                target_id=target_id,
                force=force,
                reason=reason,
            )
        except PermissionError as exc:
            return RuntimeToolOutcome(
                result=ToolResult.denied(
                    tool_name=self.get_name(),
                    reason=str(exc),
                    tool_call_id=str(parameters.get("tool_call_id", "")),
                    input_args=parameters,
                    start_time=start_time,
                    content=f"Permission denied: {exc}.",
                    output={"success": False},
                )
            )

        if outcome == "missing":
            return RuntimeToolOutcome(
                result=ToolResult.failed(
                    tool_name=self.get_name(),
                    error=f"Agent '{target_id}' not found",
                    tool_call_id=parameters.get("tool_call_id", ""),
                    input_args=parameters,
                    content=f"Agent '{target_id}' not found.",
                    output={"success": False},
                    start_time=start_time,
                )
            )

        if outcome == "already_terminal" and target_state is not None:
            return RuntimeToolOutcome(
                result=ToolResult.success(
                    tool_name=self.get_name(),
                    tool_call_id=parameters.get("tool_call_id", ""),
                    input_args=parameters,
                    content=f"Agent '{target_id}' is already in terminal state: {target_state.status.value}.",
                    output={"success": False, "status": target_state.status.value},
                    start_time=start_time,
                )
            )

        if outcome == "requires_force" and target_state is not None:
            content_parts = [
                f"Agent '{target_id}' is currently RUNNING. "
                f"Last activity: {target_state.last_activity_at.isoformat() if target_state.last_activity_at else 'unknown'}.",
            ]
            proc_info: list[dict] = []
            if running_procs:
                content_parts.append(
                    f"\n{len(running_procs)} background process(es) are running under this agent:"
                )
                for p in running_procs:
                    content_parts.append(f"  - [{p['process_id']}] {p['command']}")
                    proc_info.append(p)
                content_parts.append(
                    "\nUse force=true to terminate the agent and all its processes."
                )
            else:
                content_parts.append(
                    "\nNo background processes detected. Use force=true to cancel."
                )
            return RuntimeToolOutcome(
                result=ToolResult.success(
                    tool_name=self.get_name(),
                    tool_call_id=parameters.get("tool_call_id", ""),
                    input_args=parameters,
                    content="\n".join(content_parts),
                    output={
                        "success": False,
                        "requires_force": True,
                        "status": target_state.status.value,
                        "running_processes": proc_info,
                    },
                    start_time=start_time,
                )
            )

        return RuntimeToolOutcome(
            result=ToolResult.success(
                tool_name=self.get_name(),
                tool_call_id=parameters.get("tool_call_id", ""),
                input_args=parameters,
                content=f"Agent '{target_id}' and its subtree have been cancelled. Reason: {reason}",
                output={"success": True, "agent_id": target_id},
                start_time=start_time,
            )
        )


class ListAgentsTool(SchedulerRuntimeTool):
    """List all direct child agents of the calling agent with detailed status information."""

    def __init__(self, port: SchedulerControl) -> None:
        self._port = port

    def get_name(self) -> str:
        return "list_agents"

    def get_description(self) -> str:
        return (
            "List all direct child agents with their status, task, results, and recent activity. "
            "Use this to get an overview of all spawned agents and decide on next steps."
        )

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def is_concurrency_safe(self) -> bool:
        return True

    async def execute_for_agent(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
        abort_signal: AbortSignal | None = None,
    ) -> RuntimeToolOutcome:
        start_time = time.time()
        caller_id = context.agent_id

        children = await self._port.list_child_states(
            caller_id=caller_id,
            session_id=context.session_id,
        )

        if not children:
            return RuntimeToolOutcome(
                result=ToolResult.success(
                    tool_name=self.get_name(),
                    tool_call_id=parameters.get("tool_call_id", ""),
                    input_args=parameters,
                    content="No child agents found.",
                    output={"agents": []},
                    start_time=start_time,
                )
            )

        now = datetime.now(timezone.utc)
        agent_infos: list[dict[str, Any]] = []
        content_lines: list[str] = [f"Child agents ({len(children)} total):\n"]

        for state in children:
            running_secs = self._port.age_seconds(state.created_at, now=now)

            info: dict[str, Any] = {
                "agent_id": state.id,
                "status": state.status.value,
                "task": (
                    state.task[:100] + "..."
                    if isinstance(state.task, str) and len(str(state.task)) > 100
                    else state.task
                )
                if isinstance(state.task, str)
                else str(state.task),
                "created_ago_seconds": running_secs,
                "wake_count": state.wake_count,
                "explain": state.explain,
                "last_activity_at": state.last_activity_at.isoformat()
                if state.last_activity_at
                else None,
                "result_summary": summarize_text(state.result_summary, 200),
            }
            if state.recent_steps:
                info["recent_steps"] = state.recent_steps[-3:]

            agent_infos.append(info)

            task_str = info["task"]
            status_str = state.status.value.upper()
            line = f"- [{status_str}] {state.id}: {task_str} (running {running_secs}s, woke {state.wake_count}x)"
            for detail in build_child_result_detail_lines(
                result=summarize_text(state.result_summary, 100),
                explain=state.explain,
            ):
                line += f"\n  {detail}"
            if state.last_activity_at:
                activity_ago = self._port.age_seconds(state.last_activity_at, now=now)
                line += f"\n  Last activity: {activity_ago}s ago"
            content_lines.append(line)

        return RuntimeToolOutcome(
            result=ToolResult.success(
                tool_name=self.get_name(),
                tool_call_id=parameters.get("tool_call_id", ""),
                input_args=parameters,
                content="\n".join(content_lines),
                output={"agents": agent_infos},
                start_time=start_time,
            )
        )
