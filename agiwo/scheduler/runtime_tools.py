"""Scheduling runtime tools injected by the Scheduler."""

import time
from datetime import datetime, timezone
from typing import Any

from agiwo.agent import TerminationReason
from agiwo.scheduler.commands import (
    CancelChildRequest,
    SleepRequest,
    SpawnChildRequest,
)
from agiwo.scheduler.formatting import (
    build_child_result_detail_lines,
    summarize_text,
)
from agiwo.scheduler.models import TimeUnit, WaitMode, WakeType
from agiwo.scheduler.tool_control import SchedulerToolControl
from agiwo.tool.base import BaseTool, ToolGateDecision, ToolResult
from agiwo.tool.context import ToolContext
from agiwo.utils.abort_signal import AbortSignal


def _build_failed_result(
    tool_name: str,
    error: str,
    context: ToolContext,
    parameters: dict[str, Any],
    start_time: float,
) -> ToolResult:
    """Helper to build failed ToolResult with common parameters."""
    return ToolResult.failed(
        tool_name=tool_name,
        error=error,
        tool_call_id=context.tool_call_id,
        input_args=parameters,
        start_time=start_time,
    )


def _parse_optional_string_list(
    value: object,
    *,
    field_name: str,
) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    return list(value)


class _BaseSpawnChildTool(BaseTool):
    """Shared scheduler child-spawn runtime tool behavior."""

    _fork: bool = False
    _success_verb = "Spawned"

    def __init__(self, port: SchedulerToolControl) -> None:
        self._port = port
        super().__init__()

    async def gate(
        self,
        parameters: dict[str, Any],
        context: ToolContext,
    ) -> ToolGateDecision:
        del parameters
        if context.depth > 0:
            return ToolGateDecision.deny("Child agents cannot spawn further agents.")
        return ToolGateDecision.allow()

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ToolContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        del abort_signal
        start_time = time.time()
        parent_agent_id = context.agent_id
        if not parent_agent_id:
            return _build_failed_result(
                tool_name=self.name,
                error="Cannot spawn agent: no agent_id in execution context",
                context=context,
                parameters=parameters,
                start_time=start_time,
            )

        task = parameters.get("task", "")
        try:
            state = await self._port.spawn_child(
                SpawnChildRequest(
                    parent_agent_id=parent_agent_id,
                    session_id=context.session_id,
                    task=task,
                    instruction=self._get_instruction(parameters),
                    allowed_skills=self._get_allowed_skills(parameters),
                    allowed_tools=self._get_allowed_tools(parameters),
                    fork=self._fork,
                )
            )
        except ValueError as exc:
            return _build_failed_result(
                tool_name=self.name,
                error=str(exc),
                context=context,
                parameters=parameters,
                start_time=start_time,
            )

        return ToolResult.success(
            tool_name=self.name,
            tool_call_id=context.tool_call_id,
            input_args={"task": task, "child_id": state.id},
            content=f"{self._success_verb} child agent '{state.id}' for task: {task}",
            output={"child_id": state.id, "status": "pending"},
            start_time=start_time,
        )

    def _get_instruction(self, parameters: dict[str, Any]) -> str | None:
        del parameters
        return None

    def _get_allowed_skills(self, parameters: dict[str, Any]) -> list[str] | None:
        del parameters
        return None

    def _get_allowed_tools(self, parameters: dict[str, Any]) -> list[str] | None:
        del parameters
        return None


class SpawnChildAgentTool(_BaseSpawnChildTool):
    """Spawn a regular child agent to handle an independent sub-task."""

    name = "spawn_child_agent"
    description = """Spawn a child agent to handle an independent sub-task asynchronously.
Use this for genuine delegation or parallel work.
Do not use it for simple actions you can do directly.
The child agent cannot spawn further child agents.
Call sleep_and_wait to wait for completion if needed."""

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": """Brief, complete task for the child agent, including the goal, required context, and expected outcome. Specify the outcome, not a process script. Do not ask it to spawn additional child agents.""",
                },
                "instruction": {
                    "type": "string",
                    "description": "Optional instruction that guides how the child agent should approach the task.",
                },
                "allowed_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional explicit functional tool list for the child agent. Must be a subset of the parent agent's allowed_tools when the parent is restricted.",
                },
                "allowed_skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional explicit skill name list for the child agent. Must already be expanded and must be a subset of the parent allowed skills.",
                },
            },
            "required": ["task"],
        }

    def _get_instruction(self, parameters: dict[str, Any]) -> str | None:
        instruction = parameters.get("instruction")
        return instruction if isinstance(instruction, str) else None

    def _get_allowed_skills(self, parameters: dict[str, Any]) -> list[str] | None:
        return _parse_optional_string_list(
            parameters.get("allowed_skills"),
            field_name="allowed_skills",
        )

    def _get_allowed_tools(self, parameters: dict[str, Any]) -> list[str] | None:
        return _parse_optional_string_list(
            parameters.get("allowed_tools"),
            field_name="allowed_tools",
        )


class ForkChildAgentTool(_BaseSpawnChildTool):
    """Fork the current agent into a child that inherits parent context."""

    name = "fork_child_agent"
    description = """Fork the current agent into a child that inherits the parent conversation context.
Use this only when the child should continue from the same prompt and context prefix.
The child agent cannot spawn further child agents."""
    _fork = True
    _success_verb = "Forked"

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Brief task for the forked child agent to complete using the inherited parent context.",
                },
            },
            "required": ["task"],
        }


class SleepAndWaitTool(BaseTool):
    """Put the current agent to sleep until a wake condition is met."""

    name = "sleep_and_wait"
    description = (
        "Put the current agent to sleep and wait for a condition. "
        "Use 'waitset' to wait for spawned child agents to finish. "
        "Use 'timer' to sleep for a fixed duration. "
        "Use 'periodic' to periodically wake up and check."
    )
    concurrency_safe = False

    def __init__(self, port: SchedulerToolControl) -> None:
        self._port = port
        super().__init__()

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
                "no_progress": {
                    "type": "boolean",
                    "description": (
                        "Set to true when this periodic check found no meaningful "
                        "progress. The system will discard this round's context to "
                        "keep the conversation window clean. Only effective with "
                        "wake_type='periodic'."
                    ),
                    "default": False,
                },
            },
            "required": ["wake_type"],
        }

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ToolContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        del abort_signal
        start_time = time.time()
        agent_id = context.agent_id
        if not agent_id:
            return _build_failed_result(
                tool_name=self.name,
                error="Cannot sleep: no agent_id in execution context",
                context=context,
                parameters=parameters,
                start_time=start_time,
            )

        wake_type_str = parameters.get("wake_type", "waitset")
        delay_seconds = parameters.get("delay_seconds")
        time_unit_str = parameters.get("time_unit", "seconds")
        wait_mode_str = parameters.get("wait_mode", "all")
        wait_for: list[str] | None = parameters.get("wait_for")
        timeout = parameters.get("timeout")
        explain = parameters.get("explain")
        no_progress = bool(parameters.get("no_progress", False))

        try:
            wake_type = WakeType(wake_type_str)
        except ValueError:
            return _build_failed_result(
                tool_name=self.name,
                error=f"Invalid wake_type: {wake_type_str}",
                context=context,
                parameters=parameters,
                start_time=start_time,
            )

        try:
            sleep_result = await self._port.sleep_current_agent(
                SleepRequest(
                    agent_id=agent_id,
                    session_id=context.session_id,
                    wake_type=wake_type,
                    wait_mode=self._resolve_wait_mode(wait_mode_str),
                    wait_for=wait_for,
                    timeout=timeout,
                    delay_seconds=delay_seconds,
                    time_unit=self._resolve_time_unit(time_unit_str),
                    explain=explain,
                    no_progress=no_progress and wake_type == WakeType.PERIODIC,
                )
            )
        except ValueError as exc:
            return _build_failed_result(
                tool_name=self.name,
                error=str(exc),
                context=context,
                parameters=parameters,
                start_time=start_time,
            )

        return ToolResult.success(
            tool_name=self.name,
            tool_call_id=context.tool_call_id,
            input_args=parameters,
            content=sleep_result.summary,
            output={"wake_type": wake_type_str, "agent_id": agent_id},
            start_time=start_time,
            termination_reason=TerminationReason.SLEEPING,
        )

    def _resolve_wait_mode(self, value: str) -> WaitMode:
        try:
            return WaitMode(value)
        except ValueError:
            return WaitMode.ALL

    def _resolve_time_unit(self, value: str) -> TimeUnit:
        try:
            return TimeUnit(value)
        except ValueError:
            return TimeUnit.SECONDS


class QuerySpawnedAgentTool(BaseTool):
    """Query the status and result of a spawned child agent."""

    name = "query_spawned_agent"
    description = (
        "Query the current status and result of a spawned child agent. "
        "Use this after waking up to check what the child agents have accomplished."
    )

    def __init__(self, port: SchedulerToolControl) -> None:
        self._port = port
        super().__init__()

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

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ToolContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        del abort_signal
        start_time = time.time()
        target_id = parameters.get("agent_id", "")

        state = await self._port.get_child_state(target_id)
        if state is None:
            return _build_failed_result(
                tool_name=self.name,
                error=f"Agent '{target_id}' not found",
                context=context,
                parameters=parameters,
                start_time=start_time,
            )
        result_summary = await self._port.get_child_result_summary(target_id)

        result_info: dict[str, Any] = {
            "agent_id": state.id,
            "status": state.status.value,
            "task": state.task if isinstance(state.task, str) else str(state.task),
            "result_summary": result_summary,
            "explain": state.explain,
            "wake_count": state.wake_count,
        }

        content_parts = [
            f"Agent: {state.id}",
            f"Status: {state.status.value}",
            f"Task: {state.task if isinstance(state.task, str) else str(state.task)}",
        ]
        content_parts.extend(
            build_child_result_detail_lines(
                result=result_summary,
                explain=state.explain,
            )
        )

        return ToolResult.success(
            tool_name=self.name,
            tool_call_id=context.tool_call_id,
            input_args=parameters,
            content="\n".join(content_parts),
            output=result_info,
            start_time=start_time,
        )


class CancelAgentTool(BaseTool):
    """Cancel a child agent, optionally forcing termination even with running processes."""

    name = "cancel_agent"
    description = (
        "Cancel a spawned child agent and its entire subtree. "
        "Use force=false first to check for running processes; "
        "if confirmed safe, use force=true to terminate."
    )

    def __init__(self, port: SchedulerToolControl) -> None:
        self._port = port
        super().__init__()

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

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ToolContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        del abort_signal
        start_time = time.time()
        caller_id = context.agent_id
        target_id = parameters.get("agent_id", "")
        force = parameters.get("force", False)
        reason = parameters.get("reason", "Cancelled by parent agent")

        try:
            cancel_result = await self._port.cancel_child(
                CancelChildRequest(
                    caller_id=caller_id,
                    target_id=target_id,
                    force=force,
                    reason=reason,
                )
            )
        except PermissionError as exc:
            return ToolResult.denied(
                tool_name=self.name,
                reason=str(exc),
                tool_call_id=context.tool_call_id,
                input_args=parameters,
                content=f"Permission denied: {exc}.",
                output={"success": False},
                start_time=start_time,
            )

        if cancel_result.outcome == "missing":
            return _build_failed_result(
                tool_name=self.name,
                error=f"Agent '{target_id}' not found",
                context=context,
                parameters=parameters,
                start_time=start_time,
            )

        target_state = cancel_result.state
        if cancel_result.outcome == "already_terminal" and target_state is not None:
            return ToolResult.success(
                tool_name=self.name,
                tool_call_id=context.tool_call_id,
                input_args=parameters,
                content=(
                    f"Agent '{target_id}' is already in terminal state: "
                    f"{target_state.status.value}."
                ),
                output={"success": False, "status": target_state.status.value},
                start_time=start_time,
            )

        if cancel_result.outcome == "requires_force" and target_state is not None:
            content_parts = [
                f"Agent '{target_id}' is currently RUNNING.",
            ]
            proc_info: list[dict] = []
            if cancel_result.running_processes:
                content_parts.append(
                    f"\n{len(cancel_result.running_processes)} background process(es) are running under this agent:"
                )
                for p in cancel_result.running_processes:
                    content_parts.append(f"  - [{p['process_id']}] {p['command']}")
                    proc_info.append(p)
                content_parts.append(
                    "\nUse force=true to terminate the agent and all its processes."
                )
            else:
                content_parts.append(
                    "\nNo background processes detected. Use force=true to cancel."
                )
            return ToolResult.success(
                tool_name=self.name,
                tool_call_id=context.tool_call_id,
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

        return ToolResult.success(
            tool_name=self.name,
            tool_call_id=context.tool_call_id,
            input_args=parameters,
            content=f"Agent '{target_id}' and its subtree have been cancelled. Reason: {reason}",
            output={"success": True, "agent_id": target_id},
            start_time=start_time,
        )


class ListAgentsTool(BaseTool):
    """List all direct child agents of the calling agent with detailed status information."""

    name = "list_agents"
    description = (
        "List all direct child agents with their status, task, and results. "
        "Use this to get an overview of all spawned agents and decide on next steps."
    )

    def __init__(self, port: SchedulerToolControl) -> None:
        self._port = port
        super().__init__()

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ToolContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        del abort_signal
        start_time = time.time()
        caller_id = context.agent_id

        children = await self._port.list_child_states(
            caller_id=caller_id,
            session_id=context.session_id,
        )

        if not children:
            return ToolResult.success(
                tool_name=self.name,
                tool_call_id=context.tool_call_id,
                input_args=parameters,
                content="No child agents found.",
                output={"agents": []},
                start_time=start_time,
            )

        now = datetime.now(timezone.utc)
        agent_infos: list[dict[str, Any]] = []
        content_lines: list[str] = [f"Child agents ({len(children)} total):\n"]

        for state in children:
            running_secs = self._port.age_seconds(state.created_at, now=now)
            result_summary = await self._port.get_child_result_summary(state.id)

            info: dict[str, Any] = {
                "agent_id": state.id,
                "status": state.status.value,
                "task": (
                    (
                        state.task[:100] + "..."
                        if isinstance(state.task, str) and len(str(state.task)) > 100
                        else state.task
                    )
                    if isinstance(state.task, str)
                    else str(state.task)
                ),
                "created_ago_seconds": running_secs,
                "wake_count": state.wake_count,
                "explain": state.explain,
                "result_summary": summarize_text(result_summary, 200),
            }

            agent_infos.append(info)

            task_str = info["task"]
            status_str = state.status.value.upper()
            line = f"- [{status_str}] {state.id}: {task_str} (running {running_secs}s, woke {state.wake_count}x)"
            for detail in build_child_result_detail_lines(
                result=summarize_text(result_summary, 100),
                explain=state.explain,
            ):
                line += f"\n  {detail}"
            content_lines.append(line)

        return ToolResult.success(
            tool_name=self.name,
            tool_call_id=context.tool_call_id,
            input_args=parameters,
            content="\n".join(content_lines),
            output={"agents": agent_infos},
            start_time=start_time,
        )


class DeclareMilestonesTool(BaseTool):
    """Declare or update the milestones for the current task."""

    name = "declare_milestones"
    description = (
        "Declare or update the milestones for the current task. "
        "Break the user's request into concrete, verifiable sub-goals. "
        "Each milestone should have a clear id and a specific description "
        "of what 'done' looks like. The system uses these milestones to "
        "evaluate whether your work stays on track.\n\n"
        'Example: [{"id":"understand","description":"Identify how '
        'auth tokens are validated"}, {"id":"fix","description":'
        '"Apply the fix and verify with tests"}]'
    )
    concurrency_safe = False

    def __init__(self, port: SchedulerToolControl) -> None:
        self._port = port
        super().__init__()

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "milestones": {
                    "type": "array",
                    "description": (
                        "Ordered list of milestone objects. Each must have: "
                        "id (string, unique identifier) and "
                        "description (string, what 'done' looks like)."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["id", "description"],
                    },
                },
            },
            "required": ["milestones"],
        }

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ToolContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        del abort_signal
        start_time = time.time()
        milestones = parameters.get("milestones", [])
        if not isinstance(milestones, list):
            return _build_failed_result(
                tool_name=self.name,
                error="milestones must be an array",
                context=context,
                parameters=parameters,
                start_time=start_time,
            )
        if not milestones:
            return _build_failed_result(
                tool_name=self.name,
                error="milestones must be a non-empty array",
                context=context,
                parameters=parameters,
                start_time=start_time,
            )
        ids: list[str] = []
        normalized_milestones: list[dict[str, str]] = []
        for milestone in milestones:
            if (
                not isinstance(milestone, dict)
                or "id" not in milestone
                or not isinstance(milestone["id"], str)
                or not milestone["id"].strip()
                or "description" not in milestone
                or not isinstance(milestone["description"], str)
                or not milestone["description"].strip()
            ):
                return _build_failed_result(
                    tool_name=self.name,
                    error=(
                        "milestones must be an array of objects each containing "
                        "non-empty string 'id' and 'description' fields"
                    ),
                    context=context,
                    parameters=parameters,
                    start_time=start_time,
                )
            milestone_id = milestone["id"].strip()
            description = milestone["description"].strip()
            ids.append(milestone_id)
            normalized_milestones.append(
                {
                    "id": milestone_id,
                    "description": description,
                }
            )
        return ToolResult.success(
            tool_name=self.name,
            tool_call_id=context.tool_call_id,
            input_args=parameters,
            content=f"Milestones declared: {', '.join(ids)}",
            output={"milestones": normalized_milestones},
            start_time=start_time,
        )


class ReviewTrajectoryTool(BaseTool):
    """Respond to a <system-review> prompt."""

    name = "review_trajectory"
    description = (
        "Respond to a <system-review> prompt by assessing whether "
        "your recent tool calls advance the active milestone.\n\n"
        "Parameters:\n"
        "- aligned (boolean, required): true if trajectory aligns with "
        "milestone, false if drifted.\n"
        "- experience (string, required when aligned=false): A concise "
        "summary covering what was attempted, what was learned, and "
        "how this should inform the next approach.\n\n"
        "This tool call and result are temporary review metadata and may be "
        "removed after processing."
    )
    concurrency_safe = False

    def __init__(self, port: SchedulerToolControl) -> None:
        self._port = port
        super().__init__()

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "aligned": {
                    "type": "boolean",
                    "description": (
                        "true if your recent trajectory aligns with the "
                        "active milestone, false if it has drifted."
                    ),
                },
                "experience": {
                    "type": "string",
                    "description": (
                        "Required when aligned=false. A concise summary: "
                        "what was attempted, what was learned, and how "
                        "this should inform the next approach."
                    ),
                },
            },
            "required": ["aligned"],
        }

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ToolContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        del abort_signal
        start_time = time.time()
        aligned = parameters.get("aligned")
        experience = parameters.get("experience", "")

        if not isinstance(aligned, bool):
            return _build_failed_result(
                tool_name=self.name,
                error="aligned must be a boolean",
                context=context,
                parameters=parameters,
                start_time=start_time,
            )
        if experience is None:
            experience = ""
        if not isinstance(experience, str):
            return _build_failed_result(
                tool_name=self.name,
                error="experience must be a string",
                context=context,
                parameters=parameters,
                start_time=start_time,
            )
        if not aligned and not experience:
            return _build_failed_result(
                tool_name=self.name,
                error="experience is required when aligned=false",
                context=context,
                parameters=parameters,
                start_time=start_time,
            )

        content = (
            f"Trajectory review: aligned={aligned}. {experience}"
            if experience
            else f"Trajectory review: aligned={aligned}."
        )
        return ToolResult.success(
            tool_name=self.name,
            tool_call_id=context.tool_call_id,
            input_args=parameters,
            content=content,
            output={"aligned": aligned, "experience": experience},
            start_time=start_time,
        )
