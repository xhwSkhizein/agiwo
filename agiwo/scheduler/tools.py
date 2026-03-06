"""
Scheduling tools — SpawnAgentTool, SleepAndWaitTool, QuerySpawnedAgentTool,
CancelAgentTool, ListAgentsTool.

These tools are injected into agents by the Scheduler. They read `context.agent_id`
directly (no SchedulingMeta needed) and interact with AgentStateStorage.
"""

import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any
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

if TYPE_CHECKING:
    from agiwo.scheduler.scheduler import Scheduler


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
                "explain": {
                    "type": "string",
                    "description": "Optional brief reason for sleeping — visible to the parent agent and operators for observability.",
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
        explain = parameters.get("explain")

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
            explain=explain,
        )

        end_time = time.time()
        summary = f"Agent '{agent_id}' entering sleep. Wake condition: {wake_type_str}"
        if wake_type == WakeType.WAITSET:
            summary += f" (waiting_for={len(wc.wait_for)}, mode={wc.wait_mode.value}, already_done={len(wc.completed_ids)})"
        elif delay_seconds is not None:
            summary += f" (delay={delay_seconds} {time_unit_str})"
        if explain:
            summary += f" | reason: {explain}"

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

        result_info: dict[str, Any] = {
            "agent_id": state.id,
            "status": state.status.value,
            "task": state.task if isinstance(state.task, str) else str(state.task),
            "result_summary": state.result_summary,
            "explain": state.explain,
            "wake_count": state.wake_count,
            "last_activity_at": state.last_activity_at.isoformat() if state.last_activity_at else None,
        }
        if state.recent_steps:
            result_info["recent_steps"] = state.recent_steps

        content_parts = [
            f"Agent: {state.id}",
            f"Status: {state.status.value}",
            f"Task: {state.task if isinstance(state.task, str) else str(state.task)}",
        ]
        if state.explain:
            content_parts.append(f"Sleep reason: {state.explain}")
        if state.last_activity_at:
            content_parts.append(f"Last activity: {state.last_activity_at.isoformat()}")
        if state.result_summary:
            content_parts.append(f"Result: {state.result_summary}")
        if state.recent_steps:
            steps_text = []
            for step in state.recent_steps[-3:]:
                role = step.get("role", "?")
                ts = step.get("timestamp", "")
                tools = step.get("tool_calls", [])
                if tools:
                    steps_text.append(f"  [{ts[:19]}] {role}: called {', '.join(tools)}")
                else:
                    steps_text.append(f"  [{ts[:19]}] {role}")
            content_parts.append("Recent steps:\n" + "\n".join(steps_text))

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


class CancelAgentTool(BaseTool):
    """Cancel a child agent, optionally forcing termination even with running processes."""

    def __init__(self, scheduler: "Scheduler") -> None:
        self._scheduler = scheduler
        super().__init__()

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

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        start_time = time.time()
        caller_id = context.agent_id
        target_id = parameters.get("agent_id", "")
        force = parameters.get("force", False)
        reason = parameters.get("reason", "Cancelled by parent agent")

        target_state = await self._scheduler.store.get_state(target_id)
        if target_state is None:
            end_time = time.time()
            return ToolResult(
                tool_name=self.get_name(),
                tool_call_id=parameters.get("tool_call_id", ""),
                input_args=parameters,
                content=f"Agent '{target_id}' not found.",
                output={"success": False},
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                is_success=False,
                error=f"Agent '{target_id}' not found",
            )

        if target_state.parent_id != caller_id:
            end_time = time.time()
            return ToolResult(
                tool_name=self.get_name(),
                tool_call_id=parameters.get("tool_call_id", ""),
                input_args=parameters,
                content=f"Permission denied: agent '{target_id}' is not a direct child of '{caller_id}'.",
                output={"success": False},
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                is_success=False,
                error="Permission denied: not a direct child",
            )

        if target_state.status not in (
            AgentStateStatus.RUNNING,
            AgentStateStatus.SLEEPING,
            AgentStateStatus.PENDING,
        ):
            end_time = time.time()
            return ToolResult(
                tool_name=self.get_name(),
                tool_call_id=parameters.get("tool_call_id", ""),
                input_args=parameters,
                content=f"Agent '{target_id}' is already in terminal state: {target_state.status.value}.",
                output={"success": False, "status": target_state.status.value},
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
            )

        if not force and target_state.status == AgentStateStatus.RUNNING:
            # Check for running BashTool background processes before cancelling
            running_procs = await self._get_running_bash_processes(target_id)
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
                content_parts.append("\nUse force=true to terminate the agent and all its processes.")
            else:
                content_parts.append("\nNo background processes detected. Use force=true to cancel.")
            end_time = time.time()
            return ToolResult(
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
                end_time=end_time,
                duration=end_time - start_time,
            )

        await self._scheduler._recursive_cancel(target_id, reason)

        end_time = time.time()
        return ToolResult(
            tool_name=self.get_name(),
            tool_call_id=parameters.get("tool_call_id", ""),
            input_args=parameters,
            content=f"Agent '{target_id}' and its subtree have been cancelled. Reason: {reason}",
            output={"success": True, "agent_id": target_id},
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
        )

    async def _get_running_bash_processes(self, agent_id: str) -> list[dict]:
        """Find running BashTool background processes for the given agent."""
        from agiwo.tool.builtin.bash_tool.tool import BashTool

        agent = self._scheduler._agents.get(agent_id)
        if agent is None:
            return []
        for tool in agent.tools:
            if isinstance(tool, BashTool):
                try:
                    procs = await tool.config.sandbox.list_processes_by_agent(agent_id, state="running")
                    return [
                        {
                            "process_id": p.process_id,
                            "command": p.command,
                            "started_at": p.started_at,
                        }
                        for p in procs
                    ]
                except Exception:
                    return []
        return []


class ListAgentsTool(BaseTool):
    """List all direct child agents of the calling agent with detailed status information."""

    def __init__(self, store: AgentStateStorage) -> None:
        self._store = store
        super().__init__()

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

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        start_time = time.time()
        caller_id = context.agent_id

        children = await self._store.get_states_by_parent(caller_id)
        # Filter to current session
        children = [c for c in children if c.session_id == context.session_id]

        if not children:
            end_time = time.time()
            return ToolResult(
                tool_name=self.get_name(),
                tool_call_id=parameters.get("tool_call_id", ""),
                input_args=parameters,
                content="No child agents found.",
                output={"agents": []},
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
            )

        now = datetime.now(timezone.utc)
        agent_infos: list[dict[str, Any]] = []
        content_lines: list[str] = [f"Child agents ({len(children)} total):\n"]

        for state in children:
            created_ts = state.created_at
            if created_ts.tzinfo is None:
                created_ts = created_ts.replace(tzinfo=timezone.utc)
            running_secs = int((now - created_ts).total_seconds())

            info: dict[str, Any] = {
                "agent_id": state.id,
                "status": state.status.value,
                "task": (state.task[:100] + "..." if isinstance(state.task, str) and len(str(state.task)) > 100 else state.task) if isinstance(state.task, str) else str(state.task),
                "created_ago_seconds": running_secs,
                "wake_count": state.wake_count,
                "explain": state.explain,
                "last_activity_at": state.last_activity_at.isoformat() if state.last_activity_at else None,
                "result_summary": (state.result_summary[:200] + "..." if state.result_summary and len(state.result_summary) > 200 else state.result_summary),
            }
            if state.recent_steps:
                info["recent_steps"] = state.recent_steps[-3:]

            agent_infos.append(info)

            task_str = info["task"]
            status_str = state.status.value.upper()
            line = f"- [{status_str}] {state.id}: {task_str} (running {running_secs}s, woke {state.wake_count}x)"
            if state.explain:
                line += f"\n  Sleep reason: {state.explain}"
            if state.result_summary:
                summary_short = state.result_summary[:100] + ("..." if len(state.result_summary) > 100 else "")
                line += f"\n  Result: {summary_short}"
            if state.last_activity_at:
                activity_ago = int((now - (state.last_activity_at if state.last_activity_at.tzinfo else state.last_activity_at.replace(tzinfo=timezone.utc))).total_seconds())
                line += f"\n  Last activity: {activity_ago}s ago"
            content_lines.append(line)

        end_time = time.time()
        return ToolResult(
            tool_name=self.get_name(),
            tool_call_id=parameters.get("tool_call_id", ""),
            input_args=parameters,
            content="\n".join(content_lines),
            output={"agents": agent_infos},
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
        )
