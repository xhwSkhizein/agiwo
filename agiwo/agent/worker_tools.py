"""System tools for MainAgent Worker spawn (ADR 0049 Wave D)."""

import time
from typing import Any

from agiwo.agent.worker import WorkerService
from agiwo.tool.base import BaseTool, ToolGateDecision, ToolResult
from agiwo.tool.context import ToolContext
from agiwo.utils.abort_signal import AbortSignal


class SpawnWorkerTool(BaseTool):
    """Spawn a one-shot Scheduler-delegated Worker (not Agent-as-Tool)."""

    name = "spawn_worker"
    description = (
        "Spawn a one-shot Worker to handle an independent sub-task. "
        "Workers run with an isolated agent_id; only their final report "
        "returns to the main conversation. "
        "Set sync=true to block until the Worker completes; sync=false "
        "continues immediately and delivers the report via the main queue later. "
        "Workers cannot spawn further Workers."
    )
    concurrency_safe = False

    def __init__(self, service: WorkerService) -> None:
        self._service = service
        super().__init__()

    async def gate(
        self,
        parameters: dict[str, Any],
        context: ToolContext,
    ) -> ToolGateDecision:
        del parameters
        if context.depth > 0:
            return ToolGateDecision.deny("Workers cannot spawn Workers.")
        if not context.agent_id:
            return ToolGateDecision.deny("Missing agent_id in execution context.")
        return ToolGateDecision.allow()

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Brief task for the Worker including goal and expected outcome.",
                },
                "sync": {
                    "type": "boolean",
                    "description": "When true, block until the Worker completes and return its report.",
                    "default": False,
                },
                "instruction": {
                    "type": "string",
                    "description": "Optional guidance for how the Worker should approach the task.",
                },
            },
            "required": ["task"],
        }

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ToolContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        del abort_signal
        start_time = time.time()
        task = parameters.get("task", "")
        if not isinstance(task, str) or not task.strip():
            return ToolResult.failed(
                tool_name=self.name,
                error="task must be a non-empty string",
                tool_call_id=context.tool_call_id,
                input_args=parameters,
                start_time=start_time,
            )

        sync = bool(parameters.get("sync", False))
        instruction = parameters.get("instruction")
        instruction_text = instruction if isinstance(instruction, str) else None

        try:
            handle, report = await self._service.spawn_worker(
                task=task,
                main_run_id=context.run_id or "",
                sync=sync,
                instruction=instruction_text,
            )
        except ValueError as exc:
            return ToolResult.failed(
                tool_name=self.name,
                error=str(exc),
                tool_call_id=context.tool_call_id,
                input_args=parameters,
                start_time=start_time,
            )

        if sync:
            assert report is not None
            return ToolResult.success(
                tool_name=self.name,
                tool_call_id=context.tool_call_id,
                input_args=parameters,
                content=f"Worker '{handle.worker_id}' completed.\n\n{report}",
                output={"worker_id": handle.worker_id, "report": report, "sync": True},
                start_time=start_time,
            )

        return ToolResult.success(
            tool_name=self.name,
            tool_call_id=context.tool_call_id,
            input_args=parameters,
            content=(
                f"Spawned async Worker '{handle.worker_id}' for task: {task}. "
                "The report will arrive on the main queue when it completes."
            ),
            output={"worker_id": handle.worker_id, "sync": False},
            start_time=start_time,
        )


__all__ = ["SpawnWorkerTool"]
