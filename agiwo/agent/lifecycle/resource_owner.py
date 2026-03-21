import asyncio
from asyncio import Task
from dataclasses import dataclass
from typing import Callable

from agiwo.agent.runtime import RunOutput
from agiwo.agent.storage.base import RunStepStorage
from agiwo.agent.storage.session import SessionStorage
from agiwo.observability.base import BaseTraceStorage


@dataclass(slots=True)
class ActiveRootExecution:
    run_id: str
    task: Task[RunOutput]
    cancel_callback: Callable[[str | None], None]

    async def wait(self) -> RunOutput:
        return await self.task

    def cancel(self, reason: str | None = None) -> None:
        self.cancel_callback(reason)


class AgentResourceOwner:
    """Own shared agent resources and root execution lifecycle."""

    def __init__(
        self,
        *,
        run_step_storage: RunStepStorage,
        session_storage: SessionStorage,
        trace_storage: BaseTraceStorage | None,
    ) -> None:
        self.run_step_storage = run_step_storage
        self.session_storage = session_storage
        self.trace_storage = trace_storage
        self._active_executions: dict[str, ActiveRootExecution] = {}
        self._closing = False
        self._closed = False
        self._close_lock = asyncio.Lock()

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def is_closing(self) -> bool:
        return self._closing

    def ensure_open(self) -> None:
        if self._closing or self._closed:
            raise RuntimeError("agent_closed")

    def register_execution(self, execution: ActiveRootExecution) -> None:
        self.ensure_open()
        if execution.task.done():
            return
        self._active_executions[execution.run_id] = execution
        execution.task.add_done_callback(
            lambda _task, run_id=execution.run_id: self._active_executions.pop(
                run_id, None
            )
        )

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closing = True
            active_executions = tuple(self._active_executions.values())
            for execution in active_executions:
                execution.cancel("Agent closed")
            if active_executions:
                await asyncio.gather(
                    *[execution.wait() for execution in active_executions],
                    return_exceptions=True,
                )
            await self.run_step_storage.close()
            await self.session_storage.close()
            if self.trace_storage is not None:
                await self.trace_storage.close()
            self._active_executions.clear()
            self._closed = True
            self._closing = False


__all__ = ["ActiveRootExecution", "AgentResourceOwner"]
