"""Live root execution handle owned by a SessionRuntime."""

from asyncio import Task
from collections.abc import AsyncIterator

from agiwo.agent.models.input import UserInput
from agiwo.agent.models.run import RunOutput
from agiwo.agent.models.stream import AgentStreamItem
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.session import SessionRuntime


class AgentExecutionHandle:
    """One live root execution owned by a SessionRuntime."""

    def __init__(
        self,
        *,
        run_id: str,
        session_id: str,
        session_runtime: SessionRuntime,
        task: Task[RunOutput],
        context: RunContext | None = None,
    ) -> None:
        self._run_id = run_id
        self._session_id = session_id
        self._session_runtime = session_runtime
        self._task = task
        self._context = context

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def session_id(self) -> str:
        return self._session_id

    def stream(self) -> AsyncIterator[AgentStreamItem]:
        return self._session_runtime.subscribe()

    async def wait(self) -> RunOutput:
        return await self._task

    @property
    def is_active(self) -> bool:
        return not self._task.done()

    async def enqueue_message(self, user_input: UserInput) -> bool:
        """Append one pending user-role message for the next loop turn."""
        return await self._session_runtime.enqueue_message(user_input)

    def cancel(self, reason: str | None = None) -> None:
        self._session_runtime.abort_signal.abort(reason or "Cancelled by caller")

    def request_pause(self, reason: str) -> None:
        """Ask the live run to pause at the next safe boundary (not cancel)."""
        if self._context is not None:
            self._context.request_pause(reason)
