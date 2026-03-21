from asyncio import Task
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agiwo.agent.runtime import AgentStreamItem, RunOutput


@dataclass(frozen=True)
class ChildAgentSpec:
    """Pure child execution overrides resolved against an Agent at runtime."""

    agent_id: str
    agent_name: str
    description: str
    instruction: str | None = None
    system_prompt_override: str | None = None
    exclude_tool_names: frozenset[str] = field(default_factory=frozenset)
    metadata_overrides: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class AgentExecutionHandlePort(Protocol):
    run_id: str
    session_id: str

    def stream(self) -> AsyncIterator[AgentStreamItem]: ...

    async def wait(self) -> RunOutput: ...

    async def steer(self, message: str) -> bool: ...

    def cancel(self, reason: str | None = None) -> None: ...


@runtime_checkable
class AbortSignalPort(Protocol):
    def abort(self, reason: str) -> None: ...


@runtime_checkable
class AgentExecutionSessionPort(Protocol):
    abort_signal: AbortSignalPort

    def subscribe(self) -> AsyncIterator[AgentStreamItem]: ...

    async def enqueue_steer(self, message: str) -> bool: ...


class AgentExecutionHandle(AgentExecutionHandlePort):
    """One live root execution owned by an AgentSessionRuntime."""

    def __init__(
        self,
        *,
        run_id: str,
        session_id: str,
        session_runtime: AgentExecutionSessionPort,
        task: Task[RunOutput],
    ) -> None:
        self._run_id = run_id
        self._session_id = session_id
        self._session_runtime = session_runtime
        self._task = task

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

    async def steer(self, message: str) -> bool:
        return await self._session_runtime.enqueue_steer(message)

    def cancel(self, reason: str | None = None) -> None:
        self._session_runtime.abort_signal.abort(reason or "Cancelled by caller")


__all__ = [
    "AgentExecutionHandle",
    "AgentExecutionHandlePort",
    "ChildAgentSpec",
]
