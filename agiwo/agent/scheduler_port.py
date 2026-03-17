from dataclasses import dataclass, field
from typing import Awaitable, Callable, Protocol, runtime_checkable

from agiwo.agent.execution import AgentExecutionHandlePort
from agiwo.agent.input import UserInput
from agiwo.agent.runtime import StepRecord
from agiwo.utils.abort_signal import AbortSignal


StepObserver = Callable[[StepRecord], Awaitable[None]]


class SchedulerToolLike(Protocol):
    def get_name(self) -> str:
        ...


@dataclass(frozen=True)
class ChildAgentOverrides:
    instruction: str | None = None
    system_prompt: str | None = None
    exclude_tool_names: set[str] = field(default_factory=set)


@runtime_checkable
class SchedulerAgentPort(Protocol):
    id: str

    @property
    def tools(self) -> tuple[SchedulerToolLike, ...]:
        ...

    def install_runtime_tools(self, tools: list[object]) -> None:
        ...

    def set_termination_summary_enabled(self, enabled: bool) -> None:
        ...

    def start(
        self,
        user_input: UserInput,
        *,
        session_id: str | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> AgentExecutionHandlePort:
        ...

    async def create_scheduler_child(
        self,
        *,
        child_id: str,
        overrides: ChildAgentOverrides,
    ) -> "SchedulerAgentPort":
        ...

    async def close(self) -> None:
        ...

    def unwrap_agent(self) -> object | None:
        ...


class AgentSchedulerPort:
    """Stable scheduler-facing adapter over the concrete Agent implementation."""

    def __init__(self, agent) -> None:
        self._agent = agent

    @property
    def id(self) -> str:
        return self._agent.id

    @property
    def tools(self) -> tuple[SchedulerToolLike, ...]:
        return self._agent.tools

    def install_runtime_tools(self, tools: list[object]) -> None:
        self._agent.install_runtime_tools(tools)

    def set_termination_summary_enabled(self, enabled: bool) -> None:
        self._agent.set_termination_summary_enabled(enabled)

    def start(
        self,
        user_input: UserInput,
        *,
        session_id: str | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> AgentExecutionHandlePort:
        return self._agent.start(
            user_input,
            session_id=session_id,
            abort_signal=abort_signal,
        )

    async def create_scheduler_child(
        self,
        *,
        child_id: str,
        overrides: ChildAgentOverrides,
    ) -> "SchedulerAgentPort":
        child = await self._agent.create_scheduler_child_agent(
            child_id=child_id,
            instruction=overrides.instruction,
            system_prompt=overrides.system_prompt,
            exclude_tool_names=overrides.exclude_tool_names,
        )
        return AgentSchedulerPort(child)

    async def close(self) -> None:
        await self._agent.close()

    def unwrap_agent(self) -> object | None:
        return self._agent


def adapt_scheduler_agent(agent) -> SchedulerAgentPort:
    if isinstance(agent, AgentSchedulerPort):
        return agent
    if isinstance(agent, SchedulerAgentPort):
        return agent
    return AgentSchedulerPort(agent)


__all__ = [
    "AgentSchedulerPort",
    "ChildAgentOverrides",
    "SchedulerAgentPort",
    "StepObserver",
    "adapt_scheduler_agent",
]
