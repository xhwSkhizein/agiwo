from dataclasses import dataclass, field
from typing import Awaitable, Callable, Protocol, runtime_checkable

from agiwo.agent.input import UserInput
from agiwo.agent.runtime import RunOutput, StepRecord
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

    def add_step_observer(self, observer: StepObserver) -> None:
        ...

    def remove_step_observer(self, observer: StepObserver) -> None:
        ...

    async def run(
        self,
        user_input: UserInput,
        *,
        session_id: str | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> RunOutput:
        ...

    async def derive_child_for_scheduler(
        self,
        *,
        child_id: str,
        overrides: ChildAgentOverrides,
    ) -> "SchedulerAgentPort":
        ...

    async def steer(self, message: str) -> bool:
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

    def add_step_observer(self, observer: StepObserver) -> None:
        self._agent.add_step_observer(observer)

    def remove_step_observer(self, observer: StepObserver) -> None:
        self._agent.remove_step_observer(observer)

    async def run(
        self,
        user_input: UserInput,
        *,
        session_id: str | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> RunOutput:
        return await self._agent.run(
            user_input,
            session_id=session_id,
            abort_signal=abort_signal,
        )

    async def derive_child_for_scheduler(
        self,
        *,
        child_id: str,
        overrides: ChildAgentOverrides,
    ) -> "SchedulerAgentPort":
        child = await self._agent.derive_child(
            child_id=child_id,
            instruction=overrides.instruction,
            system_prompt_override=overrides.system_prompt,
            exclude_tool_names=overrides.exclude_tool_names,
        )
        return AgentSchedulerPort(child)

    async def steer(self, message: str) -> bool:
        return await self._agent.steer(message)

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
