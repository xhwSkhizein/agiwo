"""
Agent - the primary entry point for the Agiwo Agent SDK.
"""

import copy
import secrets
from typing import AsyncIterator
from typing import TYPE_CHECKING

from agiwo.agent.assembly import (
    build_agent_definition_runtime,
    build_agent_resource_owner,
)
from agiwo.agent.config import AgentConfig
from agiwo.agent.execution import AgentExecutionHandle, ChildAgentSpec
from agiwo.agent.hooks import AgentHooks
from agiwo.agent.inner.runner import AgentRunner
from agiwo.agent.input import UserInput
from agiwo.agent.options import AgentOptions
from agiwo.agent.runtime import AgentStreamItem, RunOutput
from agiwo.agent.runtime_tools import RuntimeToolLike
from agiwo.agent.scheduler_port import StepObserver
from agiwo.agent.storage.base import RunStepStorage
from agiwo.agent.storage.session import SessionStorage
from agiwo.agent.streaming import consume_execution_stream
from agiwo.llm.base import Model
from agiwo.observability.base import BaseTraceStorage
from agiwo.tool.base import BaseTool
from agiwo.tool.builtin.bash_tool import ensure_bash_tool_pair
from agiwo.utils.abort_signal import AbortSignal

if TYPE_CHECKING:
    from agiwo.agent.inner.context import AgentRunContext


def _generate_default_id(name: str) -> str:
    suffix = secrets.token_hex(3)
    return f"{name}-{suffix}"


class Agent:
    """Thin facade over the internal agent runtime."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        model: Model,
        tools: list[RuntimeToolLike] | None = None,
        hooks: AgentHooks | None = None,
        id: str | None = None,
    ) -> None:
        self._config = copy.deepcopy(config)
        self._id = id or _generate_default_id(self._config.name)
        self._model = model
        provided_tools = list(tools or [])
        base_tools = [tool for tool in provided_tools if isinstance(tool, BaseTool)]
        runtime_only_tools = [
            tool for tool in provided_tools if not isinstance(tool, BaseTool)
        ]
        normalized_tools: list[RuntimeToolLike] = [
            *ensure_bash_tool_pair(base_tools),
            *runtime_only_tools,
        ]
        self._definition_runtime = build_agent_definition_runtime(
            config=self._config,
            agent_id=self._id,
            provided_tools=normalized_tools,
            hooks=hooks,
        )
        self._resource_owner = build_agent_resource_owner(
            config=self._config,
        )
        self._step_observers: list[StepObserver] = []
        self._runner = AgentRunner()

    @property
    def config(self) -> AgentConfig:
        return copy.deepcopy(self._config)

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def description(self) -> str:
        return self._config.description

    @property
    def options(self) -> AgentOptions:
        return self._config.options.model_copy(deep=True)

    @property
    def model(self) -> Model:
        return self._model

    @property
    def hooks(self) -> AgentHooks:
        return self._definition_runtime.hooks

    @hooks.setter
    def hooks(self, hooks: AgentHooks | None) -> None:
        self._definition_runtime.hooks = hooks

    @property
    def run_step_storage(self) -> RunStepStorage:
        return self._resource_owner.run_step_storage

    @property
    def trace_storage(self) -> BaseTraceStorage | None:
        return self._resource_owner.trace_storage

    @property
    def session_storage(self) -> SessionStorage:
        return self._resource_owner.session_storage

    @property
    def tools(self) -> tuple[RuntimeToolLike, ...]:
        return self._definition_runtime.tools

    def install_runtime_tools(self, tools: list[RuntimeToolLike]) -> None:
        self._definition_runtime.install_runtime_tools(tools)

    def set_termination_summary_enabled(self, enabled: bool) -> None:
        self._config.options.enable_termination_summary = enabled

    def add_step_observer(self, observer: StepObserver) -> None:
        if observer not in self._step_observers:
            self._step_observers.append(observer)

    def remove_step_observer(self, observer: StepObserver) -> None:
        self._step_observers = [
            existing for existing in self._step_observers if existing != observer
        ]

    async def get_effective_system_prompt(self) -> str:
        return await self._definition_runtime.get_effective_system_prompt()

    def derive_child_spec(
        self,
        *,
        child_id: str,
        instruction: str | None = None,
        system_prompt_override: str | None = None,
        exclude_tool_names: set[str] | None = None,
        metadata_overrides: dict | None = None,
    ) -> ChildAgentSpec:
        return ChildAgentSpec(
            agent_id=child_id,
            agent_name=self.name,
            description=self.description,
            instruction=instruction,
            system_prompt_override=system_prompt_override,
            exclude_tool_names=frozenset(exclude_tool_names or ()),
            metadata_overrides=dict(metadata_overrides or {}),
        )

    async def run_child(
        self,
        user_input: UserInput,
        *,
        spec: ChildAgentSpec,
        parent_context: "AgentRunContext",
        metadata_updates: dict | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> RunOutput:
        definition = await self._definition_runtime.snapshot_child_definition(
            model=self._model,
            spec=spec,
        )
        combined_metadata = dict(spec.metadata_overrides)
        if metadata_updates:
            combined_metadata.update(metadata_updates)
        return await self._runner.run_child(
            user_input,
            definition=definition,
            parent_context=parent_context,
            step_observers=tuple(self._step_observers),
            metadata_updates=combined_metadata or None,
            abort_signal=abort_signal,
        )

    async def create_scheduler_child_agent(
        self,
        *,
        child_id: str,
        instruction: str | None = None,
        system_prompt: str | None = None,
        exclude_tool_names: set[str] | None = None,
    ) -> "Agent":
        clone_spec = self._definition_runtime.build_scheduler_child_clone(
            child_id=child_id,
            instruction=instruction,
            system_prompt_override=system_prompt,
            exclude_tool_names=exclude_tool_names,
        )
        return self.__class__(
            clone_spec.config,
            model=self.model,
            tools=list(clone_spec.tools),
            hooks=clone_spec.hooks,
            id=clone_spec.agent_id,
        )

    def start(
        self,
        user_input: UserInput,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> AgentExecutionHandle:
        self._resource_owner.ensure_open()
        handle, active_execution = self._runner.start_root(
            user_input,
            agent_id=self.id,
            agent_name=self.name,
            resource_owner=self._resource_owner,
            step_observers=tuple(self._step_observers),
            resolve_definition=lambda: (
                self._definition_runtime.snapshot_root_definition(model=self._model)
            ),
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            abort_signal=abort_signal,
        )
        self._resource_owner.register_execution(active_execution)
        return handle

    async def close(self) -> None:
        await self._resource_owner.close()

    async def run(
        self,
        user_input: UserInput,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> RunOutput:
        handle = self.start(
            user_input,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            abort_signal=abort_signal,
        )
        return await handle.wait()

    async def run_stream(
        self,
        user_input: UserInput,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> AsyncIterator[AgentStreamItem]:
        handle = self.start(
            user_input,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            abort_signal=abort_signal,
        )
        async for event in consume_execution_stream(
            handle,
            cancel_reason="run_stream consumer closed",
        ):
            yield event


__all__ = ["Agent"]
