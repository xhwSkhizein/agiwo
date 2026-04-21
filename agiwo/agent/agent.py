"""Agent — the primary entry point for the Agiwo Agent SDK."""

import asyncio
import copy
import secrets
from asyncio import Task
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from agiwo.agent.nested.agent_tool import AgentTool
from agiwo.agent.models.config import AgentConfig, AgentOptions
from agiwo.agent.definition import (
    ResolvedChildDefinition,
    build_agent_hooks,
    resolve_agent_definition,
    resolve_child_definition,
)
from agiwo.agent.hooks import AgentHooks
from agiwo.agent.models.input import UserInput, UserMessage
from agiwo.agent.prompt import build_system_prompt
from agiwo.agent.models.run import RunOutput
from agiwo.agent.run_loop import execute_run
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.models.stream import AgentStreamItem
from agiwo.agent.storage.base import RunStepStorage
from agiwo.agent.storage.factory import create_run_step_storage
from agiwo.agent.trace_writer import AgentTraceCollector
from agiwo.skill.manager import get_global_skill_manager
from agiwo.tool.base import BaseTool
from agiwo.tool.manager import get_global_tool_manager
from agiwo.llm.base import Model
from agiwo.observability.base import BaseTraceStorage
from agiwo.observability.factory import create_trace_storage
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger
from agiwo.workspace import WorkspaceBootstrapper, WorkspaceDocumentStore


def _generate_default_id(name: str) -> str:
    suffix = secrets.token_hex(3)
    return f"{name}-{suffix}"


logger = get_logger(__name__)


class AgentExecutionHandle:
    """One live root execution owned by a SessionRuntime."""

    def __init__(
        self,
        *,
        run_id: str,
        session_id: str,
        session_runtime: SessionRuntime,
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

    async def steer(self, user_input: UserInput) -> bool:
        return await self._session_runtime.enqueue_steer(user_input)

    def cancel(self, reason: str | None = None) -> None:
        self._session_runtime.abort_signal.abort(reason or "Cancelled by caller")


class Agent:
    """Thin facade over the internal agent runtime."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        model: Model,
        tools: list[BaseTool] | None = None,
        hooks: AgentHooks | None = None,
        id: str | None = None,
    ) -> None:
        """Create an Agent.

        Args:
            config: Agent configuration (``allowed_tools`` / ``allowed_skills``
                    drive which builtin + skill tools are assembled).
            model: LLM model to use.
            tools: Extra / custom functional tools (e.g. AgentTool, user-
                   supplied BaseTool).  Subject to ``allowed_tools`` filtering.
            hooks: Optional agent hooks.
            id: Stable instance identifier.  Auto-generated if omitted.
        """
        self._config = copy.deepcopy(config)
        self._id = id or _generate_default_id(self._config.name)
        self._model = model
        self._config.allowed_skills = (
            get_global_skill_manager().validate_explicit_allowed_skills(
                self._config.allowed_skills
            )
        )
        resolved_definition = resolve_agent_definition(
            config=self._config,
            agent_id=self._id,
            hooks=hooks,
        )
        self._hooks = resolved_definition.hooks

        self._extra_tools: tuple[BaseTool, ...] = tuple(tools) if tools else ()
        self._system_tools: tuple[BaseTool, ...] = ()
        tool_manager = get_global_tool_manager()
        self._tools = tool_manager.get_tools(
            allowed_tools=self._config.allowed_tools,
            extra_tools=list(self._extra_tools) if self._extra_tools else None,
            allowed_skills=self._config.allowed_skills,
        )

        self._workspace = resolved_definition.workspace
        self._run_step_storage = create_run_step_storage(
            self._config.options.storage.run_step_storage
        )
        self._trace_storage = create_trace_storage(
            self._config.options.storage.trace_storage
        )
        self._active_executions: dict[str, tuple[Task[RunOutput], AbortSignal]] = {}
        self._closing = False
        self._closed = False
        self._close_lock = asyncio.Lock()

    # --- Properties ---

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
        return self._hooks

    @hooks.setter
    def hooks(self, hooks: AgentHooks | None) -> None:
        self._hooks = build_agent_hooks(self._config, hooks)

    @property
    def run_step_storage(self) -> RunStepStorage:
        return self._run_step_storage

    @property
    def trace_storage(self) -> BaseTraceStorage | None:
        return self._trace_storage

    @property
    def tools(self) -> tuple[BaseTool, ...]:
        return self._tools

    @property
    def extra_tools(self) -> tuple[BaseTool, ...]:
        """Extra / custom functional tools originally passed at construction time."""
        return self._extra_tools

    @property
    def system_tools(self) -> tuple[BaseTool, ...]:
        """System-level tools (e.g. scheduler runtime tools)."""
        return self._system_tools

    def _inject_system_tools(self, system_tools: list[BaseTool]) -> None:
        """Inject system-level tools and rebuild the resolved tool list.

        This is a scheduler-internal API used to inject runtime tools
        (e.g. ``SpawnAgentTool``, ``SleepAndWaitTool``) after construction.
        System tools bypass ``allowed_tools`` filtering.

        **Note**: This method is intended for scheduler use only. Do not call
        this method directly in application code unless you are implementing
        custom scheduler logic.
        """
        self._system_tools = tuple(system_tools)
        tool_manager = get_global_tool_manager()
        self._tools = tool_manager.get_tools(
            allowed_tools=self._config.allowed_tools,
            extra_tools=list(self._extra_tools) if self._extra_tools else None,
            allowed_skills=self._config.allowed_skills,
            system_tools=system_tools,
        )

    async def get_effective_system_prompt(self) -> str:
        return await self._build_system_prompt(self._config.system_prompt)

    async def _build_system_prompt(self, base_prompt: str) -> str:
        return await build_system_prompt(
            base_prompt=base_prompt,
            workspace=self._workspace,
            tools=list(self._tools),
            allowed_skills=self._config.allowed_skills,
            bootstrapper=WorkspaceBootstrapper(),
            document_store=WorkspaceDocumentStore(),
        )

    def as_tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        max_depth: int = 5,
    ) -> BaseTool:
        return AgentTool(
            self,
            name=name,
            description=description,
            max_depth=max_depth,
        )

    # --- Child agent ---
    async def run_child(
        self,
        user_input: UserInput,
        *,
        session_runtime: SessionRuntime,
        parent_run_id: str,
        parent_depth: int,
        parent_user_id: str | None,
        parent_timeout_at: float | None,
        parent_metadata: dict[str, Any],
        instruction: str | None = None,
        system_prompt_override: str | None = None,
        child_allowed_tools: list[str] | None = None,
        child_allowed_skills: list[str] | None = None,
        metadata_overrides: dict[str, Any] | None = None,
        metadata_updates: dict | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> RunOutput:
        resolved_child: ResolvedChildDefinition = resolve_child_definition(
            parent_config=self._config,
            parent_extra_tools=self._extra_tools,
            parent_agent_id=self._id,
            instruction=instruction,
            system_prompt_override=system_prompt_override,
            child_allowed_tools=child_allowed_tools,
            child_allowed_skills=child_allowed_skills,
        )
        context = RunContext(
            session_runtime=session_runtime,
            run_id=str(uuid4()),
            agent_id=self._id,
            agent_name=self.name,
            user_id=parent_user_id,
            depth=parent_depth + 1,
            parent_run_id=parent_run_id,
            timeout_at=parent_timeout_at,
            metadata=dict(parent_metadata),
        )
        combined_metadata = dict(metadata_overrides or {})
        if metadata_updates:
            combined_metadata.update(metadata_updates)
        if combined_metadata:
            context.metadata.update(combined_metadata)
        child_abort_signal = abort_signal or session_runtime.abort_signal

        return await execute_run(
            user_input,
            context=context,
            model=self._model,
            system_prompt=await build_system_prompt(
                base_prompt=resolved_child.config.system_prompt,
                workspace=self._workspace,
                tools=resolved_child.extra_tools,
                allowed_skills=resolved_child.config.allowed_skills,
                bootstrapper=WorkspaceBootstrapper(),
                document_store=WorkspaceDocumentStore(),
            ),
            tools=resolved_child.extra_tools,
            hooks=build_agent_hooks(self._config, self._hooks),
            options=resolved_child.config.options.model_copy(deep=True),
            abort_signal=child_abort_signal,
            root_path=resolved_child.config.options.get_effective_root_path(),
        )

    async def create_child_agent(
        self,
        *,
        child_id: str,
        instruction: str | None = None,
        system_prompt_override: str | None = None,
        child_allowed_tools: list[str] | None = None,
        child_allowed_skills: list[str] | None = None,
        extra_tools: list[BaseTool] | None = None,
        inherit_all_extra_tools: bool = False,
        system_tools: list[BaseTool] | None = None,
    ) -> "Agent":
        """Create a child Agent with inherited configuration.

        Parent's extra tools are inherited automatically (minus self-referencing
        AgentTool).  The *extra_tools* parameter adds caller-provided tools on
        top.

        When *inherit_all_extra_tools* is ``True`` (fork mode), the exclusion
        filter is skipped so that the child receives an identical tool set for
        LLM KV cache reuse.

        *system_tools* are injected unconditionally and not subject to
        ``allowed_tools`` filtering.
        """
        resolved_child: ResolvedChildDefinition = resolve_child_definition(
            parent_config=self._config,
            parent_extra_tools=self._extra_tools,
            parent_agent_id=self._id,
            instruction=instruction,
            system_prompt_override=system_prompt_override,
            child_allowed_tools=child_allowed_tools,
            child_allowed_skills=child_allowed_skills,
            extra_tools=extra_tools,
            inherit_all_extra_tools=inherit_all_extra_tools,
        )

        child = self.__class__(
            resolved_child.config,
            id=child_id,
            model=self.model,
            tools=resolved_child.extra_tools or None,
            hooks=build_agent_hooks(self._config, self._hooks),
        )
        if system_tools:
            child._inject_system_tools(system_tools)
        return child

    # --- Execution ---

    def start(
        self,
        user_input: UserInput,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> AgentExecutionHandle:
        self._ensure_open()
        resolved_session_id = session_id or str(uuid4())
        resolved_abort_signal = abort_signal or AbortSignal()
        trace_runtime = self._start_trace_runtime(
            session_id=resolved_session_id,
            user_id=user_id,
            user_input=user_input,
        )
        session_runtime = SessionRuntime(
            session_id=resolved_session_id,
            run_step_storage=self._run_step_storage,
            trace_runtime=trace_runtime,
            abort_signal=resolved_abort_signal,
        )
        context = RunContext(
            session_runtime=session_runtime,
            run_id=str(uuid4()),
            agent_id=self._id,
            agent_name=self.name,
            user_id=user_id,
            metadata=dict(metadata or {}),
        )
        task = asyncio.create_task(
            self._execute_root(
                user_input,
                context=context,
                abort_signal=resolved_abort_signal,
            )
        )
        handle = AgentExecutionHandle(
            run_id=context.run_id,
            session_id=context.session_id,
            session_runtime=session_runtime,
            task=task,
        )
        self._register_execution(context.run_id, task, resolved_abort_signal)
        return handle

    async def _execute_root(
        self,
        user_input: UserInput,
        *,
        context: RunContext,
        abort_signal: AbortSignal,
    ) -> RunOutput:
        try:
            system_prompt = await self.get_effective_system_prompt()
            options = self._config.options.model_copy(deep=True)
        except Exception:
            await context.session_runtime.close()
            raise
        try:
            return await execute_run(
                user_input,
                context=context,
                model=self._model,
                system_prompt=system_prompt,
                tools=list(self._tools),
                hooks=self._hooks,
                options=options,
                abort_signal=abort_signal,
                root_path=options.get_effective_root_path(),
            )
        finally:
            await context.session_runtime.close()

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
        completed = False
        try:
            async for event in handle.stream():
                yield event
            await handle.wait()
            completed = True
        finally:
            if not completed:
                handle.cancel("run_stream consumer closed")
                try:
                    await handle.wait()
                except asyncio.CancelledError:
                    pass

    # --- Resource lifecycle ---

    def _ensure_open(self) -> None:
        if self._closing or self._closed:
            raise RuntimeError("agent_closed")

    def _register_execution(
        self,
        run_id: str,
        task: Task[RunOutput],
        abort_signal: AbortSignal,
    ) -> None:
        if task.done():
            return
        self._active_executions[run_id] = (task, abort_signal)
        task.add_done_callback(
            lambda _t, rid=run_id: self._active_executions.pop(rid, None)
        )

    def _start_trace_runtime(
        self,
        *,
        session_id: str,
        user_id: str | None,
        user_input: UserInput,
    ) -> AgentTraceCollector | None:
        if self._trace_storage is None:
            return None
        collector = AgentTraceCollector(store=self._trace_storage)
        collector.start(
            agent_id=self._id,
            session_id=session_id,
            user_id=user_id,
            input_query=UserMessage.from_value(user_input).extract_text(),
        )
        return collector

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closing = True
            active = list(self._active_executions.values())
            for _, signal in active:
                signal.abort("Agent closed")
            if active:
                await asyncio.gather(
                    *[task for task, _ in active],
                    return_exceptions=True,
                )
            storage_names = ["run_step_storage"]
            close_coros = [
                self._run_step_storage.close(),
            ]
            if self._trace_storage is not None:
                storage_names.append("trace_storage")
                close_coros.append(self._trace_storage.close())
            results = await asyncio.gather(*close_coros, return_exceptions=True)
            for name, result in zip(storage_names, results):
                if isinstance(result, BaseException):
                    logger.error(
                        "storage_close_failed", storage=name, error=str(result)
                    )
            self._active_executions.clear()
            self._closed = True
            self._closing = False


__all__ = ["Agent"]
