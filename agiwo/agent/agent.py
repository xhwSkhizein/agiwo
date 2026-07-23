"""Agent — the primary entry point for the Agiwo Agent SDK."""

import asyncio
import copy
import secrets
from asyncio import Task
from collections.abc import AsyncIterator, Callable
from uuid import uuid4

from agiwo.agent.nested.agent_tool import AgentTool
from agiwo.agent.nested.child_ops import AgentChildOps
from agiwo.agent.models.config import AgentConfig, AgentOptions
from agiwo.agent.definition import (
    build_agent_hooks,
    resolve_agent_definition,
)
from agiwo.agent.execution_handle import AgentExecutionHandle
from agiwo.agent.hooks import HookRegistration, HookRegistry
from agiwo.agent.introspect.tool import ReviewTrajectoryTool
from agiwo.agent.models.input import UserInput, UserMessage
from agiwo.agent.plan import UpdatePlanTool
from agiwo.agent.prompt import build_system_prompt
from agiwo.agent.models.execution import RunExecutionRequest
from agiwo.agent.models.run import RunIdentity, RunOutput
from agiwo.agent.run_loop import execute_run
from agiwo.agent.run_resume import AgentResumeOps
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.models.stream import AgentStreamItem
from agiwo.agent.storage.base import RunLogStorage
from agiwo.agent.storage.factory import create_run_log_storage
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


class Agent(AgentChildOps, AgentResumeOps):
    """Thin facade over the internal agent runtime."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        model: Model,
        tools: list[BaseTool] | None = None,
        hooks: HookRegistry | list[HookRegistration] | None = None,
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
        self._rebuild_tools(system_tools=self._owned_system_tools())

        self._workspace = resolved_definition.workspace
        self._run_log_storage = create_run_log_storage(
            self._config.options.storage.run_log_storage
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
    def options_snapshot(self) -> AgentOptions:
        """Zero-copy read of live options; do not mutate the returned object."""
        return self._config.options

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
    def hooks(self) -> HookRegistry:
        return self._hooks

    @hooks.setter
    def hooks(self, hooks: HookRegistry | list[HookRegistration] | None) -> None:
        self._hooks = build_agent_hooks(self._config, hooks)

    @property
    def run_log_storage(self) -> RunLogStorage:
        return self._run_log_storage

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

    def _owned_system_tools(self) -> list[BaseTool]:
        """System tools owned by Agent itself (not Scheduler)."""
        tools: list[BaseTool] = [UpdatePlanTool()]
        if self._config.options.enable_trajectory_review:
            tools.append(ReviewTrajectoryTool())
        return tools

    def _merge_system_tools(
        self, injected: list[BaseTool] | None = None
    ) -> list[BaseTool]:
        by_name: dict[str, BaseTool] = {
            tool.name: tool for tool in self._owned_system_tools()
        }
        for tool in injected or []:
            if tool.name in by_name and tool.name in {
                "update_plan",
                "review_trajectory",
            }:
                continue
            by_name[tool.name] = tool
        # Keep agent-owned tools first for stable schema ordering.
        owned_names = {tool.name for tool in self._owned_system_tools()}
        ordered: list[BaseTool] = [
            by_name[name] for name in sorted(owned_names) if name in by_name
        ]
        ordered.extend(
            tool for name, tool in by_name.items() if name not in owned_names
        )
        return ordered

    def _rebuild_tools(self, *, system_tools: list[BaseTool]) -> None:
        self._system_tools = tuple(system_tools)
        tool_manager = get_global_tool_manager()
        self._tools = tool_manager.get_tools(
            allowed_tools=self._config.allowed_tools,
            extra_tools=list(self._extra_tools) if self._extra_tools else None,
            allowed_skills=self._config.allowed_skills,
            system_tools=system_tools,
        )

    def _inject_system_tools(self, system_tools: list[BaseTool]) -> None:
        """Inject system-level tools and rebuild the resolved tool list.

        This is a scheduler-internal API used to inject runtime tools
        (e.g. ``SpawnChildAgentTool``, ``ForkChildAgentTool``,
        ``SleepAndWaitTool``) after construction. System tools bypass
        ``allowed_tools`` filtering. Agent-owned tools such as
        ``update_plan`` are always retained.

        **Note**: This method is intended for scheduler use only. Do not call
        this method directly in application code unless you are implementing
        custom scheduler logic.
        """
        self._rebuild_tools(system_tools=self._merge_system_tools(system_tools))

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
        """Start a root run from genuine user input.

        Rejects ``UserMessage(is_user_provided=False)``. Scheduler-owned wakes
        that inject system-attributed user-role turns must call
        ``start_prevalidated`` instead.
        """
        return self.start_prevalidated(
            UserMessage.require_user_provided(user_input),
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            abort_signal=abort_signal,
        )

    def start_prevalidated(
        self,
        user_input: UserInput | None,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
        abort_signal: AbortSignal | None = None,
        execution_request: RunExecutionRequest | None = None,
        active_worker_ids: Callable[[], frozenset[str]] | None = None,
    ) -> AgentExecutionHandle:
        """Start a root run without re-checking user-input provenance.

        Internal Scheduler contract: external Scheduler APIs already validated
        user input, and wake/fork paths may inject ``UserMessage.from_system()``.
        ``execution_request`` lets Scheduler dispatch supply a preallocated
        ``run_id``. ``user_input`` may be ``None`` when Session history already
        holds the user message (ADR 0048).
        ``active_worker_ids`` is an optional Loop gate dependency (ADR 0049).
        """
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
            run_log_storage=self._run_log_storage,
            trace_runtime=trace_runtime,
            abort_signal=resolved_abort_signal,
        )
        request = execution_request or RunExecutionRequest(run_id=str(uuid4()))
        context = RunContext(
            identity=RunIdentity(
                run_id=request.run_id,
                agent_id=self._id,
                agent_name=self.name,
                user_id=user_id,
                run_tree_role=request.run_tree_role,
                metadata=dict(metadata or {}),
            ),
            session_runtime=session_runtime,
        )
        task = asyncio.create_task(
            self._execute_root(
                user_input,
                context=context,
                abort_signal=resolved_abort_signal,
                active_worker_ids=active_worker_ids,
            )
        )
        handle = AgentExecutionHandle(
            run_id=context.run_id,
            session_id=context.session_id,
            session_runtime=session_runtime,
            task=task,
            context=context,
        )
        self._register_execution(context.run_id, task, resolved_abort_signal)
        return handle

    async def _execute_root(
        self,
        user_input: UserInput | None,
        *,
        context: RunContext,
        abort_signal: AbortSignal,
        active_worker_ids: Callable[[], frozenset[str]] | None = None,
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
                active_worker_ids=active_worker_ids,
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
        user_input: UserInput | None,
    ) -> AgentTraceCollector | None:
        if self._trace_storage is None:
            return None
        collector = AgentTraceCollector(store=self._trace_storage)
        input_query = (
            UserMessage.from_value(user_input).extract_text()
            if user_input is not None
            else ""
        )
        collector.start(
            agent_id=self._id,
            session_id=session_id,
            user_id=user_id,
            input_query=input_query,
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
            storage_names = ["run_log_storage"]
            close_coros = [
                self._run_log_storage.close(),
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


__all__ = ["Agent", "AgentExecutionHandle"]
