"""
Agent - The primary entry point for the Agiwo Agent SDK.

Usage:
    agent = Agent(
        name="my-agent",
        description="A helpful assistant",
        model=DeepseekModel(id="deepseek-chat", name="deepseek-chat"),
        tools=[MyTool()],
        system_prompt="You are helpful.",
    )

    result = await agent.run("Hello!")
    async for event in agent.run_stream("Hello!"):
        print(event)
"""

import asyncio
import copy
import time
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

import secrets

from agiwo.agent.execution_context import ExecutionContext, SessionSequenceCounter
from agiwo.agent.input import UserInput
from agiwo.agent.input_codec import extract_text, normalize_to_message
from agiwo.agent.inner.event_emitter import EventEmitter
from agiwo.agent.inner.run_payloads import build_run_completed_event_data
from agiwo.agent.inner.storage_sink import StorageSink
from agiwo.agent.inner.system_prompt_builder import DefaultSystemPromptBuilder
from agiwo.agent.memory_hooks import DefaultMemoryHook
from agiwo.agent.hooks import AgentHooks
from agiwo.agent.runtime import Run, RunOutput, RunStatus, StepRecord, StreamEvent
from agiwo.agent.storage.base import RunStepStorage
from agiwo.agent.storage.factory import StorageFactory
from agiwo.agent.storage.session import (
    InMemorySessionStorage,
    SessionStorage,
    SQLiteSessionStorage,
)
from agiwo.agent.stream_channel import StreamChannel
from agiwo.config.settings import settings
from agiwo.llm.base import Model
from agiwo.observability.base import BaseTraceStorage
from agiwo.observability.collector import TraceCollector
from agiwo.agent.inner.executor import AgentExecutor
from agiwo.agent.options import AgentOptions
from agiwo.skill.manager import SkillManager
from agiwo.tool.base import BaseTool
from agiwo.tool.builtin.bash_tool import ensure_bash_tool_pair
from agiwo.tool.builtin.registry import DEFAULT_TOOLS
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


def _resolve_skills_dirs(options: AgentOptions) -> list[Path]:
    """Resolve skill directories from options."""
    return options.get_configured_skills_dirs()


def _create_skill_manager(options: AgentOptions, agent_name: str) -> SkillManager:
    """Create SkillManager from options configuration."""
    skills_dirs = _resolve_skills_dirs(options)
    manager = SkillManager(skills_dirs=skills_dirs)
    logger.info("skill_manager_created", agent_name=agent_name, skills_dirs=[str(d) for d in skills_dirs])
    return manager


def create_agent(
    name: str,
    description: str,
    model: Model,
    *,
    id: str | None = None,
    tools: list[BaseTool] | None = None,
    system_prompt: str = "",
    options: AgentOptions | None = None,
    hooks: AgentHooks | None = None,
) -> "Agent":
    """Create an Agent with full assembly: tool merging, hook injection, storage, prompt builder.

    This is the standard way to create an Agent. Use Agent() directly only when
    you have pre-assembled components (e.g. derive_child).

    Example:
        agent = create_agent(
            name="assistant",
            description="A helpful assistant",
            model=my_model,
            tools=[calculator],
            system_prompt="You are helpful.",
            hooks=AgentHooks(on_after_tool_call=my_callback),
        )
        result = await agent.run("What is 2+2?")
    """
    resolved_id = id or _generate_default_id(name)
    resolved_options = options or AgentOptions()
    resolved_hooks = hooks or AgentHooks()

    # Merge user tools with DEFAULT_TOOLS (user takes priority on name collision)
    resolved_tools = ensure_bash_tool_pair(tools or [])
    user_tool_names = {t.get_name() for t in resolved_tools}
    default_tools = [cls() for tool_name, cls in DEFAULT_TOOLS.items() if tool_name not in user_tool_names]
    resolved_tools = resolved_tools + default_tools

    # Inject default memory retrieve hook if not provided
    if resolved_hooks.on_memory_retrieve is None:
        memory_hook = DefaultMemoryHook(
            embedding_provider="auto",
            top_k=5,
        )
        resolved_hooks.on_memory_retrieve = memory_hook.retrieve_memories
        logger.debug("default_memory_hook_injected", agent_id=resolved_id)

    # Create skill manager and add skill tool if enabled
    skill_manager: SkillManager | None = None
    if resolved_options.enable_skill:
        skill_manager = _create_skill_manager(resolved_options, name)
        skill_tool = skill_manager.get_skill_tool()
        resolved_tools.append(skill_tool)
        logger.debug("skill_tool_added", agent_id=resolved_id, tool_name=skill_tool.get_name())

    # Storage instances (created synchronously, connect lazily on first use)
    run_step_storage = StorageFactory.create_run_step_storage(
        resolved_options.run_step_storage
    )
    trace_storage = StorageFactory.create_trace_storage(
        resolved_options.trace_storage
    )
    session_storage = _create_session_storage(resolved_options)

    # Prompt builder (lazy build on first execution)
    prompt_builder = DefaultSystemPromptBuilder(
        base_prompt=system_prompt,
        agent_name=name,
        agent_id=resolved_id,
        options=resolved_options,
        tools=resolved_tools,
        skill_manager=skill_manager,
    )

    return Agent(
        name=name,
        description=description,
        model=model,
        id=resolved_id,
        tools=resolved_tools,
        options=resolved_options,
        hooks=resolved_hooks,
        run_step_storage=run_step_storage,
        trace_storage=trace_storage,
        session_storage=session_storage,
        prompt_builder=prompt_builder,
        skill_manager=skill_manager,
    )


def _generate_default_id(name: str) -> str:
    """Generate semantic default ID: name + short random hex suffix."""
    suffix = secrets.token_hex(3)
    return f"{name}-{suffix}"


def _create_session_storage(options: AgentOptions) -> SessionStorage:
    """Create SessionStorage based on RunStepStorage configuration."""
    storage_config = options.run_step_storage
    if storage_config.storage_type == "sqlite":
        db_path = storage_config.config.get("db_path", "agiwo.db")
        resolved_path = settings.resolve_path(db_path)
        return SQLiteSessionStorage(str(resolved_path) if resolved_path else db_path)
    return InMemorySessionStorage()


class Agent:
    """
    Agent is the primary entry point for the Agiwo Agent SDK.

    Responsibilities:
    1. Configuration (Model, Tools, Prompts)
    2. Execution lifecycle (run / run_stream)
    3. Event streaming and observability
    4. Lifecycle hooks for extensibility

    Use create_agent() to construct an Agent with full assembly.
    Use Agent() directly only when you have pre-assembled components.
    """

    def __init__(
        self,
        name: str,
        description: str,
        model: Model,
        *,
        id: str,
        tools: list[BaseTool],
        options: AgentOptions,
        hooks: AgentHooks,
        run_step_storage: RunStepStorage,
        trace_storage: BaseTraceStorage | None,
        session_storage: SessionStorage,
        prompt_builder: DefaultSystemPromptBuilder,
        skill_manager: SkillManager | None = None,
    ):
        self.name = name
        self.id = id
        self.description = description
        self.model = model
        self.tools = tools
        self.options = options
        self.hooks = hooks
        self.run_step_storage = run_step_storage
        self.trace_storage = trace_storage
        self.session_storage = session_storage
        self._prompt_builder = prompt_builder
        self._skill_manager = skill_manager
        self._system_prompt: str | None = None

    async def get_effective_system_prompt(self) -> str:
        """Get the fully built system prompt (triggers build if needed).

        Used by Scheduler to inherit parent's complete system configuration.
        """
        if self._system_prompt is not None:
            return self._system_prompt
        return await self._prompt_builder.get_system_prompt()

    async def derive_child(
        self,
        *,
        child_id: str,
        instruction: str | None = None,
        system_prompt_override: str | None = None,
        exclude_tool_names: set[str] | None = None,
    ) -> "Agent":
        """Create a child agent inheriting the current agent's effective configuration."""
        child_options = copy.deepcopy(self.options) if self.options else AgentOptions()
        child_options.enable_termination_summary = True

        system_prompt = system_prompt_override or await self.get_effective_system_prompt()
        if instruction:
            system_prompt += (
                f"\n\n<task-instruction>\n{instruction}\n</task-instruction>"
            )

        tools = list(self.tools)
        if exclude_tool_names:
            tools = [tool for tool in tools if tool.get_name() not in exclude_tool_names]

        child_prompt_builder = DefaultSystemPromptBuilder(
            base_prompt=system_prompt,
            agent_name=self.name,
            agent_id=child_id,
            options=child_options,
            tools=tools,
            skill_manager=self._skill_manager,
        )

        return Agent(
            name=self.name,
            description=self.description,
            model=self.model,
            id=child_id,
            tools=tools,
            options=child_options,
            hooks=copy.deepcopy(self.hooks),
            run_step_storage=self.run_step_storage,
            trace_storage=self.trace_storage,
            session_storage=self.session_storage,
            prompt_builder=child_prompt_builder,
            skill_manager=self._skill_manager,
        )

    @property
    def system_prompt(self) -> str | None:
        """Get the built system prompt (available after first execution)."""
        return self._system_prompt


    async def close(self) -> None:
        """Close agent and release all resources."""
        await self.run_step_storage.close()
        await self.session_storage.close()
        if self.trace_storage is not None:
            await self.trace_storage.close()

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    async def run(
        self,
        user_input: UserInput,
        *,
        context: ExecutionContext | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> RunOutput:
        """
        Execute agent (blocks until completion).

        Args:
            user_input: User message — str for text-only, list[ContentPart] for multi-modal.
            context: Existing ExecutionContext for nested agent calls. When provided,
                the agent shares the parent's StreamChannel and does not create a new one.
            session_id: Session ID for conversation continuity. Auto-generated if None.
                Ignored when context is provided.
            user_id: Optional user identifier. Ignored when context is provided.
            metadata: Optional metadata dict. Ignored when context is provided.
            abort_signal: Optional signal for graceful cancellation.

        Returns:
            RunOutput with response, metrics, and termination reason.

        Concurrency contract:
            SDK does not enforce any session-level exclusivity for root runs.
            If an application requires "at most one active root run per session_id",
            that constraint must be implemented and managed at the application layer.
        """
        is_root = context is None
        if is_root:
            context = await self._create_context(session_id, user_id, metadata)

        if not is_root:
            return await self._execute_workflow(user_input, context, abort_signal)

        run = self._create_run(user_input, context)
        try:
            task = self._start_execution_task(
                user_input, context, abort_signal, close_channel_on_complete=True
            )
            text_query = extract_text(user_input)
            stream = self._build_stream_pipeline(context, run, text_query)
            drain_task = asyncio.create_task(self._drain_stream(stream))

            return await task
        finally:
            await self._cleanup_stream_drain_task(drain_task)

    async def run_stream(
        self,
        user_input: UserInput,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Execute agent and yield events in real-time.

        Args:
            user_input: User message — str for text-only, list[ContentPart] for multi-modal.
            session_id: Session ID for conversation continuity. Auto-generated if None.
            user_id: Optional user identifier.
            metadata: Optional metadata dict passed to execution context.
            abort_signal: Optional signal for graceful cancellation.

        Yields:
            StreamEvent objects for each execution step and delta.

        Concurrency contract:
            SDK does not enforce any session-level exclusivity for root streams.
            If an application requires "at most one active root run per session_id",
            that constraint must be implemented and managed at the application layer.
        """
        context = await self._create_context(session_id, user_id, metadata)

        task = self._start_execution_task(
            user_input, context, abort_signal, close_channel_on_complete=True
        )

        run = self._create_run(user_input, context)
        text_query = extract_text(user_input)
        stream = self._build_stream_pipeline(context, run, text_query)

        try:
            async for event in stream:
                if self.hooks.on_event:
                    await self.hooks.on_event(event)
                if task.done() and not task.cancelled():
                    if exc := task.exception():
                        raise exc
                yield event
        finally:
            await self._cleanup_task(task)

    # ──────────────────────────────────────────────────────────────────────
    # Core Execution Workflow
    # ──────────────────────────────────────────────────────────────────────

    async def _create_context(
        self,
        session_id: str | None,
        user_id: str | None,
        metadata: dict | None,
    ) -> ExecutionContext:
        resolved_session_id = session_id or str(uuid4())
        all_steps = await self.run_step_storage.get_steps(
            session_id=resolved_session_id,
        )
        initial_seq = max((s.sequence for s in all_steps), default=-1) + 1
        steering_queue: asyncio.Queue = asyncio.Queue()
        context = ExecutionContext(
            run_id=str(uuid4()),
            session_id=resolved_session_id,
            channel=StreamChannel(),
            user_id=user_id,
            agent_id=self.id,
            agent_name=self.name,
            sequence_counter=SessionSequenceCounter(initial_seq),
            metadata=metadata or {},
            steering_queue=steering_queue,
        )
        self._current_steering_queue = steering_queue
        return context

    def get_steering_queue(self) -> asyncio.Queue | None:
        """Return the steering queue for the current execution, if active.

        External callers (e.g. Scheduler) can push messages here to be injected
        into the next LLM call while the agent is RUNNING.
        """
        return getattr(self, "_current_steering_queue", None)

    def _start_execution_task(
        self,
        user_input: UserInput,
        context: ExecutionContext,
        abort_signal: AbortSignal | None = None,
        close_channel_on_complete: bool = True,
    ) -> asyncio.Task[RunOutput]:
        async def _wrapper():
            try:
                return await self._execute_workflow(user_input, context, abort_signal)
            except Exception as e:
                logger.exception(
                    "agent_execution_crashed",
                    run_id=context.run_id,
                    agent_id=self.id,
                    error=str(e),
                )
                raise
            finally:
                if close_channel_on_complete:
                    await context.channel.close()
                self._current_steering_queue = None  # 清理过期 queue

        return asyncio.create_task(_wrapper())

    async def _execute_workflow(
        self,
        user_input: UserInput,
        context: ExecutionContext,
        abort_signal: AbortSignal | None,
    ) -> RunOutput:
        # Get system prompt (auto-refreshes if SOUL.md or skills changed)
        system_prompt = await self._prompt_builder.get_system_prompt()

        emitter = EventEmitter(context)

        # Record user step (Executor handles history loading internally)
        user_step = await self._record_user_step(user_input, context, emitter)

        # Emit RUN_STARTED
        await emitter.emit_run_started(
            {"query": user_input, "session_id": context.session_id}
        )

        try:
            # Before-run hook
            before_run_hook_result = None
            if self.hooks.on_before_run:
                before_run_hook_result = await self.hooks.on_before_run(
                    user_input, context
                )

            # Retrieve memories
            memories = []
            if self.hooks.on_memory_retrieve and user_input:
                memories = await self.hooks.on_memory_retrieve(user_input, context)

            # Execute core loop (MessageAssembler moved inside Executor)
            user_message = normalize_to_message(user_input)
            executor = AgentExecutor(
                model=self.model,
                tools=self.tools or [],
                emitter=emitter,
                options=self.options,
                hooks=self.hooks,
                run_step_storage=self.run_step_storage,
                session_storage=self.session_storage,
                root_path=self.options.get_effective_root_path(),
            )
            result = await executor.execute(
                system_prompt=system_prompt,
                user_step=user_step,
                context=context,
                memories=memories,
                before_run_hook_result=before_run_hook_result,
                channel_context=user_message.context,
                abort_signal=abort_signal,
            )

            # After-run hook
            if self.hooks.on_after_run:
                await self.hooks.on_after_run(result, context)

            # Memory write hook
            if self.hooks.on_memory_write and result.response:
                await self.hooks.on_memory_write(user_input, result, context)

            await emitter.emit_run_completed(build_run_completed_event_data(result))
            return result

        except Exception as e:
            await emitter.emit_run_failed(e)
            raise

    # ──────────────────────────────────────────────────────────────────────
    # Helpers: Context & Data
    # ──────────────────────────────────────────────────────────────────────

    async def _record_user_step(
        self,
        user_input: UserInput,
        context: ExecutionContext,
        emitter: EventEmitter,
    ) -> StepRecord:
        seq = await context.sequence_counter.next()
        user_step = StepRecord.user(
            context,
            sequence=seq,
            user_input=user_input,
            agent_id=self.id,
        )
        await emitter.emit_step_completed(user_step)
        if self.hooks.on_step:
            await self.hooks.on_step(user_step)
        return user_step

    def _create_run(self, user_input: UserInput, context: ExecutionContext) -> Run:
        run = Run(
            id=context.run_id,
            agent_id=context.agent_id,
            session_id=context.session_id,
            user_input=user_input,
            status=RunStatus.RUNNING,
            parent_run_id=context.parent_run_id,
        )
        run.metrics.start_at = time.time()
        return run

    # ──────────────────────────────────────────────────────────────────────
    # Helpers: Observability & Stream
    # ──────────────────────────────────────────────────────────────────────

    def _should_trace(self, context: ExecutionContext) -> bool:
        """
        Determine if tracing should be enabled for this execution.

        Tracing is enabled when:
        1. A trace_storage is configured
        2. Either this is a root agent (parent_run_id is None, depth is 0)
           OR this is a child agent with inherited trace context (trace_id is set)
        """
        if self.trace_storage is None:
            return False

        # Root agent - always trace if store is configured
        if context.parent_run_id is None and context.depth == 0:
            return True

        # Child agent - trace if inherited trace context from parent
        if context.trace_id is not None:
            return True

        return False

    def _build_stream_pipeline(
        self, context: ExecutionContext, run: Run, user_input: str
    ) -> AsyncIterator[StreamEvent]:
        stream = context.channel.read()

        # Layer 1: Storage persistence
        stream = StorageSink(self.run_step_storage, run).wrap_stream(stream)

        # Layer 2: Trace collection (optional)
        if self._should_trace(context) and self.trace_storage:
            collector = TraceCollector(store=self.trace_storage)
            stream = collector.wrap_stream(
                stream,
                agent_id=self.id,
                session_id=context.session_id,
                user_id=context.user_id,
                input_query=user_input,
            )

        return stream

    async def _cleanup_task(self, task: asyncio.Task) -> None:
        if not task.done():
            task.cancel()
            try:
                async with asyncio.timeout(self.options.stream_cleanup_timeout):
                    await task
            except (asyncio.TimeoutError, asyncio.CancelledError):
                logger.warning("stream_task_cancelled_timeout", agent_id=self.id)
            except Exception as e:  # noqa: BLE001 - cleanup boundary
                logger.error("stream_task_cleanup_error", error=str(e))

        if task.done() and not task.cancelled():
            if exc := task.exception():
                raise exc

    async def _drain_stream(self, stream: AsyncIterator[StreamEvent]) -> None:
        async for event in stream:
            if self.hooks.on_event:
                await self.hooks.on_event(event)

    async def _cleanup_stream_drain_task(self, task: asyncio.Task[None]) -> None:
        try:
            async with asyncio.timeout(self.options.stream_cleanup_timeout):
                await task
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                return
            except Exception as e:  # noqa: BLE001 - cleanup boundary
                logger.error("stream_drain_task_cleanup_error", error=str(e))
                return
        except asyncio.CancelledError:
            return
        except Exception as e:  # noqa: BLE001 - cleanup boundary
            logger.error("stream_drain_task_failed", error=str(e), exc_info=True)
