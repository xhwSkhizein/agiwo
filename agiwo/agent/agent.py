"""
Agent - the primary entry point for the Agiwo Agent SDK.

Usage:
    agent = Agent(
        AgentConfig(
            name="my-agent",
            description="A helpful assistant",
            system_prompt="You are helpful.",
        ),
        model=DeepseekModel(id="deepseek-chat", name="deepseek-chat"),
        tools=[MyTool()],
    )

    result = await agent.run("Hello!")
    async for event in agent.run_stream("Hello!"):
        print(event)
"""

import asyncio
import copy
import secrets
import time
from typing import AsyncIterator
from uuid import uuid4

from agiwo.agent.assembly import (
    build_agent_runtime_state,
    build_effective_hooks,
    build_prompt_runtime,
)
from agiwo.agent.config import AgentConfig
from agiwo.agent.execution_context import ExecutionContext, SessionSequenceCounter
from agiwo.agent.hooks import AgentHooks
from agiwo.agent.input import UserInput
from agiwo.agent.input_codec import extract_text, normalize_to_message
from agiwo.agent.inner.event_emitter import EventEmitter
from agiwo.agent.inner.executor import AgentExecutor
from agiwo.agent.inner.run_payloads import build_run_completed_event_data
from agiwo.agent.inner.storage_sink import StorageSink
from agiwo.agent.options import AgentOptions
from agiwo.agent.prompt import AgentPromptRuntime
from agiwo.agent.runtime import Run, RunOutput, RunStatus, StepRecord, StreamEvent
from agiwo.agent.runtime_state import AgentRuntimeState
from agiwo.agent.runtime_tools import RuntimeToolLike
from agiwo.agent.scheduler_port import StepObserver
from agiwo.agent.storage.base import RunStepStorage
from agiwo.agent.storage.session import SessionStorage
from agiwo.agent.stream_channel import StreamChannel
from agiwo.agent.trace import AgentTraceCollector
from agiwo.llm.base import Model
from agiwo.observability.base import BaseTraceStorage
from agiwo.tool.base import BaseTool
from agiwo.tool.builtin.bash_tool import ensure_bash_tool_pair
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


def _generate_default_id(name: str) -> str:
    """Generate semantic default ID: name + short random hex suffix."""
    suffix = secrets.token_hex(3)
    return f"{name}-{suffix}"


class Agent:
    """
    Agent is the primary entry point for the Agiwo Agent SDK.

    Responsibilities:
    1. Hold pure agent configuration via AgentConfig.
    2. Hold reusable runtime dependencies (model, provided tools, hooks).
    3. Create and own agent-local runtime state (storage, prompt builder, skill manager).
    """

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
        self._provided_tools: list[RuntimeToolLike] = [
            *ensure_bash_tool_pair(base_tools),
            *runtime_only_tools,
        ]
        self._runtime_tools: list[RuntimeToolLike] = []
        self._runtime_state: AgentRuntimeState = build_agent_runtime_state(
            config=self._config,
            agent_id=self._id,
            provided_tools=self._provided_tools,
            hooks=hooks,
        )
        self._step_observers: list[StepObserver] = []
        self._current_steering_queue: asyncio.Queue | None = None

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
        return self._runtime_state.hooks

    @hooks.setter
    def hooks(self, hooks: AgentHooks | None) -> None:
        self._runtime_state.hooks = self._build_effective_hooks(hooks)

    @property
    def run_step_storage(self) -> RunStepStorage:
        return self._runtime_state.run_step_storage

    @property
    def trace_storage(self) -> BaseTraceStorage | None:
        return self._runtime_state.trace_storage

    @property
    def session_storage(self) -> SessionStorage:
        return self._runtime_state.session_storage

    @property
    def tools(self) -> tuple[RuntimeToolLike, ...]:
        return tuple(self._iter_effective_tools())

    def _iter_effective_tools(self) -> list[RuntimeToolLike]:
        return [
            *self._provided_tools,
            *self._runtime_state.sdk_tools,
            *self._runtime_tools,
        ]

    def _build_effective_hooks(self, hooks: AgentHooks | None) -> AgentHooks:
        return build_effective_hooks(
            config=self._config,
            agent_id=self._id,
            hooks=hooks,
        )

    def _create_prompt_builder(self, *, base_prompt: str) -> AgentPromptRuntime:
        return build_prompt_runtime(
            base_prompt=base_prompt,
            options=self._config.options,
            agent_name=self.name,
            agent_id=self.id,
            tools=list(self.tools),
            skill_manager=self._runtime_state.skill_manager,
        )

    def _refresh_prompt_builder(self) -> None:
        self._runtime_state.prompt_runtime = self._create_prompt_builder(
            base_prompt=self._config.system_prompt
        )

    def install_runtime_tools(self, tools: list[RuntimeToolLike]) -> None:
        existing_names = {tool.get_name() for tool in self._iter_effective_tools()}
        changed = False
        for tool in tools:
            tool_name = tool.get_name()
            if tool_name in existing_names:
                continue
            self._runtime_tools.append(tool)
            existing_names.add(tool_name)
            changed = True
        if changed:
            self._refresh_prompt_builder()

    def _remove_tools_by_name(self, names: set[str]) -> None:
        if not names:
            return
        before_counts = (
            len(self._provided_tools),
            len(self._runtime_state.sdk_tools),
            len(self._runtime_tools),
        )
        self._provided_tools = [
            tool for tool in self._provided_tools if tool.get_name() not in names
        ]
        self._runtime_state.sdk_tools = [
            tool for tool in self._runtime_state.sdk_tools if tool.get_name() not in names
        ]
        self._runtime_tools = [
            tool for tool in self._runtime_tools if tool.get_name() not in names
        ]
        after_counts = (
            len(self._provided_tools),
            len(self._runtime_state.sdk_tools),
            len(self._runtime_tools),
        )
        if after_counts != before_counts:
            self._refresh_prompt_builder()

    def set_termination_summary_enabled(self, enabled: bool) -> None:
        self._config.options.enable_termination_summary = enabled

    def add_step_observer(self, observer: StepObserver) -> None:
        if observer not in self._step_observers:
            self._step_observers.append(observer)

    def remove_step_observer(self, observer: StepObserver) -> None:
        self._step_observers = [
            existing for existing in self._step_observers if existing != observer
        ]

    async def _notify_step_observers(self, step: StepRecord) -> None:
        for observer in list(self._step_observers):
            await observer(step)

    async def get_effective_system_prompt(self) -> str:
        """Get the fully built system prompt (triggers build if needed)."""
        return await self._runtime_state.prompt_runtime.get_system_prompt()

    async def derive_child(
        self,
        *,
        child_id: str,
        instruction: str | None = None,
        system_prompt_override: str | None = None,
        exclude_tool_names: set[str] | None = None,
    ) -> "Agent":
        """Create a child agent inheriting current config and effective hooks."""
        child_config = copy.deepcopy(self._config)
        child_config.options.enable_termination_summary = True

        child_system_prompt = (
            system_prompt_override or await self.get_effective_system_prompt()
        )
        if instruction:
            child_system_prompt += (
                f"\n\n<task-instruction>\n{instruction}\n</task-instruction>"
            )
        child_config.system_prompt = child_system_prompt

        child = Agent(
            child_config,
            model=self._model,
            tools=list(self._provided_tools),
            hooks=copy.deepcopy(self.hooks),
            id=child_id,
        )
        child.install_runtime_tools(list(self._runtime_tools))
        if exclude_tool_names:
            child._remove_tools_by_name(exclude_tool_names)
        return child

    async def steer(self, message: str) -> bool:
        if not message.strip() or self._current_steering_queue is None:
            return False
        await self._current_steering_queue.put(message)
        return True

    def get_steering_queue(self) -> asyncio.Queue | None:
        """Backward-compatible access during scheduler migration."""
        return self._current_steering_queue

    async def close(self) -> None:
        """Close agent and release all owned resources."""
        await self.run_step_storage.close()
        await self.session_storage.close()
        if self.trace_storage is not None:
            await self.trace_storage.close()

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
        drain_task: asyncio.Task[None] | None = None
        try:
            task = self._start_execution_task(
                user_input,
                context,
                abort_signal,
                close_channel_on_complete=True,
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

        Concurrency contract:
            SDK does not enforce any session-level exclusivity for root streams.
            If an application requires "at most one active root run per session_id",
            that constraint must be implemented and managed at the application layer.
        """
        context = await self._create_context(session_id, user_id, metadata)
        task = self._start_execution_task(
            user_input,
            context,
            abort_signal,
            close_channel_on_complete=True,
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

    async def _create_context(
        self,
        session_id: str | None,
        user_id: str | None,
        metadata: dict | None,
    ) -> ExecutionContext:
        resolved_session_id = session_id or str(uuid4())
        all_steps = await self.run_step_storage.get_steps(session_id=resolved_session_id)
        initial_seq = max((step.sequence for step in all_steps), default=-1) + 1
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

    def _start_execution_task(
        self,
        user_input: UserInput,
        context: ExecutionContext,
        abort_signal: AbortSignal | None = None,
        close_channel_on_complete: bool = True,
    ) -> asyncio.Task[RunOutput]:
        async def _wrapper() -> RunOutput:
            try:
                return await self._execute_workflow(user_input, context, abort_signal)
            except Exception as error:
                logger.exception(
                    "agent_execution_crashed",
                    run_id=context.run_id,
                    agent_id=self.id,
                    error=str(error),
                )
                raise
            finally:
                if close_channel_on_complete:
                    await context.channel.close()
                self._current_steering_queue = None

        return asyncio.create_task(_wrapper())

    async def _execute_workflow(
        self,
        user_input: UserInput,
        context: ExecutionContext,
        abort_signal: AbortSignal | None,
    ) -> RunOutput:
        system_prompt = await self._runtime_state.prompt_runtime.get_system_prompt()
        emitter = EventEmitter(context)
        user_step = await self._record_user_step(user_input, context, emitter)

        await emitter.emit_run_started(
            {"query": user_input, "session_id": context.session_id}
        )

        try:
            before_run_hook_result = None
            if self.hooks.on_before_run:
                before_run_hook_result = await self.hooks.on_before_run(
                    user_input, context
                )

            memories = []
            if self.hooks.on_memory_retrieve and user_input:
                memories = await self.hooks.on_memory_retrieve(user_input, context)

            user_message = normalize_to_message(user_input)
            executor = AgentExecutor(
                model=self._model,
                tools=list(self.tools),
                emitter=emitter,
                options=self._config.options,
                hooks=self.hooks,
                run_step_storage=self.run_step_storage,
                session_storage=self.session_storage,
                root_path=self._config.options.get_effective_root_path(),
                step_observers=list(self._step_observers),
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

            if self.hooks.on_after_run:
                await self.hooks.on_after_run(result, context)

            if self.hooks.on_memory_write and result.response:
                await self.hooks.on_memory_write(user_input, result, context)

            await emitter.emit_run_completed(build_run_completed_event_data(result))
            return result
        except Exception as error:
            await emitter.emit_run_failed(error)
            raise

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
        await self._notify_step_observers(user_step)
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

    def _should_trace(self, context: ExecutionContext) -> bool:
        """Determine if tracing should be enabled for this execution."""
        if self.trace_storage is None:
            return False
        if context.parent_run_id is None and context.depth == 0:
            return True
        return context.trace_id is not None

    def _build_stream_pipeline(
        self,
        context: ExecutionContext,
        run: Run,
        user_input: str,
    ) -> AsyncIterator[StreamEvent]:
        stream = context.channel.read()
        stream = StorageSink(self.run_step_storage, run).wrap_stream(stream)

        if self._should_trace(context) and self.trace_storage is not None:
            collector = AgentTraceCollector(store=self.trace_storage)
            stream = collector.wrap_stream(
                stream,
                agent_id=self.id,
                session_id=context.session_id,
                user_id=context.user_id,
                input_query=user_input,
            )
        return stream

    async def _cleanup_task(self, task: asyncio.Task[RunOutput]) -> None:
        if not task.done():
            task.cancel()
            try:
                async with asyncio.timeout(self._config.options.stream_cleanup_timeout):
                    await task
            except (asyncio.TimeoutError, asyncio.CancelledError):
                logger.warning("stream_task_cancelled_timeout", agent_id=self.id)
            except Exception as error:  # noqa: BLE001 - cleanup boundary
                logger.error("stream_task_cleanup_error", error=str(error))

        if task.done() and not task.cancelled():
            if exc := task.exception():
                raise exc

    async def _drain_stream(self, stream: AsyncIterator[StreamEvent]) -> None:
        async for event in stream:
            if self.hooks.on_event:
                await self.hooks.on_event(event)

    async def _cleanup_stream_drain_task(
        self,
        task: asyncio.Task[None] | None,
    ) -> None:
        if task is None:
            return
        try:
            async with asyncio.timeout(self._config.options.stream_cleanup_timeout):
                await task
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                return
            except Exception as error:  # noqa: BLE001 - cleanup boundary
                logger.error("stream_drain_task_cleanup_error", error=str(error))
                return
        except asyncio.CancelledError:
            return
        except Exception as error:  # noqa: BLE001 - cleanup boundary
            logger.error("stream_drain_task_failed", error=str(error), exc_info=True)
