"""
Agent - The primary entry point for the Agiwo Agent SDK.

Usage:
    agent = Agent(
        id="my-agent",
        description="A helpful assistant",
        model=DeepseekModel(id="deepseek-chat", name="deepseek-chat"),
        tools=[MyTool()],
        system_prompt="You are helpful.",
    )

    result = await agent.run("Hello!")
    async for event in agent.run_stream("Hello!"):
        print(event)
"""

import time
import asyncio
from typing import AsyncIterator
from uuid import uuid4

from agiwo.agent.schema import (
    UserInput,
    extract_text,
    normalize_input,
    to_message_content,
)
from agiwo.agent.execution_context import ExecutionContext, SessionSequenceCounter
from agiwo.agent.inner.event_emitter import EventEmitter
from agiwo.agent.inner.message_assembler import MessageAssembler
from agiwo.agent.inner.storage_sink import StorageSink
from agiwo.agent.stream_channel import StreamChannel
from agiwo.agent.inner.system_prompt_builder import DefaultSystemPromptBuilder
from agiwo.llm.base import Model
from agiwo.tool.base import BaseTool
from agiwo.tool.builtin import DEFAULT_TOOLS
from agiwo.agent.hooks import AgentHooks
from agiwo.agent.schema import Run, RunOutput, RunStatus, StreamEvent, StepRecord
from agiwo.agent.storage.base import RunStepStorage
from agiwo.agent.storage.factory import StorageFactory
from agiwo.observability.base import BaseTraceStorage
from agiwo.observability.collector import TraceCollector
from agiwo.utils.abort_signal import AbortSignal
from agiwo.agent.inner.executor import AgentExecutor
from agiwo.agent.options import AgentOptions
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class Agent:
    """
    Agent is the primary entry point for the Agiwo Agent SDK.

    Responsibilities:
    1. Configuration (Model, Tools, Prompts)
    2. Execution lifecycle (run / run_stream)
    3. Event streaming and observability
    4. Lifecycle hooks for extensibility

    Example:
        agent = Agent(
            id="assistant",
            description="A helpful assistant",
            model=my_model,
            tools=[calculator],
            system_prompt="You are helpful.",
            hooks=AgentHooks(on_after_tool_call=my_callback),
        )
        result = await agent.run("What is 2+2?")
    """

    def __init__(
        self,
        id: str,
        description: str,
        model: Model,
        tools: list[BaseTool] | None = None,
        system_prompt: str = "",
        options: AgentOptions | None = None,
        hooks: AgentHooks | None = None,
    ):
        self.id = id
        self.description = description
        self.model = model
        self.tools = tools if tools is not None else [cls() for cls in DEFAULT_TOOLS.values()]
        self.options = options or AgentOptions()
        self.hooks = hooks or AgentHooks()

        # Storage instances (created synchronously, connect lazily on first use)
        self.run_step_storage: RunStepStorage = StorageFactory.create_run_step_storage(
            self.options.run_step_storage
        )
        self.trace_storage: BaseTraceStorage | None = StorageFactory.create_trace_storage(
            self.options.trace_storage
        )

        # Initialize prompt builder
        self._prompt_builder = DefaultSystemPromptBuilder(
            base_prompt=system_prompt,
            agent_id=self.id,
            options=self.options,
        )
        self.system_prompt = self._prompt_builder.build()

    async def close(self) -> None:
        """Close agent and release all resources."""
        await self.run_step_storage.close()
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
        return ExecutionContext(
            run_id=str(uuid4()),
            session_id=resolved_session_id,
            channel=StreamChannel(),
            user_id=user_id,
            agent_id=self.id,
            sequence_counter=SessionSequenceCounter(initial_seq),
            metadata=metadata or {},
        )

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

        return asyncio.create_task(_wrapper())

    async def _execute_workflow(
        self,
        user_input: UserInput,
        context: ExecutionContext,
        abort_signal: AbortSignal | None,
    ) -> RunOutput:
        emitter = EventEmitter(context)

        # Load agent-specific history
        existing_steps = await self.run_step_storage.get_steps(
            session_id=context.session_id,
            agent_id=self.id,
        )

        # Record user step and include in history
        user_step = await self._record_user_step(user_input, context, emitter)
        existing_steps.append(user_step)

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

            # Assemble messages
            messages = MessageAssembler.assemble(
                self.system_prompt,
                existing_steps,
                memories,
                before_run_hook_result,
            )

            # Execute core loop
            executor = AgentExecutor(
                model=self.model,
                tools=self.tools or [],
                emitter=emitter,
                options=self.options,
                hooks=self.hooks,
            )
            result = await executor.execute(
                messages, context, abort_signal=abort_signal
            )

            # After-run hook
            if self.hooks.on_after_run:
                await self.hooks.on_after_run(result, context)

            # Memory write hook
            if self.hooks.on_memory_write and result.response:
                await self.hooks.on_memory_write(user_input, result, context)

            # Emit RUN_COMPLETED
            metrics = result.metrics
            data: dict = {
                "response": result.response or "",
                "metrics": {
                    "duration": metrics.duration_ms if metrics else 0,
                    "total_tokens": metrics.total_tokens if metrics else 0,
                    "input_tokens": metrics.input_tokens if metrics else 0,
                    "output_tokens": metrics.output_tokens if metrics else 0,
                    "tool_calls_count": metrics.tool_calls_count if metrics else 0,
                },
            }
            if result.termination_reason:
                data["termination_reason"] = result.termination_reason
            await emitter.emit_run_completed(data)
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
        content = to_message_content(normalize_input(user_input))
        user_step = StepRecord.user(
            context,
            sequence=seq,
            content=content,
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
            except Exception as e:
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
            except Exception as e:
                logger.error("stream_drain_task_cleanup_error", error=str(e))
                return
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error("stream_drain_task_failed", error=str(e), exc_info=True)
