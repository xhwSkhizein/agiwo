import asyncio
import time
from collections.abc import AsyncIterator

from agiwo.agent.hooks import AgentHooks
from agiwo.agent.input import UserInput
from agiwo.agent.input_codec import normalize_to_message
from agiwo.agent.inner.context import AgentRunContext
from agiwo.agent.inner.event_emitter import EventEmitter
from agiwo.agent.inner.event_pump import EventPump
from agiwo.agent.inner.executor import AgentExecutor
from agiwo.agent.inner.run_payloads import build_run_completed_event_data
from agiwo.agent.inner.step_recorder import StepRecorder
from agiwo.agent.inner.storage_sink import StorageSink
from agiwo.agent.runtime import Run, RunOutput, RunStatus, StreamEvent
from agiwo.agent.trace import AgentTraceCollector
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class AgentRunner:
    """Single owner for root/child run orchestration."""

    def __init__(self, agent) -> None:
        self._agent = agent
        self._current_steering_queue: asyncio.Queue | None = None

    async def steer(self, message: str) -> bool:
        if not message.strip() or self._current_steering_queue is None:
            return False
        await self._current_steering_queue.put(message)
        return True

    async def run_root(
        self,
        user_input: UserInput,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> RunOutput:
        execution = self._start_root_execution(
            user_input,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            abort_signal=abort_signal,
        )
        try:
            result = await execution.task
            await execution.pump.wait()
            return result
        finally:
            await self._cleanup_root_execution(execution)

    async def run_root_stream(
        self,
        user_input: UserInput,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> AsyncIterator[StreamEvent]:
        execution = self._start_root_execution(
            user_input,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            abort_signal=abort_signal,
        )
        stream = execution.pump.subscribe()
        try:
            async for event in stream:
                yield event
            if execution.task.done() and not execution.task.cancelled():
                if exc := execution.task.exception():
                    raise exc
            await execution.task
            await execution.pump.wait()
        finally:
            await self._cleanup_root_execution(execution)

    async def run_child(
        self,
        user_input: UserInput,
        *,
        parent_context: AgentRunContext,
        abort_signal: AbortSignal | None = None,
    ) -> RunOutput:
        context = parent_context.new_child(
            agent_id=self._agent.id,
            agent_name=self._agent.name,
        )
        return await self._execute_workflow(
            user_input,
            context=context,
            close_channel_on_complete=False,
            abort_signal=abort_signal,
        )

    def _start_root_execution(
        self,
        user_input: UserInput,
        *,
        session_id: str | None,
        user_id: str | None,
        metadata: dict | None,
        abort_signal: AbortSignal | None,
    ):
        steering_queue: asyncio.Queue = asyncio.Queue()
        context = AgentRunContext.create_root(
            session_id=session_id or self._agent._new_session_id(),
            agent_id=self._agent.id,
            agent_name=self._agent.name,
            run_step_storage=self._agent.run_step_storage,
            user_id=user_id,
            metadata=metadata,
            steering_queue=steering_queue,
        )
        run = self._create_run(user_input, context)
        collector = None
        if self._agent.trace_storage is not None:
            collector = AgentTraceCollector(store=self._agent.trace_storage)
            collector.start(
                agent_id=self._agent.id,
                session_id=context.session_id,
                user_id=context.user_id,
                input_query=self._agent._extract_text(user_input),
            )
        pump = EventPump(
            channel=context.channel,
            storage_sink=StorageSink(self._agent.run_step_storage, run),
            hooks=self._agent.hooks,
            trace_collector=collector,
        )
        pump.start()
        self._current_steering_queue = steering_queue
        task = asyncio.create_task(
            self._execute_workflow(
                user_input,
                context=context,
                close_channel_on_complete=True,
                abort_signal=abort_signal,
            )
        )
        return _RootExecution(context=context, task=task, pump=pump)

    async def _execute_workflow(
        self,
        user_input: UserInput,
        *,
        context: AgentRunContext,
        close_channel_on_complete: bool,
        abort_signal: AbortSignal | None,
    ) -> RunOutput:
        system_prompt = await self._agent._runtime_state.prompt_runtime.get_system_prompt()
        emitter = EventEmitter(context)
        recorder = StepRecorder(
            context=context,
            emitter=emitter,
            hooks=self._agent.hooks,
            step_observers=list(self._agent._step_observers),
        )
        try:
            user_step = await recorder.create_user_step(user_input=user_input)
            await recorder.record(user_step, append_message=False)
            await emitter.emit_run_started({"query": user_input, "session_id": context.session_id})

            before_run_hook_result = None
            if self._agent.hooks.on_before_run:
                before_run_hook_result = await self._agent.hooks.on_before_run(
                    user_input,
                    context,
                )

            memories = []
            if self._agent.hooks.on_memory_retrieve and user_input:
                memories = await self._agent.hooks.on_memory_retrieve(user_input, context)

            user_message = normalize_to_message(user_input)
            executor = AgentExecutor(
                model=self._agent.model,
                tools=list(self._agent.tools),
                emitter=emitter,
                options=self._agent._config.options,
                hooks=self._agent.hooks,
                run_step_storage=self._agent.run_step_storage,
                session_storage=self._agent.session_storage,
                root_path=self._agent._config.options.get_effective_root_path(),
                step_observers=list(self._agent._step_observers),
                step_recorder=recorder,
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

            if self._agent.hooks.on_after_run:
                await self._agent.hooks.on_after_run(result, context)
            if self._agent.hooks.on_memory_write and result.response:
                await self._agent.hooks.on_memory_write(user_input, result, context)

            await emitter.emit_run_completed(build_run_completed_event_data(result))
            return result
        except Exception as error:
            await emitter.emit_run_failed(error)
            raise
        finally:
            if close_channel_on_complete:
                await context.channel.close()

    async def _cleanup_root_execution(self, execution: "_RootExecution") -> None:
        if self._current_steering_queue is execution.context.steering_queue:
            self._current_steering_queue = None
        if not execution.task.done():
            execution.task.cancel()
            try:
                async with asyncio.timeout(self._agent._config.options.stream_cleanup_timeout):
                    await execution.task
            except (asyncio.TimeoutError, asyncio.CancelledError):
                logger.warning("root_execution_cancelled_timeout", agent_id=self._agent.id)
        try:
            await execution.pump.wait()
        except asyncio.CancelledError:
            pass

    @staticmethod
    def _create_run(user_input: UserInput, context: AgentRunContext) -> Run:
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


class _RootExecution:
    def __init__(
        self,
        *,
        context: AgentRunContext,
        task: asyncio.Task[RunOutput],
        pump: EventPump,
    ) -> None:
        self.context = context
        self.task = task
        self.pump = pump


__all__ = ["AgentRunner"]
