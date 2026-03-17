import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from uuid import uuid4

from agiwo.agent.execution import AgentExecutionHandle
from agiwo.agent.hooks import AgentHooks
from agiwo.agent.input import UserInput
from agiwo.agent.input_codec import extract_text, normalize_to_message
from agiwo.agent.inner.context import AgentRunContext
from agiwo.agent.inner.definition import ResolvedExecutionDefinition
from agiwo.agent.inner.executor import AgentExecutor
from agiwo.agent.inner.resource_owner import ActiveRootExecution, AgentResourceOwner
from agiwo.agent.inner.run_recorder import RunRecorder
from agiwo.agent.inner.session_runtime import AgentSessionRuntime
from agiwo.agent.runtime import Run, RunOutput, RunStatus
from agiwo.agent.scheduler_port import StepObserver
from agiwo.agent.trace import AgentTraceCollector
from agiwo.utils.abort_signal import AbortSignal


class AgentRunner:
    """Single owner for root/child run orchestration."""

    def start_root(
        self,
        user_input: UserInput,
        *,
        agent_id: str,
        agent_name: str,
        resource_owner: AgentResourceOwner,
        step_observers: Sequence[StepObserver],
        resolve_definition: Callable[[], Awaitable[ResolvedExecutionDefinition]],
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> tuple[AgentExecutionHandle, ActiveRootExecution]:
        resolved_session_id = session_id or str(uuid4())
        resolved_abort_signal = abort_signal or AbortSignal()
        trace_runtime = self._start_trace_runtime(
            resource_owner=resource_owner,
            agent_id=agent_id,
            session_id=resolved_session_id,
            user_id=user_id,
            user_input=user_input,
        )
        session_runtime = AgentSessionRuntime(
            session_id=resolved_session_id,
            run_step_storage=resource_owner.run_step_storage,
            session_storage=resource_owner.session_storage,
            trace_runtime=trace_runtime,
            trace_id=trace_runtime.trace_id if trace_runtime is not None else None,
            abort_signal=resolved_abort_signal,
        )
        context = AgentRunContext.create_root(
            session_runtime=session_runtime,
            agent_id=agent_id,
            agent_name=agent_name,
            user_id=user_id,
            metadata=metadata,
        )
        task = asyncio.create_task(
            self._execute_root(
                user_input,
                context=context,
                step_observers=step_observers,
                resolve_definition=resolve_definition,
                abort_signal=resolved_abort_signal,
            )
        )
        handle = AgentExecutionHandle(
            run_id=context.run_id,
            session_id=context.session_id,
            session_runtime=session_runtime,
            task=task,
        )
        execution = ActiveRootExecution(
            run_id=context.run_id,
            task=task,
            cancel_callback=lambda reason=None: resolved_abort_signal.abort(
                reason or "Cancelled by caller"
            ),
        )
        return handle, execution

    async def run_child(
        self,
        user_input: UserInput,
        *,
        definition: ResolvedExecutionDefinition,
        parent_context: AgentRunContext,
        step_observers: Sequence[StepObserver],
        metadata_updates: dict | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> RunOutput:
        context = parent_context.new_child(
            agent_id=definition.agent_id,
            agent_name=definition.agent_name,
        )
        if metadata_updates:
            context.metadata.update(metadata_updates)
        child_abort_signal = abort_signal or parent_context.session_runtime.abort_signal
        return await self._execute_workflow(
            user_input,
            context=context,
            definition=definition,
            step_observers=step_observers,
            close_session_on_complete=False,
            abort_signal=child_abort_signal,
        )

    async def _execute_root(
        self,
        user_input: UserInput,
        *,
        context: AgentRunContext,
        step_observers: Sequence[StepObserver],
        resolve_definition: Callable[[], Awaitable[ResolvedExecutionDefinition]],
        abort_signal: AbortSignal,
    ) -> RunOutput:
        definition = await resolve_definition()
        return await self._execute_workflow(
            user_input,
            context=context,
            definition=definition,
            step_observers=step_observers,
            close_session_on_complete=True,
            abort_signal=abort_signal,
        )

    async def _execute_workflow(
        self,
        user_input: UserInput,
        *,
        context: AgentRunContext,
        definition: ResolvedExecutionDefinition,
        step_observers: Sequence[StepObserver],
        close_session_on_complete: bool,
        abort_signal: AbortSignal,
    ) -> RunOutput:
        recorder = RunRecorder(
            context=context,
            hooks=definition.hooks,
            step_observers=step_observers,
        )
        run = self._create_run(user_input, context)
        await recorder.start_run(run)
        try:
            before_run_hook_result = await self._run_before_run_hook(
                hooks=definition.hooks,
                user_input=user_input,
                context=context,
            )
            memories = await self._retrieve_memories(
                hooks=definition.hooks,
                user_input=user_input,
                context=context,
            )
            user_step = await recorder.create_user_step(user_input=user_input)
            await recorder.commit_step(user_step, append_message=False)

            user_message = normalize_to_message(user_input)
            executor = AgentExecutor(
                model=definition.model,
                tools=list(definition.tools),
                options=definition.options,
                hooks=definition.hooks,
                run_recorder=recorder,
                root_path=definition.options.get_effective_root_path(),
            )
            result = await executor.execute(
                system_prompt=definition.system_prompt,
                user_step=user_step,
                context=context,
                memories=memories,
                before_run_hook_result=before_run_hook_result,
                channel_context=user_message.context,
                abort_signal=abort_signal,
            )

            if definition.hooks.on_after_run:
                await definition.hooks.on_after_run(result, context)
            if definition.hooks.on_memory_write and result.response:
                await definition.hooks.on_memory_write(user_input, result, context)

            await recorder.complete_run(result)
            return result
        except Exception as error:
            await recorder.fail_run(error)
            raise
        finally:
            if close_session_on_complete:
                await context.session_runtime.close()

    @staticmethod
    async def _run_before_run_hook(
        *,
        hooks: AgentHooks,
        user_input: UserInput,
        context: AgentRunContext,
    ):
        if hooks.on_before_run is None:
            return None
        return await hooks.on_before_run(user_input, context)

    @staticmethod
    async def _retrieve_memories(
        *,
        hooks: AgentHooks,
        user_input: UserInput,
        context: AgentRunContext,
    ) -> list:
        if hooks.on_memory_retrieve is None or not user_input:
            return []
        return await hooks.on_memory_retrieve(user_input, context)

    @staticmethod
    def _start_trace_runtime(
        *,
        resource_owner: AgentResourceOwner,
        agent_id: str,
        session_id: str,
        user_id: str | None,
        user_input: UserInput,
    ) -> AgentTraceCollector | None:
        if resource_owner.trace_storage is None:
            return None
        collector = AgentTraceCollector(store=resource_owner.trace_storage)
        collector.start(
            agent_id=agent_id,
            session_id=session_id,
            user_id=user_id,
            input_query=extract_text(user_input),
        )
        return collector

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


__all__ = ["AgentRunner"]
