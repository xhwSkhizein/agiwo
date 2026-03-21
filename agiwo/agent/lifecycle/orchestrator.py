import asyncio
from collections.abc import Awaitable, Callable, Sequence
from uuid import uuid4

from agiwo.agent.engine.context import AgentRunContext
from agiwo.agent.engine.engine import ExecutionEngine
from agiwo.agent.execution import AgentExecutionHandle
from agiwo.agent.input import UserInput
from agiwo.agent.input_codec import extract_text
from agiwo.agent.lifecycle.definition import ResolvedExecutionDefinition
from agiwo.agent.lifecycle.resource_owner import ActiveRootExecution, AgentResourceOwner
from agiwo.agent.lifecycle.session import AgentSessionRuntime
from agiwo.agent.runtime import RunOutput
from agiwo.agent.scheduler_port import StepObserver
from agiwo.agent.trace import AgentTraceCollector
from agiwo.utils.abort_signal import AbortSignal


class ExecutionOrchestrator:
    """Own root/child session wiring and live execution lifecycle."""

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
                step_observers=tuple(step_observers),
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
        return await self._execute(
            user_input,
            context=context,
            definition=definition,
            step_observers=tuple(step_observers),
            close_session_on_complete=False,
            abort_signal=child_abort_signal,
        )

    async def _execute_root(
        self,
        user_input: UserInput,
        *,
        context: AgentRunContext,
        step_observers: tuple[StepObserver, ...],
        resolve_definition: Callable[[], Awaitable[ResolvedExecutionDefinition]],
        abort_signal: AbortSignal,
    ) -> RunOutput:
        try:
            definition = await resolve_definition()
        except Exception:
            await self._close_session_if_needed(
                session_runtime=context.session_runtime,
                close_session_on_complete=True,
            )
            raise
        return await self._execute(
            user_input,
            context=context,
            definition=definition,
            step_observers=step_observers,
            close_session_on_complete=True,
            abort_signal=abort_signal,
        )

    async def _execute(
        self,
        user_input: UserInput,
        *,
        context: AgentRunContext,
        definition: ResolvedExecutionDefinition,
        step_observers: tuple[StepObserver, ...],
        close_session_on_complete: bool,
        abort_signal: AbortSignal,
    ) -> RunOutput:
        engine = ExecutionEngine(
            definition=definition,
            step_observers=step_observers,
            root_path=definition.options.get_effective_root_path(),
        )
        try:
            return await engine.execute(
                user_input,
                context=context,
                abort_signal=abort_signal,
            )
        finally:
            await self._close_session_if_needed(
                session_runtime=context.session_runtime,
                close_session_on_complete=close_session_on_complete,
            )

    @staticmethod
    async def _close_session_if_needed(
        *,
        session_runtime: AgentSessionRuntime,
        close_session_on_complete: bool,
    ) -> None:
        if close_session_on_complete:
            await session_runtime.close()

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


__all__ = ["ExecutionOrchestrator"]
