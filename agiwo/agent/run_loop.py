"""Single-run execution engine — the core run loop."""

import asyncio

from agiwo.agent.compaction import compact_if_needed
from agiwo.agent.completion_gates import AllowComplete, CompletionGates, Continue
from agiwo.agent.completion_gates.context import CompletionGateContext
from agiwo.agent.hooks import HookRegistration, HookRegistry
from agiwo.agent.llm_caller import ModelCallLimitExceeded, execute_model_call
from agiwo.agent.models.execution import RunTreeRole
from agiwo.agent.models.finalization import RunFinalizationResult, completion_result
from agiwo.agent.models.model_call import ModelCallPhase
from agiwo.agent.models.config import AgentOptions
from agiwo.agent.models.input import UserInput, UserMessage
from agiwo.agent.models.run import RunOutput, TerminationReason
from agiwo.agent.models.step import LLMCallContext, StepView
from agiwo.agent.prompt import append_pending_user_messages
from agiwo.agent.retry import (
    RetryCoordinator,
    RetryPolicy,
    RunBlockingFaultError,
)
from agiwo.agent.run_bootstrap import prepare_run_context
from agiwo.agent.run_loop_compaction import RunLoopCompactionOps
from agiwo.agent.run_loop_finalization import RunLoopFinalizationOps
from agiwo.agent.run_loop_models import (
    CompactionCycleResult,
    LoopCompleted,
    LoopExit,
    LoopFault,
    LoopPaused,
)
from agiwo.agent.run_tool_batch import execute_tool_batch_cycle
from agiwo.agent.runtime.context import RunContext, RunRuntime
from agiwo.agent.runtime.state_writer import RunStateWriter
from agiwo.agent.termination.limits import (
    check_non_recoverable_limits,
    check_post_llm_limits,
)
from agiwo.config.settings import settings
from agiwo.llm.base import Model
from agiwo.llm.limits import (
    resolve_max_context_window,
    resolve_max_input_tokens_per_call,
)
from agiwo.tool.base import BaseTool
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class RunLoopOrchestrator(
    RunLoopFinalizationOps,
    RunLoopCompactionOps,
):
    """RunLoopOrchestrator is the single-run execution owner."""

    def __init__(
        self,
        context: RunContext,
        runtime: RunRuntime,
    ):
        self.context = context
        self.runtime = runtime
        self.writer = RunStateWriter(context)
        self._finalization: RunFinalizationResult | None = None

    async def _commit_step(
        self,
        step: StepView,
        *,
        append_message: bool = True,
        track_state: bool = True,
    ) -> StepView:
        await self.writer.commit_step(
            step,
            append_message=append_message,
            track_state=track_state,
        )
        await self.context.hooks.on_step(step, self.context)
        return step

    async def execute_run(
        self,
        user_input: UserInput | None,
        system_prompt: str,
        pending_tool_calls: list[dict] | None = None,
        *,
        resume: bool = False,
        resume_checkpoint_id: str | None = None,
        continuation: bool = False,
    ) -> RunOutput:
        """Execute a single agent run with the orchestrator."""
        try:
            if resume:
                if resume_checkpoint_id is None:
                    raise ValueError("resume requires resume_checkpoint_id")
                await self.writer.record_resumed(checkpoint_id=resume_checkpoint_id)
            elif continuation:
                bootstrap = await prepare_run_context(
                    context=self.context,
                    runtime=self.runtime,
                    user_input=user_input,
                    system_prompt=system_prompt,
                    writer=self.writer,
                )
                if bootstrap.user_step is not None:
                    await self._commit_step(
                        bootstrap.user_step,
                        append_message=False,
                        track_state=False,
                    )
                self.runtime.compact_start_seq = bootstrap.compact_start_seq
            else:
                await self._start_run(user_input)
                bootstrap = await prepare_run_context(
                    context=self.context,
                    runtime=self.runtime,
                    user_input=user_input,
                    system_prompt=system_prompt,
                    writer=self.writer,
                )
                if bootstrap.user_step is not None:
                    await self._commit_step(
                        bootstrap.user_step,
                        append_message=False,
                        track_state=False,
                    )
                self.runtime.compact_start_seq = bootstrap.compact_start_seq

            exit_result = await self._run_loop(pending_tool_calls=pending_tool_calls)
            return await self._dispatch_loop_exit(user_input, exit_result)
        except Exception as error:
            await self._fail_run(error)
            raise

    async def _run_loop(
        self,
        pending_tool_calls: list[dict] | None,
    ) -> LoopExit:
        """Main run loop. Returns an explicit exit; does not use pause/fault exceptions."""
        try:
            pending_exit = await self._run_pending_tools(pending_tool_calls)
            if pending_exit is not None:
                return pending_exit

            while not self.context.is_terminal:
                exit_result = await self._run_loop_iteration()
                if exit_result is not None:
                    return exit_result
            return LoopCompleted()
        except RunBlockingFaultError as blocked:
            return LoopFault(blocked)
        except asyncio.CancelledError:
            return await self._exit_cancelled()
        except Exception:
            await self._mark_loop_exception()
            raise

    async def _run_pending_tools(
        self,
        pending_tool_calls: list[dict] | None,
    ) -> LoopExit | None:
        if not pending_tool_calls:
            return None
        paused = await self._maybe_pause()
        if paused is not None:
            return paused
        terminated = await self._execute_tool_calls(
            tool_calls=pending_tool_calls,
            assistant_step_id=None,
        )
        if terminated:
            return LoopCompleted()
        return await self._maybe_pause()

    async def _exit_cancelled(self) -> LoopCompleted:
        await self._set_termination_reason(
            TerminationReason.CANCELLED,
            phase="run_loop",
            source="cancelled_error",
        )
        logger.info("agent_execution_cancelled", run_id=self.context.run_id)
        return LoopCompleted()

    async def _mark_loop_exception(self) -> None:
        await self._set_termination_reason(
            TerminationReason.ERROR_WITH_CONTEXT
            if self.context.ledger.steps.assistant > 0
            else TerminationReason.ERROR,
            phase="run_loop",
            source="exception",
        )
        logger.error(
            "agent_execution_failed",
            run_id=self.context.run_id,
            steps_completed=self.context.ledger.steps.total,
            termination_reason=self.context.ledger.termination_reason,
            exc_info=True,
        )

    async def _maybe_pause(self) -> LoopPaused | None:
        if self.context.pause_request is None:
            return None
        return await self._commit_pause()

    async def _commit_pause(self) -> LoopPaused:
        request = self.context.pause_request
        reason = request.reason if request is not None else "pause"
        self.context.pause_request = None
        _, checkpoint_id = await self.writer.pause_at_checkpoint(reason=reason)
        logger.info(
            "agent_run_paused",
            run_id=self.context.run_id,
            reason=reason,
            checkpoint_id=checkpoint_id,
        )
        return LoopPaused(reason=reason, checkpoint_id=checkpoint_id)

    async def _run_loop_iteration(self) -> LoopExit | None:
        """One iteration. None = continue; LoopExit = stop with that outcome."""
        paused = await self._maybe_pause()
        if paused is not None:
            return paused
        reason = check_non_recoverable_limits(
            self.context,
            self.runtime.config,
            self.context.ledger.steps.current,
        )
        if reason is not None:
            await self._set_termination_reason(
                reason,
                phase="pre_llm",
                source="non_recoverable_limit",
            )
            return LoopCompleted()

        result = await self._run_compaction_cycle()
        compact_start_seq = result.compact_start_seq
        skip_assistant_turn = result.skip_assistant_turn
        self.runtime.compact_start_seq = compact_start_seq
        if skip_assistant_turn or self.context.is_terminal:
            return LoopCompleted() if self.context.is_terminal else None

        self.context.ledger.steps.current += 1
        try:
            step, llm_context = await self._run_assistant_turn()
        except ModelCallLimitExceeded:
            await self._set_termination_reason(
                TerminationReason.MAX_STEPS,
                phase="pre_llm",
                source="model_call_limit",
            )
            return LoopCompleted()
        return await self._handle_assistant_turn_result(
            step=step,
            llm_context=llm_context,
        )

    async def _run_assistant_turn(self) -> tuple[StepView, LLMCallContext]:
        """Execute an assistant turn (LLM call)."""
        request_messages = self.context.snapshot_messages()
        pending_inputs = self.context.session_runtime.peek_pending_inputs()
        if pending_inputs:
            request_messages = append_pending_user_messages(
                request_messages, pending_inputs
            )
            await self.writer.rebuild_messages(
                reason="before_llm",
                messages=request_messages,
            )
            self.context.session_runtime.ack_pending_inputs(len(pending_inputs))
        request_messages = self.context.snapshot_messages()
        modified = await self.context.hooks.before_llm_call(
            request_messages, self.context
        )
        if modified is not None:
            request_messages = modified
        if request_messages != self.context.snapshot_messages():
            await self.writer.rebuild_messages(
                reason="before_llm",
                messages=request_messages,
            )

        try:
            call_result = await execute_model_call(
                model=self.runtime.model,
                state=self.context,
                writer=self.writer,
                phase=ModelCallPhase.ASSISTANT,
                abort_signal=self.runtime.abort_signal,
                messages=self.context.snapshot_messages(),
                tools=self.context.copy_tool_schemas(),
                use_state_tools=False,
                retry_coordinator=self.runtime.retry_coordinator,
            )
        except ModelCallLimitExceeded as exc:
            logger.warning(
                "assistant_model_call_refused",
                run_id=self.context.run_id,
                reason=exc.reason,
            )
            raise
        step = call_result.step
        llm_context = call_result.llm_context
        await self._commit_step(step)
        await self.context.hooks.after_llm_call(step, self.context)
        return step, llm_context

    async def _handle_assistant_turn_result(
        self,
        step: StepView,
        llm_context: LLMCallContext,
    ) -> LoopExit | None:
        """Handle the result of an assistant turn.

        Returns None to continue the loop, or a LoopExit to stop.
        """
        reason = check_post_llm_limits(
            self.context,
            step,
            llm_context,
            options=self.runtime.config,
            max_input_tokens_per_call=self.runtime.max_input_tokens_per_call,
        )
        if reason is not None:
            await self._set_termination_reason(
                reason,
                phase="post_llm",
                source="post_llm_limit",
            )
            return LoopCompleted()

        if not step.tool_calls:
            paused = await self._maybe_pause()
            if paused is not None:
                return paused
            if self.context.run_tree_role is RunTreeRole.ROOT:
                gate_decision = await self._evaluate_completion_gates()
                if isinstance(gate_decision, Continue):
                    await self._enqueue_gate_feedback(gate_decision.feedback_text)
                    return None
            report = step.content if isinstance(step.content, str) else ""
            if report:
                self._finalization = completion_result(report)
            await self._set_termination_reason(
                TerminationReason.COMPLETED,
                phase="post_llm",
                source="assistant_completed_without_tools",
            )
            return LoopCompleted()

        terminated = await self._execute_tool_calls(
            tool_calls=step.tool_calls,
            assistant_step_id=step.id,
        )
        if terminated:
            return LoopCompleted()
        return await self._maybe_pause()

    async def _evaluate_completion_gates(self) -> AllowComplete | Continue:
        gate_context = self.runtime.completion_gate_context or CompletionGateContext()
        active_worker_ids = gate_context.active_worker_ids()
        gates = CompletionGates(
            enable_semantic=self.runtime.config.enable_semantic_completion_gates,
        )
        return await gates.evaluate(
            plan=self.context.ledger.plan,
            active_worker_ids=active_worker_ids,
        )

    async def _enqueue_gate_feedback(self, feedback_text: str) -> None:
        gate_context = self.runtime.completion_gate_context
        if gate_context is not None and gate_context.on_gate_feedback is not None:
            await gate_context.on_gate_feedback(feedback_text)
        message = UserMessage.from_system(feedback_text)
        await self.context.session_runtime.enqueue_message(message)
        logger.info(
            "completion_gate_feedback_enqueued",
            run_id=self.context.run_id,
            feedback_chars=len(feedback_text),
        )

    async def _execute_tool_calls(
        self,
        tool_calls: list[dict[str, object]],
        *,
        assistant_step_id: str | None,
    ) -> bool:
        """Execute a batch of tool calls."""

        async def _set_tool_termination(
            reason: TerminationReason,
            source: str,
        ) -> None:
            await self._set_termination_reason(
                reason,
                phase="tool_result",
                source=source,
            )

        return await execute_tool_batch_cycle(
            context=self.context,
            runtime=self.runtime,
            tool_calls=tool_calls,
            assistant_step_id=assistant_step_id,
            set_termination_reason=_set_tool_termination,
            commit_step=self._commit_step,
        )


async def execute_run(
    user_input: UserInput | None,
    *,
    context: RunContext,
    system_prompt: str,
    model: Model,
    tools: tuple[BaseTool, ...],
    options: AgentOptions | None = None,
    hooks: HookRegistry | list[HookRegistration] | None = None,
    pending_tool_calls: list[dict] | None = None,
    abort_signal: AbortSignal | None = None,
    root_path: str | None = None,
    resume: bool = False,
    resume_checkpoint_id: str | None = None,
    continuation: bool = False,
    completion_gate_context: CompletionGateContext | None = None,
) -> RunOutput:
    """Execute a single agent run — the core entry point."""
    options = options or AgentOptions()
    hooks = hooks if isinstance(hooks, HookRegistry) else HookRegistry(hooks or [])
    context.config = options
    context.hooks = hooks

    max_context_window = resolve_max_context_window(model)
    max_input_tokens_per_call = resolve_max_input_tokens_per_call(
        options.max_input_tokens_per_call,
        model,
    )

    async def _retry_continue() -> bool:
        return context.pause_request is None

    runtime = RunRuntime(
        session_runtime=context.session_runtime,
        config=options,
        hooks=hooks,
        model=model,
        tools_map={tool.name: tool for tool in tools},
        abort_signal=abort_signal,
        root_path=root_path or settings.root_path,
        compact_start_seq=0,
        max_input_tokens_per_call=max_input_tokens_per_call,
        max_context_window=max_context_window,
        compact_prompt=options.compact_prompt,
        retry_coordinator=RetryCoordinator(
            RetryPolicy(
                max_attempts=options.max_provider_attempts,
                min_backoff_seconds=options.retry_min_backoff_seconds,
                max_backoff_seconds=options.retry_max_backoff_seconds,
            ),
            should_continue=_retry_continue,
        ),
        completion_gate_context=completion_gate_context,
    )

    orchestrator = RunLoopOrchestrator(context, runtime)
    if resume:
        # Resume continues with already-loaded messages; user_input is unused.
        return await orchestrator.execute_run(
            user_input,
            system_prompt,
            pending_tool_calls,
            resume=True,
            resume_checkpoint_id=resume_checkpoint_id,
        )
    if continuation:
        return await orchestrator.execute_run(
            user_input,
            system_prompt,
            pending_tool_calls,
            continuation=True,
        )
    return await orchestrator.execute_run(
        user_input,
        system_prompt,
        pending_tool_calls,
    )


__all__ = [
    "compact_if_needed",
    "CompactionCycleResult",
    "LoopCompleted",
    "LoopExit",
    "LoopFault",
    "LoopPaused",
    "RunLoopOrchestrator",
    "execute_run",
]
