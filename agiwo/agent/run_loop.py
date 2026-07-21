"""Single-run execution engine — the core run loop."""

import asyncio
from dataclasses import dataclass
from typing import NamedTuple

from agiwo.agent.budget_gate import LlmBudgetDenied
from agiwo.agent.compaction import CompactResult, compact_if_needed
from agiwo.agent.hooks import HookRegistration, HookRegistry
from agiwo.agent.llm_caller import ModelCallLimitExceeded, execute_model_call
from agiwo.agent.models.execution import RunTreeRole
from agiwo.agent.models.finalization import (
    RunFinalizationResult,
    derive_mechanical_finalization,
    mechanical_agent_handoff_result,
)
from agiwo.agent.models.model_call import ModelCallPhase
from agiwo.agent.models.config import AgentOptions
from agiwo.agent.models.input import UserInput
from agiwo.agent.models.run import RunMetrics, RunOutput, TerminationReason
from agiwo.agent.models.step import LLMCallContext, StepView
from agiwo.agent.pause import PauseReason
from agiwo.agent.prompt import apply_steering_messages
from agiwo.agent.retry import (
    RetryCoordinator,
    RetryPolicy,
    RunBlockingFaultError,
    finalization_for_blocking_fault,
)
from agiwo.agent.run_bootstrap import prepare_run_context
from agiwo.agent.run_tool_batch import execute_tool_batch_cycle
from agiwo.agent.runtime.context import RunContext, RunRuntime
from agiwo.agent.runtime.state_writer import RunStateWriter
from agiwo.agent.termination.limits import (
    check_non_recoverable_limits,
    check_post_llm_limits,
)
from agiwo.agent.termination.summarizer import maybe_generate_termination_summary
from agiwo.config.settings import settings
from agiwo.llm.base import Model
from agiwo.llm.limits import (
    resolve_max_context_window,
    resolve_max_input_tokens_per_call,
)
from agiwo.tool.base import BaseTool
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger


class CompactionCycleResult(NamedTuple):
    """Result of a compaction cycle."""

    compact_start_seq: int
    skip_assistant_turn: bool


@dataclass(frozen=True, slots=True)
class LoopCompleted:
    """Normal loop termination; proceed to finalize."""


@dataclass(frozen=True, slots=True)
class LoopPaused:
    """Recoverable pause after a committed checkpoint."""

    reason: str
    checkpoint_id: str


@dataclass(frozen=True, slots=True)
class LoopFault:
    """Blocking fault that must be finalized or failed at the run boundary."""

    error: RunBlockingFaultError


LoopExit = LoopCompleted | LoopPaused | LoopFault

logger = get_logger(__name__)


class RunLoopOrchestrator:
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
    ) -> RunOutput:
        """Execute a single agent run with the orchestrator."""
        try:
            if resume:
                if resume_checkpoint_id is None:
                    raise ValueError("resume requires resume_checkpoint_id")
                await self.writer.record_resumed(checkpoint_id=resume_checkpoint_id)
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

    async def _dispatch_loop_exit(
        self,
        user_input: UserInput | None,
        exit_result: LoopExit,
    ) -> RunOutput:
        """Single dispatch point for loop outcomes."""
        if isinstance(exit_result, LoopPaused):
            return self._build_paused_output(
                reason=exit_result.reason,
                checkpoint_id=exit_result.checkpoint_id,
            )
        if isinstance(exit_result, LoopFault):
            return await self._finalize_blocking_fault(user_input, exit_result.error)
        return await self._finalize_run(user_input)

    def _build_paused_output(self, *, reason: str, checkpoint_id: str) -> RunOutput:
        return RunOutput(
            response=self.context.ledger.response_content,
            run_id=self.context.run_id,
            session_id=self.context.session_id,
            metrics=RunMetrics.from_ledger(
                self.context.ledger,
                elapsed_ms=self.context.elapsed * 1000,
            ),
            termination_reason=None,
            metadata={
                "run_start_seq": self.context.ledger.run_start_seq,
                "pause_reason": reason,
            },
            paused=True,
            checkpoint_id=checkpoint_id,
        )

    async def _finalize_blocking_fault(
        self,
        user_input: UserInput | None,
        blocked: RunBlockingFaultError,
    ) -> RunOutput:
        """End ROOT assignment via mechanical Decision; fail non-root runs."""
        del user_input
        if self.context.run_tree_role is not RunTreeRole.ROOT:
            await self._fail_run(blocked)
            raise blocked
        carry = [
            {
                "id": m.id,
                "description": m.description,
                "status": m.status
                if isinstance(m.status, str)
                else getattr(m.status, "value", str(m.status)),
            }
            for m in self.context.ledger.plan.milestones
            if (
                m.status
                if isinstance(m.status, str)
                else getattr(m.status, "value", str(m.status))
            )
            in {"pending", "active"}
        ]
        self._finalization = finalization_for_blocking_fault(
            blocked,
            carry_forward=carry,
            plan_items=carry,
        )
        await self._set_termination_reason(
            TerminationReason.COMPLETED,
            phase="fault_boundary",
            source=blocked.fault.disposition.value,
        )
        return await self._finalize_run(None)

    async def _start_run(self, user_input: UserInput | None) -> None:
        """Initialize and start the run."""
        self.context.ledger.model_calls.configured_limit = (
            self.runtime.config.max_steps_per_run
        )
        await self.writer.start_run(user_input)

    async def _complete_run(self, result: RunOutput) -> None:
        """Complete the run successfully."""
        await self.writer.finish_run(result)

    async def _fail_run(self, error: Exception) -> None:
        """Handle run failure."""
        await self.writer.fail_run(error)

    def _build_output(self) -> RunOutput:
        """Build the run output from the current state."""
        return RunOutput(
            response=self.context.ledger.response_content,
            run_id=self.context.run_id,
            session_id=self.context.session_id,
            metrics=RunMetrics.from_ledger(
                self.context.ledger,
                elapsed_ms=self.context.elapsed * 1000,
            ),
            termination_reason=self.context.ledger.termination_reason,
            metadata={"run_start_seq": self.context.ledger.run_start_seq},
            finalization=self._finalization,
        )

    async def _set_termination_reason(
        self,
        reason: TerminationReason,
        *,
        phase: str,
        source: str,
    ) -> None:
        if self.context.ledger.termination_reason == reason:
            return
        await self.writer.record_termination_decided(
            termination_reason=reason,
            phase=phase,
            source=source,
        )

    async def _finalize_run(self, user_input: UserInput | None) -> RunOutput:
        """Generate summary, build output, and complete the run."""
        if (
            self.context.run_tree_role is RunTreeRole.ROOT
            and self.context.ledger.termination_reason is TerminationReason.MAX_STEPS
            and self._finalization is None
        ):
            self._finalization = mechanical_agent_handoff_result(
                self.context.ledger.response_content or "",
                reason="max_steps_per_run_mechanical_handoff",
                carry_forward=self._remaining_plan_items(),
            )
        await maybe_generate_termination_summary(
            state=self.context,
            options=self.runtime.config,
            model=self.runtime.model,
            abort_signal=self.runtime.abort_signal,
            commit_step=self._commit_step,
        )
        result = self._build_output()
        if self._finalization is not None:
            # A termination summary is diagnostic only; it must not replace the
            # ordinary report submitted to ObjectiveService.
            result.response = self._finalization.report
            self.context.ledger.response_content = self._finalization.report
        await self.context.hooks.after_run(result, self.context)
        if result.response is not None and user_input is not None:
            await self.context.hooks.memory_write(user_input, result, self.context)
        await self._complete_run(result)
        return result

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
        except LlmBudgetDenied as denied:
            self.context.request_pause(denied.reason or PauseReason.LLM_BUDGET_DENIED)
            return await self._commit_pause()
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

    async def _run_compaction_cycle(self) -> CompactionCycleResult:
        """Run compaction cycle if needed."""
        result: CompactResult = await compact_if_needed(
            state=self.context,
            model=self.runtime.model,
            abort_signal=self.runtime.abort_signal,
            max_context_window=self.runtime.max_context_window,
            commit_step=self._commit_step,
            compact_prompt=self.runtime.compact_prompt,
            compact_start_seq=self.runtime.compact_start_seq,
            root_path=self.runtime.root_path,
        )
        if result.failed:
            failure_count = self.writer.next_compaction_failure_attempt()
            err = result.error or ""
            terminal = failure_count >= 3
            await self.writer.record_compaction_failed(
                error=err,
                attempt=failure_count,
                max_attempts=3,
                terminal=terminal,
            )
            await self.context.hooks.compaction_failed(
                self.context.run_id, err, failure_count, self.context
            )
            logger.warning(
                "compaction_failed",
                run_id=self.context.run_id,
                error=err,
                failure_count=failure_count,
            )
            if terminal:
                await self._set_termination_reason(
                    TerminationReason.MAX_INPUT_TOKENS_PER_CALL,
                    phase="compaction",
                    source="compaction_failure_limit",
                )
            return CompactionCycleResult(self.runtime.compact_start_seq, False)

        compact_metadata = result.metadata
        if compact_metadata is None:
            return CompactionCycleResult(self.runtime.compact_start_seq, False)

        new_start_seq = compact_metadata.end_seq + 1
        logger.info(
            "compact_triggered",
            run_id=self.context.run_id,
            before_messages=len(self.context.ledger.messages),
        )
        if (
            self.runtime.config.max_run_cost is not None
            and self.context.ledger.tokens.cost >= self.runtime.config.max_run_cost
        ):
            await self._set_termination_reason(
                TerminationReason.MAX_RUN_COST,
                phase="compaction",
                source="max_run_cost_after_compaction",
            )
            return CompactionCycleResult(new_start_seq, True)
        return CompactionCycleResult(new_start_seq, False)

    async def _run_assistant_turn(self) -> tuple[StepView, LLMCallContext]:
        """Execute an assistant turn (LLM call)."""
        request_messages = self.context.snapshot_messages()
        pending_steer = self.context.session_runtime.peek_pending_steer_inputs()
        if pending_steer:
            request_messages = apply_steering_messages(request_messages, pending_steer)
            await self.writer.rebuild_messages(
                reason="before_llm",
                messages=request_messages,
            )
            self.context.session_runtime.ack_pending_steer_inputs(len(pending_steer))
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
                if self._has_unfinished_plan_items():
                    await self._append_plan_guard_reminder()
                    return None
                report = step.content if isinstance(step.content, str) else ""
                self._finalization = derive_mechanical_finalization(
                    report=report,
                    verification_required=(
                        self.context.verification_required
                        or bool(self.context.ledger.plan.milestones)
                    ),
                    objective_run_role=self.context.objective_run_role,
                )
            elif self.context.run_tree_role is RunTreeRole.NONE and (
                self.context.verification_required
                or bool(self.context.ledger.plan.milestones)
            ):
                # Plain Session turn that grew a plan → mechanical HandoffDecision
                # so SessionGateway can upgrade into an Objective (ADR 0047).
                report = step.content if isinstance(step.content, str) else ""
                self._finalization = derive_mechanical_finalization(
                    report=report,
                    verification_required=True,
                    objective_run_role=self.context.objective_run_role,
                )
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

    def _has_unfinished_plan_items(self) -> bool:
        return any(
            milestone.status in {"pending", "active"}
            for milestone in self.context.ledger.plan.milestones
        )

    def _remaining_plan_items(self) -> list[dict[str, str]]:
        return [
            {
                "id": milestone.id,
                "description": milestone.description,
                "status": milestone.status,
            }
            for milestone in self.context.ledger.plan.milestones
            if milestone.status in {"pending", "active"}
        ]

    async def _append_plan_guard_reminder(self) -> None:
        remaining = self._remaining_plan_items()
        items = "\n".join(
            f"- [{item['status']}] {item['id']}: {item['description']}"
            for item in remaining
        )
        messages = self.context.snapshot_messages()
        messages.append(
            {
                "role": "user",
                "content": (
                    "The run plan still has unfinished milestones. Continue the "
                    "assignment and use update_plan to complete, abandon, or "
                    "revise them before ending:\n"
                    f"{items}"
                ),
                "is_user_provided": False,
                "origin": "assignment_plan_guard",
            }
        )
        await self.writer.rebuild_messages(
            reason="assignment_plan_guard",
            messages=messages,
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
    return await orchestrator.execute_run(
        user_input,
        system_prompt,
        pending_tool_calls,
    )


__all__ = [
    "LoopCompleted",
    "LoopExit",
    "LoopFault",
    "LoopPaused",
    "RunLoopOrchestrator",
    "execute_run",
]
