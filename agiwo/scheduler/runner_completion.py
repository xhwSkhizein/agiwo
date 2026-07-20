"""Focused helpers for translating agent run outputs into scheduler state."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from agiwo.agent import RunOutput, TerminationReason
from agiwo.scheduler.commands import DispatchAction
from agiwo.scheduler.formatting import SHUTDOWN_SUMMARY_TASK
from agiwo.scheduler.models import (
    AgentState,
    SchedulerEventType,
    SchedulerRunResult,
    WakeType,
)
from agiwo.utils.abort_signal import AbortSignal

_FAILED_TERMINATIONS = frozenset(
    {
        TerminationReason.CANCELLED,
        TerminationReason.ERROR,
        TerminationReason.ERROR_WITH_CONTEXT,
        TerminationReason.TIMEOUT,
    }
)
_INTERRUPTED_TERMINATIONS = frozenset(
    {
        TerminationReason.MAX_STEPS,
        TerminationReason.MAX_OUTPUT_TOKENS,
        TerminationReason.MAX_INPUT_TOKENS_PER_CALL,
        TerminationReason.MAX_RUN_COST,
        TerminationReason.TOOL_LIMIT,
    }
)


def build_last_run_result(
    *,
    termination_reason: TerminationReason,
    run_id: str | None,
    summary: str | None = None,
    error: str | None = None,
) -> SchedulerRunResult:
    return SchedulerRunResult(
        run_id=run_id,
        termination_reason=termination_reason,
        summary=summary,
        error=error,
    )


def is_failed_output(output: RunOutput) -> bool:
    return any(output.termination_reason is reason for reason in _FAILED_TERMINATIONS)


def is_interrupted_output(output: RunOutput) -> bool:
    return any(
        output.termination_reason is reason for reason in _INTERRUPTED_TERMINATIONS
    )


def is_sleeping_output(output: RunOutput) -> bool:
    return output.termination_reason is TerminationReason.SLEEPING


def is_paused_output(output: RunOutput) -> bool:
    """Recoverable interrupt: not a termination, not cancel."""
    return bool(output.paused)


def periodic_wait_seconds(action: DispatchAction) -> float | None:
    wake_condition = action.state.wake_condition
    if wake_condition is None or wake_condition.type is not WakeType.PERIODIC:
        return None
    return wake_condition.to_seconds()


def is_periodic_wake(action: DispatchAction) -> bool:
    return periodic_wait_seconds(action) is not None


def is_normal_completion(action: DispatchAction, output: RunOutput) -> bool:
    return (
        output.termination_reason is TerminationReason.COMPLETED
        and not is_periodic_wake(action)
    )


@dataclass(frozen=True, slots=True)
class RunnerCompletionContext:
    """Minimal side-effect surface needed to finalize one scheduler cycle."""

    save_state: Callable[[AgentState], Awaitable[None]]
    emit_event_to_parent: Callable[
        [AgentState, SchedulerEventType, dict[str, object]],
        Awaitable[None],
    ]
    resolve_abort_signal: Callable[[str, AbortSignal | None], AbortSignal | None]
    should_rollback: Callable[[AgentState], bool]
    rollback_run_steps: Callable[[AgentState, RunOutput], Awaitable[None]]
    shutdown_requested: set[str]


class RunnerCompletionHandler:
    """Own the scheduler's run-output classification and completion rules."""

    def __init__(self, context: RunnerCompletionContext) -> None:
        self._ctx = context

    async def handle(
        self,
        *,
        state: AgentState,
        output: RunOutput,
        action: DispatchAction,
        abort_signal: AbortSignal | None,
    ) -> bool:
        text = output.response

        if await self._finalize_if_aborted_terminal(state, abort_signal):
            return True

        handled = False
        if state.id in self._ctx.shutdown_requested:
            handled = await self._handle_shutdown_requested(state, text)
        elif is_paused_output(output):
            handled = await self._handle_paused_output(state, output, text)
        elif is_failed_output(output):
            handled = await self._handle_failed_output(state, output, text)
        elif is_interrupted_output(output):
            handled = await self._handle_interrupted_output(state, output, text)
        elif is_sleeping_output(output):
            handled = await self._handle_sleeping(state, text)
        elif is_periodic_wake(action):
            handled = await self._handle_periodic_output(state, output, text, action)
        elif is_normal_completion(action, output):
            handled = await self._complete_state(state, output, text)
        return handled

    async def _finalize_if_aborted_terminal(
        self,
        state: AgentState,
        abort_signal: AbortSignal | None,
    ) -> bool:
        """Preserve or synthesize a cancelled result for terminal aborted states."""
        abort_signal = self._ctx.resolve_abort_signal(state.id, abort_signal)
        if not (
            state.is_terminal()
            and abort_signal is not None
            and abort_signal.is_aborted()
        ):
            return False

        if state.last_run_result is None:
            reason = abort_signal.reason or "Cancelled"
            await self._ctx.save_state(
                state.with_updates(
                    last_run_result=build_last_run_result(
                        termination_reason=TerminationReason.CANCELLED,
                        run_id=None,
                        error=reason,
                    )
                )
            )
        return True

    async def _handle_shutdown_requested(
        self,
        state: AgentState,
        text: str | None,
    ) -> bool:
        self._ctx.shutdown_requested.discard(state.id)
        if state.is_root and state.is_persistent:
            await self._ctx.save_state(
                state.with_queued(
                    pending_input=SHUTDOWN_SUMMARY_TASK,
                ).with_updates(result_summary=text)
            )
            return True

        error = "Shutdown before completion"
        await self._ctx.save_state(
            state.with_failed(error).with_updates(
                last_run_result=build_last_run_result(
                    termination_reason=TerminationReason.ERROR,
                    run_id=None,
                    summary=text,
                    error=error,
                )
            )
        )
        if state.is_child:
            await self._ctx.emit_event_to_parent(
                state,
                SchedulerEventType.CHILD_FAILED,
                {"reason": error},
            )
        return True

    async def _handle_failed_output(
        self,
        state: AgentState,
        output: RunOutput,
        text: str | None,
    ) -> bool:
        reason = output.error or text or output.termination_reason.value
        await self._ctx.save_state(
            state.with_failed(reason).with_updates(
                last_run_result=build_last_run_result(
                    termination_reason=output.termination_reason,
                    run_id=output.run_id,
                    summary=text,
                    error=reason,
                )
            )
        )
        if state.is_child:
            await self._ctx.emit_event_to_parent(
                state,
                SchedulerEventType.CHILD_FAILED,
                {"reason": reason},
            )
        return True

    async def _handle_interrupted_output(
        self,
        state: AgentState,
        output: RunOutput,
        text: str | None,
    ) -> bool:
        reason = output.termination_reason.value
        last_run_result = build_last_run_result(
            termination_reason=output.termination_reason,
            run_id=output.run_id,
            summary=text,
        )
        if state.is_root and state.is_persistent:
            await self._ctx.save_state(
                state.with_idle(result_summary=text or reason).with_updates(
                    last_run_result=last_run_result
                )
            )
            return True

        await self._ctx.save_state(
            state.with_failed(text or reason).with_updates(
                last_run_result=last_run_result
            )
        )
        if state.is_child:
            await self._ctx.emit_event_to_parent(
                state,
                SchedulerEventType.CHILD_FAILED,
                {"reason": reason},
            )
        return True

    async def _handle_paused_output(
        self,
        state: AgentState,
        output: RunOutput,
        text: str | None,
    ) -> bool:
        """Park persistent roots in IDLE without writing a termination Outcome path."""
        reason = (
            (output.metadata or {}).get("pause_reason")
            if isinstance(output.metadata, dict)
            else None
        )
        summary = text or (str(reason) if reason else "paused")
        if state.is_root and state.is_persistent:
            await self._ctx.save_state(
                state.with_idle(result_summary=summary).with_updates(
                    last_run_result=None
                )
            )
            return True
        await self._ctx.save_state(
            state.with_idle(result_summary=summary).with_updates(last_run_result=None)
        )
        if state.is_child:
            await self._ctx.emit_event_to_parent(
                state,
                SchedulerEventType.CHILD_SLEEP_RESULT,
                {
                    "result": summary,
                    "explain": "paused",
                    "paused": True,
                    "checkpoint_id": output.checkpoint_id,
                },
            )
        return True

    async def _handle_sleeping(
        self,
        state: AgentState,
        text: str | None,
    ) -> bool:
        # No save_state here: the sleep tool/caller already persisted WAITING state.
        await self._ctx.emit_event_to_parent(
            state,
            SchedulerEventType.CHILD_SLEEP_RESULT,
            {"result": text or "", "explain": state.explain},
        )
        return True

    async def _handle_periodic_output(
        self,
        state: AgentState,
        output: RunOutput,
        text: str | None,
        action: DispatchAction,
    ) -> bool:
        original_wc = action.state.wake_condition
        secs = periodic_wait_seconds(action)
        if original_wc is None or secs is None:
            raise RuntimeError("periodic output handling requires a periodic wake")

        should_rollback = state.no_progress and self._ctx.should_rollback(state)
        if should_rollback:
            await self._ctx.rollback_run_steps(state, output)

        next_wakeup = datetime.now(timezone.utc) + timedelta(seconds=secs)
        new_state = state.with_waiting(
            wake_condition=original_wc.with_next_wakeup(next_wakeup),
            result_summary=text if not should_rollback else state.result_summary,
        )
        if should_rollback:
            new_state = new_state.with_updates(
                rollback_count=state.rollback_count + 1,
            )
        await self._ctx.save_state(new_state)
        if not should_rollback:
            await self._ctx.emit_event_to_parent(
                state,
                SchedulerEventType.CHILD_SLEEP_RESULT,
                {
                    "result": text or "",
                    "explain": state.explain,
                    "periodic": True,
                },
            )
        return True

    async def _complete_state(
        self,
        state: AgentState,
        output: RunOutput,
        text: str | None,
    ) -> bool:
        last_run_result = build_last_run_result(
            termination_reason=TerminationReason.COMPLETED,
            run_id=output.run_id,
            summary=text,
        )
        if state.is_root and state.is_persistent:
            await self._ctx.save_state(
                state.with_idle(result_summary=text).with_updates(
                    last_run_result=last_run_result
                )
            )
            return True

        await self._ctx.save_state(
            state.with_completed(result_summary=text).with_updates(
                last_run_result=last_run_result
            )
        )
        if state.is_child:
            await self._ctx.emit_event_to_parent(
                state,
                SchedulerEventType.CHILD_COMPLETED,
                {"result": text or ""},
            )
        return True


__all__ = [
    "RunnerCompletionContext",
    "RunnerCompletionHandler",
    "build_last_run_result",
    "is_failed_output",
    "is_interrupted_output",
    "is_normal_completion",
    "is_paused_output",
    "is_periodic_wake",
    "is_sleeping_output",
    "periodic_wait_seconds",
]
