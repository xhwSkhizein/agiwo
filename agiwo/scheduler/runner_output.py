"""Pure helpers for classifying scheduler run outputs."""

from agiwo.agent import RunOutput, TerminationReason
from agiwo.scheduler.commands import DispatchAction
from agiwo.scheduler.models import SchedulerRunResult, WakeType

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


__all__ = [
    "build_last_run_result",
    "is_failed_output",
    "is_interrupted_output",
    "is_normal_completion",
    "is_periodic_wake",
    "is_sleeping_output",
    "periodic_wait_seconds",
]
