"""Run-level model-call limit policy for max_steps_per_run."""

from dataclasses import dataclass

from agiwo.agent.models.model_call import (
    FINALIZATION_PHASES,
    ModelCallLedger,
    ModelCallPhase,
    WORK_PHASES,
)


@dataclass(frozen=True)
class RunLimitDecision:
    allowed: bool
    reason: str | None = None


class RunLimitPolicy:
    """Decides whether a new provider attempt may start under max_steps_per_run."""

    def check_before_attempt(
        self,
        ledger: ModelCallLedger,
        phase: ModelCallPhase,
    ) -> RunLimitDecision:
        limit = ledger.configured_limit

        if phase in WORK_PHASES:
            if ledger.total_attempts >= limit:
                return RunLimitDecision(False, "work_limit_exceeded")
            return RunLimitDecision(True)

        if phase in FINALIZATION_PHASES:
            if ledger.total_attempts < limit:
                return RunLimitDecision(True)
            if ledger.is_finalization_consumed(phase):
                return RunLimitDecision(False, f"finalization_{phase.value}_exhausted")
            return RunLimitDecision(True)

        return RunLimitDecision(False, f"unsupported_phase_{phase.value}")

    def should_refuse_work_at_limit(self, ledger: ModelCallLedger) -> bool:
        return ledger.at_or_over_limit()


__all__ = ["RunLimitDecision", "RunLimitPolicy"]
