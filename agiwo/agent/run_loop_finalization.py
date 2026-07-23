"""Run finalization mixin for RunLoopOrchestrator."""

from agiwo.agent.models.execution import RunTreeRole
from agiwo.agent.models.finalization import fault_result
from agiwo.agent.models.input import UserInput
from agiwo.agent.models.run import RunMetrics, RunOutput, TerminationReason
from agiwo.agent.retry import RunBlockingFaultError, finalization_for_blocking_fault
from agiwo.agent.run_loop_models import LoopExit, LoopFault, LoopPaused
from agiwo.agent.termination.summarizer import maybe_generate_termination_summary


class RunLoopFinalizationOps:
    """Mixin: run start/complete/fail and loop-exit dispatch."""

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
            self._finalization = fault_result(
                self.context.ledger.response_content or "",
                reason="max_steps_per_run",
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
            # ordinary report; no cross-run control plane consumes it (ADR 0048).
            result.response = self._finalization.report
            self.context.ledger.response_content = self._finalization.report
        await self.context.hooks.after_run(result, self.context)
        if result.response is not None and user_input is not None:
            await self.context.hooks.memory_write(user_input, result, self.context)
        await self._complete_run(result)
        return result

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
