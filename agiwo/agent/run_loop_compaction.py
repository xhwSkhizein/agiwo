"""Compaction cycle mixin for RunLoopOrchestrator."""

from agiwo.agent.compaction import CompactResult
from agiwo.agent.models.run import TerminationReason
from agiwo.agent.run_loop_models import CompactionCycleResult
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class RunLoopCompactionOps:
    """Mixin: context compaction cycle during the run loop."""

    async def _run_compaction_cycle(self) -> CompactionCycleResult:
        """Run compaction cycle if needed."""
        from agiwo.agent import run_loop as run_loop_module  # noqa: PLC0415

        result: CompactResult = await run_loop_module.compact_if_needed(
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
