"""Single-run execution engine — the core run loop."""

import asyncio
from typing import NamedTuple

from agiwo.agent.compaction import CompactResult, compact_if_needed
from agiwo.agent.run_bootstrap import prepare_run_context
from agiwo.agent.run_tool_batch import execute_tool_batch_cycle
from agiwo.agent.runtime.context import RunContext


class CompactionCycleResult(NamedTuple):
    """Result of a compaction cycle."""

    compact_start_seq: int
    should_continue: bool


from agiwo.agent.models.config import AgentOptions
from agiwo.agent.hooks import HookRegistration, HookRegistry
from agiwo.agent.models.input import UserInput
from agiwo.agent.llm_caller import stream_assistant_step
from agiwo.agent.prompt import apply_steering_messages
from agiwo.agent.models.run import (
    RunMetrics,
    RunOutput,
    TerminationReason,
)
from agiwo.agent.models.step import (
    LLMCallContext,
    StepView,
)
from agiwo.agent.runtime.context import RunRuntime
from agiwo.agent.runtime.step_committer import commit_step
from agiwo.agent.runtime.state_writer import RunStateWriter
from agiwo.agent.termination.limits import (
    check_non_recoverable_limits,
    check_post_llm_limits,
)
from agiwo.agent.termination.summarizer import maybe_generate_termination_summary
from agiwo.tool.base import BaseTool
from agiwo.config.settings import settings
from agiwo.llm.base import Model
from agiwo.llm.limits import (
    resolve_max_context_window,
    resolve_max_input_tokens_per_call,
)
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

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

    async def _project_entries(self, entries: list[object]) -> None:
        await self.context.session_runtime.project_run_log_entries(
            entries,
            run_id=self.context.run_id,
            agent_id=self.context.agent_id,
            parent_run_id=self.context.parent_run_id,
            depth=self.context.depth,
        )

    async def execute_run(
        self,
        user_input: UserInput,
        system_prompt: str,
        pending_tool_calls: list[dict] | None = None,
    ) -> RunOutput:
        """Execute a single agent run with the orchestrator."""
        try:
            # Start the run
            await self._start_run(user_input)

            # Prepare run context after the run has a persisted identity.
            bootstrap = await prepare_run_context(
                context=self.context,
                runtime=self.runtime,
                user_input=user_input,
                system_prompt=system_prompt,
                writer=self.writer,
            )

            # Commit user step
            await commit_step(
                self.context,
                bootstrap.user_step,
                append_message=False,
                track_state=False,
            )

            # Execute the main loop
            self.runtime.compact_start_seq = bootstrap.compact_start_seq
            await self._run_loop(pending_tool_calls=pending_tool_calls)

            # Finalize and return result
            return await self._finalize_run(
                user_input,
            )
        except Exception as error:
            await self._fail_run(error)
            raise

    async def _start_run(self, user_input: UserInput) -> None:
        """Initialize and start the run."""
        entries = await self.writer.start_run(user_input)
        await self._project_entries(entries)

    async def _complete_run(self, result: RunOutput) -> None:
        """Complete the run successfully."""
        entries = await self.writer.finish_run(result)
        await self._project_entries(entries)

    async def _fail_run(self, error: Exception) -> None:
        """Handle run failure."""
        entries = await self.writer.fail_run(error)
        await self._project_entries(entries)

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
        entries = await self.writer.record_termination_decided(
            termination_reason=reason,
            phase=phase,
            source=source,
        )
        await self._project_entries(entries)

    async def _finalize_run(self, user_input: UserInput) -> RunOutput:
        """Generate summary, build output, and complete the run."""
        await maybe_generate_termination_summary(
            state=self.context,
            options=self.runtime.config,
            model=self.runtime.model,
            abort_signal=self.runtime.abort_signal,
        )
        result = self._build_output()
        await self.context.hooks.after_run(result, self.context)
        if result.response is not None:
            await self.context.hooks.memory_write(user_input, result, self.context)
        await self._complete_run(result)
        return result

    async def _run_loop(
        self,
        pending_tool_calls: list[dict] | None,
    ) -> None:
        """Main run loop."""
        try:
            if pending_tool_calls:
                terminated = await self._execute_tool_calls(
                    tool_calls=pending_tool_calls,
                )
                if terminated:
                    return

            while not self.context.is_terminal:
                should_stop = await self._run_loop_iteration()
                if should_stop:
                    return
        except asyncio.CancelledError:
            await self._set_termination_reason(
                TerminationReason.CANCELLED,
                phase="run_loop",
                source="cancelled_error",
            )
            logger.info("agent_execution_cancelled", run_id=self.context.run_id)
        except Exception:
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
            raise

    async def _run_loop_iteration(self) -> bool:
        """Execute one iteration of the run loop. Returns True if the loop should stop."""
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
            return True

        result = await self._run_compaction_cycle()
        compact_start_seq = result.compact_start_seq
        should_continue = result.should_continue
        self.runtime.compact_start_seq = compact_start_seq
        if should_continue or self.context.is_terminal:
            return self.context.is_terminal

        self.context.ledger.steps.current += 1
        step, llm_context = await self._run_assistant_turn()
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
            compact_prompt=self.runtime.compact_prompt,
            compact_start_seq=self.runtime.compact_start_seq,
            root_path=self.runtime.root_path,
        )
        if result.failed:
            failure_count = self.writer.next_compaction_failure_attempt()
            err = result.error or ""
            terminal = failure_count >= 3
            entries = await self.writer.record_compaction_failed(
                error=err,
                attempt=failure_count,
                max_attempts=3,
                terminal=terminal,
            )
            await self._project_entries(entries)
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
            rebuilt_entries = await self.writer.rebuild_messages(
                reason="before_llm",
                messages=request_messages,
            )
            await self._project_entries(rebuilt_entries)
            self.context.session_runtime.ack_pending_steer_inputs(len(pending_steer))
        request_messages = self.context.snapshot_messages()
        modified = await self.context.hooks.before_llm_call(
            request_messages, self.context
        )
        if modified is not None:
            request_messages = modified
        if request_messages != self.context.snapshot_messages():
            rebuilt_entries = await self.writer.rebuild_messages(
                reason="before_llm",
                messages=request_messages,
            )
            await self._project_entries(rebuilt_entries)
        entries = await self.writer.record_llm_call_started(
            messages=self.context.snapshot_messages(),
            tools=self.context.copy_tool_schemas(),
        )
        await self._project_entries(entries)

        step, llm_context = await stream_assistant_step(
            self.runtime.model,
            self.context,
            self.runtime.abort_signal,
        )
        await commit_step(self.context, step, llm=llm_context)
        entries = await self.writer.record_llm_call_completed(
            step=step, llm=llm_context
        )
        await self._project_entries(entries)

        await self.context.hooks.after_llm_call(step, self.context)
        return step, llm_context

    async def _handle_assistant_turn_result(
        self,
        step: StepView,
        llm_context: LLMCallContext,
    ) -> bool:
        """Handle the result of an assistant turn."""
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
            return True

        if not step.tool_calls:
            await self._set_termination_reason(
                TerminationReason.COMPLETED,
                phase="post_llm",
                source="assistant_completed_without_tools",
            )
            return True

        return await self._execute_tool_calls(tool_calls=step.tool_calls)

    async def _execute_tool_calls(
        self,
        tool_calls: list[dict[str, object]],
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
            set_termination_reason=_set_tool_termination,
        )


async def execute_run(
    user_input: UserInput,
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
    )

    orchestrator = RunLoopOrchestrator(context, runtime)
    return await orchestrator.execute_run(
        user_input,
        system_prompt,
        pending_tool_calls,
    )


RunEngine = RunLoopOrchestrator


__all__ = ["execute_run", "RunEngine", "RunLoopOrchestrator"]
