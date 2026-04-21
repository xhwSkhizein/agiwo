"""Single-run execution engine — the core run loop."""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, NamedTuple

from agiwo.agent.compaction import CompactResult, compact_if_needed
from agiwo.agent.runtime.context import RunContext


class CompactionCycleResult(NamedTuple):
    """Result of a compaction cycle."""

    compact_start_seq: int
    should_continue: bool


def increment_compaction_failure(context: RunContext) -> int:
    """Increment compaction failure count and return the new count."""
    context.ledger.compaction.failure_count += 1
    return context.ledger.compaction.failure_count


from agiwo.agent.models.config import AgentOptions
from agiwo.agent.hooks import AgentHooks
from agiwo.agent.models.input import UserInput, UserMessage
from agiwo.agent.llm_caller import stream_assistant_step
from agiwo.agent.models.run import MemoryRecord
from agiwo.agent.prompt import apply_steering_messages, assemble_run_messages
from agiwo.agent.models.run import (
    Run,
    RunMetrics,
    RunOutput,
    RunStatus,
    TerminationReason,
)
from agiwo.agent.models.step import (
    LLMCallContext,
    StepRecord,
)
from agiwo.agent.runtime.context import RunContext, RunRuntime
from agiwo.agent.runtime.state_ops import (
    record_compaction_metadata,
    replace_messages,
    set_tool_schemas,
    set_termination_reason,
)
from agiwo.agent.runtime.step_committer import commit_step
from agiwo.agent.models.stream import (
    RunCompletedEvent,
    RunFailedEvent,
    RunStartedEvent,
)
from agiwo.agent.termination.limits import (
    check_non_recoverable_limits,
    check_post_llm_limits,
)
from agiwo.agent.termination.summarizer import maybe_generate_termination_summary
from agiwo.agent.tool_executor import execute_tool_batch
from agiwo.tool.base import BaseTool
from agiwo.config.settings import settings
from agiwo.llm.base import Model
from agiwo.llm.limits import (
    resolve_max_context_window,
    resolve_max_input_tokens_per_call,
)
from agiwo.utils.abort_signal import AbortSignal
from agiwo.agent.retrospect import RetrospectBatch
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class RunLoopOrchestrator:
    """Orchestrates the run loop execution with encapsulated state."""

    def __init__(
        self,
        context: RunContext,
        runtime: RunRuntime,
    ):
        self.context = context
        self.runtime = runtime

    async def execute_run(
        self,
        user_input: UserInput,
        system_prompt: str,
        pending_tool_calls: list[dict] | None = None,
    ) -> RunOutput:
        """Execute a single agent run with the orchestrator."""
        # Prepare run context
        run, user_step, compact_start_seq = await self._prepare_run_context(
            user_input,
            system_prompt=system_prompt,
        )

        # Start the run
        await self._start_run(run)

        try:
            # Commit user step
            await commit_step(
                self.context,
                user_step,
                append_message=False,
                track_state=False,
            )

            # Execute the main loop
            self.runtime.compact_start_seq = compact_start_seq
            await self._run_loop(pending_tool_calls=pending_tool_calls)

            # Finalize and return result
            return await self._finalize_run(
                user_input,
                run,
            )
        except Exception as error:
            await self._fail_run(run, error)
            raise

    async def _start_run(self, run: Run) -> None:
        """Initialize and start the run."""
        run.trace_id = self.context.trace_id
        if self.context.session_runtime.trace_runtime is not None:
            run.trace_id = self.context.session_runtime.trace_runtime.on_run_started(
                run
            )
        await self.context.session_runtime.save_run(run)
        await self.context.session_runtime.publish(
            RunStartedEvent.from_context(self.context)
        )

    async def _complete_run(self, run: Run, result: RunOutput) -> None:
        """Complete the run successfully."""
        run.status = (
            RunStatus.CANCELLED
            if result.termination_reason == TerminationReason.CANCELLED
            else RunStatus.COMPLETED
        )
        run.response_content = result.response
        now = datetime.now(timezone.utc)
        run.updated_at = now
        run.metrics.end_at = now.timestamp()
        if result.metrics is not None:
            preserved_start_at = run.metrics.start_at
            preserved_end_at = run.metrics.end_at
            run.metrics = result.metrics
            run.metrics.start_at = preserved_start_at
            run.metrics.end_at = preserved_end_at
        await self.context.session_runtime.save_run(run)
        if self.context.session_runtime.trace_runtime is not None:
            await self.context.session_runtime.trace_runtime.on_run_completed(
                result,
                run_id=run.id,
            )
        await self.context.session_runtime.publish(
            RunCompletedEvent.from_context(
                self.context,
                response=result.response,
                metrics=result.metrics,
                termination_reason=result.termination_reason,
            ),
        )

    async def _fail_run(self, run: Run, error: Exception) -> None:
        """Handle run failure."""
        run.status = RunStatus.FAILED
        now = datetime.now(timezone.utc)
        run.updated_at = now
        run.metrics.end_at = now.timestamp()
        await self.context.session_runtime.save_run(run)
        if self.context.session_runtime.trace_runtime is not None:
            await self.context.session_runtime.trace_runtime.on_run_failed(
                error,
                run_id=run.id,
            )
        await self.context.session_runtime.publish(
            RunFailedEvent.from_context(self.context, error=str(error)),
        )

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

    async def _prepare_run_context(
        self,
        user_input: UserInput,
        system_prompt: str,
    ) -> tuple[Run, StepRecord, int]:
        """Build all state needed before the main loop starts. Returns (run, user_step, compact_start_seq)."""
        tools_map = self.runtime.tools_map
        tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": tool.get_definition().name,
                    "description": tool.get_definition().description,
                    "parameters": tool.get_definition().parameters,
                },
            }
            for tool in tools_map.values()
        ] or None

        before_run_hook_result = None
        if self.context.hooks.on_before_run is not None:
            before_run_hook_result = await self.context.hooks.on_before_run(
                user_input, self.context
            )
        memories: list[MemoryRecord] = []
        if self.context.hooks.on_memory_retrieve is not None and user_input is not None:
            memories = await self.context.hooks.on_memory_retrieve(
                user_input, self.context
            )

        run_step_storage = self.context.session_runtime.run_step_storage
        user_step = StepRecord.user(
            self.context,
            sequence=await self.context.session_runtime.allocate_sequence(),
            user_input=user_input,
        )
        self.context.ledger.run_start_seq = user_step.sequence
        last_compact = await run_step_storage.get_latest_compact_metadata(
            self.context.session_id,
            self.context.agent_id,
        )
        compact_start_seq = last_compact.end_seq + 1 if last_compact is not None else 0
        existing_steps = await run_step_storage.get_steps(
            session_id=self.context.session_id,
            agent_id=self.context.agent_id,
            start_seq=compact_start_seq if compact_start_seq > 0 else None,
        )
        if all(step.id != user_step.id for step in existing_steps):
            existing_steps.append(user_step)
            existing_steps.sort(key=lambda step: step.sequence)
        user_message = UserMessage.from_value(user_input)
        replace_messages(
            self.context,
            assemble_run_messages(
                system_prompt,
                existing_steps,
                memories,
                before_run_hook_result,
                channel_context=user_message.context,
            ),
        )
        set_tool_schemas(self.context, tool_schemas)
        record_compaction_metadata(self.context, last_compact)

        run = Run(
            id=self.context.run_id,
            agent_id=self.context.agent_id,
            session_id=self.context.session_id,
            user_input=user_input,
            status=RunStatus.RUNNING,
            parent_run_id=self.context.parent_run_id,
        )
        run.metrics.start_at = time.time()

        return run, user_step, compact_start_seq

    async def _finalize_run(self, user_input: UserInput, run: Run) -> RunOutput:
        """Generate summary, build output, and complete the run."""
        await maybe_generate_termination_summary(
            state=self.context,
            options=self.runtime.config,
            model=self.runtime.model,
            abort_signal=self.runtime.abort_signal,
        )
        result = self._build_output()
        if self.context.hooks.on_after_run:
            await self.context.hooks.on_after_run(result, self.context)
        if self.context.hooks.on_memory_write and result.response is not None:
            await self.context.hooks.on_memory_write(user_input, result, self.context)
        await self._complete_run(run, result)
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
            set_termination_reason(self.context, TerminationReason.CANCELLED)
            logger.info("agent_execution_cancelled", run_id=self.context.run_id)
        except Exception:
            set_termination_reason(
                self.context,
                TerminationReason.ERROR_WITH_CONTEXT
                if self.context.ledger.steps.assistant > 0
                else TerminationReason.ERROR,
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
            set_termination_reason(self.context, reason)
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
            failure_count = increment_compaction_failure(self.context)
            err = result.error or ""
            if self.context.hooks.on_compaction_failed:
                await self.context.hooks.on_compaction_failed(
                    self.context.run_id, err, failure_count
                )
            logger.warning(
                "compaction_failed",
                run_id=self.context.run_id,
                error=err,
                failure_count=failure_count,
            )
            if failure_count >= 3:
                set_termination_reason(
                    self.context, TerminationReason.MAX_INPUT_TOKENS_PER_CALL
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
            set_termination_reason(self.context, TerminationReason.MAX_RUN_COST)
            return CompactionCycleResult(new_start_seq, True)

    async def _run_assistant_turn(self) -> tuple[StepRecord, LLMCallContext]:
        """Execute an assistant turn (LLM call)."""
        replace_messages(
            self.context,
            apply_steering_messages(
                self.context.snapshot_messages(),
                self.context.session_runtime.steering_queue,
            ),
        )
        if self.context.hooks.on_before_llm_call:
            modified = await self.context.hooks.on_before_llm_call(
                self.context.snapshot_messages()
            )
            if modified is not None:
                replace_messages(self.context, modified)

        step, llm_context = await stream_assistant_step(
            self.runtime.model,
            self.context,
            self.runtime.abort_signal,
        )
        await commit_step(self.context, step, llm=llm_context)

        if self.context.hooks.on_after_llm_call:
            await self.context.hooks.on_after_llm_call(step)
        return step, llm_context

    async def _handle_assistant_turn_result(
        self,
        step: StepRecord,
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
            set_termination_reason(self.context, reason)
            return True

        if not step.tool_calls:
            set_termination_reason(self.context, TerminationReason.COMPLETED)
            return True

        return await self._execute_tool_calls(tool_calls=step.tool_calls)

    async def _execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> bool:
        """Execute a batch of tool calls."""
        tool_results = await execute_tool_batch(
            tool_calls,
            tools_map=self.runtime.tools_map,
            context=self.context,
            abort_signal=self.runtime.abort_signal,
        )
        terminated = False
        batch = RetrospectBatch(self.context, self.runtime.tools_map)

        for result in tool_results:
            call_id = result.tool_call_id or ""

            if self.context.hooks.on_after_tool_call:
                await self.context.hooks.on_after_tool_call(
                    call_id,
                    result.tool_name,
                    result.input_args or {},
                    result,
                )

            content = batch.process_result(result)

            tool_step = StepRecord.tool(
                self.context,
                sequence=await self.context.session_runtime.allocate_sequence(),
                tool_call_id=call_id,
                name=result.tool_name,
                content=content,
                content_for_user=result.content_for_user,
                is_error=not result.is_success,
            )
            batch.register_step(call_id, tool_step.id, tool_step.sequence)
            await commit_step(self.context, tool_step)

            if not terminated and result.termination_reason is not None:
                set_termination_reason(self.context, result.termination_reason)
                terminated = True

        outcome = await batch.finalize()
        if outcome.applied:
            replace_messages(self.context, outcome.messages)

        return terminated or self.context.is_terminal


async def execute_run(
    user_input: UserInput,
    *,
    context: RunContext,
    system_prompt: str,
    model: Model,
    tools: tuple[BaseTool, ...],
    options: AgentOptions | None = None,
    hooks: AgentHooks | None = None,
    pending_tool_calls: list[dict] | None = None,
    abort_signal: AbortSignal | None = None,
    root_path: str | None = None,
) -> RunOutput:
    """Execute a single agent run — the core entry point."""
    options = options or AgentOptions()
    hooks = hooks or AgentHooks()
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


__all__ = ["execute_run", "RunLoopOrchestrator"]
