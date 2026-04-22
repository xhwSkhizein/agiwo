"""Single-run execution engine — the core run loop."""

import asyncio
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
from agiwo.agent.hooks import HookRegistration, HookRegistry
from agiwo.agent.models.input import UserInput, UserMessage
from agiwo.agent.llm_caller import stream_assistant_step
from agiwo.agent.models.run import MemoryRecord
from agiwo.agent.prompt import apply_steering_messages, assemble_run_messages
from agiwo.agent.models.run import (
    RunMetrics,
    RunOutput,
    TerminationReason,
)
from agiwo.agent.models.step import (
    LLMCallContext,
    StepView,
)
from agiwo.agent.runtime.context import RunContext, RunRuntime
from agiwo.agent.runtime.state_ops import (
    record_compaction_metadata,
    replace_messages,
    set_tool_schemas,
    set_termination_reason,
)
from agiwo.agent.runtime.step_committer import commit_step
from agiwo.agent.runtime.state_writer import (
    build_context_assembled_entry,
    build_llm_call_completed_entry,
    build_llm_call_started_entry,
    build_messages_rebuilt_entry,
    build_run_failed_entry,
    build_run_finished_entry,
    build_run_started_entry,
    build_retrospect_applied_entry,
    build_termination_decided_entry,
)
from agiwo.agent.models.stream import (
    MessagesRebuiltEvent,
    RetrospectAppliedEvent,
    RunCompletedEvent,
    RunFailedEvent,
    RunStartedEvent,
    TerminationDecidedEvent,
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


class RunEngine:
    """RunEngine is the single-run execution owner."""

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
        user_step, compact_start_seq = await self._prepare_run_context(
            user_input,
            system_prompt=system_prompt,
        )

        # Start the run
        await self._start_run(user_input)

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
            )
        except Exception as error:
            await self._fail_run(error)
            raise

    async def _start_run(self, user_input: UserInput) -> None:
        """Initialize and start the run."""
        if self.context.session_runtime.trace_runtime is not None:
            self.context.session_runtime.trace_runtime.on_run_started(
                run_id=self.context.run_id,
                agent_id=self.context.agent_id,
                session_id=self.context.session_id,
                parent_run_id=self.context.parent_run_id,
            )
        await self.context.session_runtime.append_run_log_entries(
            [
                build_run_started_entry(
                    self.context,
                    sequence=await self.context.session_runtime.allocate_sequence(),
                    user_input=user_input,
                )
            ]
        )
        await self.context.session_runtime.publish(
            RunStartedEvent.from_context(self.context)
        )

    async def _complete_run(self, result: RunOutput) -> None:
        """Complete the run successfully."""
        if self.context.session_runtime.trace_runtime is not None:
            await self.context.session_runtime.trace_runtime.on_run_completed(
                result,
                run_id=self.context.run_id,
            )
        await self.context.session_runtime.append_run_log_entries(
            [
                build_run_finished_entry(
                    self.context,
                    sequence=await self.context.session_runtime.allocate_sequence(),
                    result=result,
                )
            ]
        )
        await self.context.session_runtime.publish(
            RunCompletedEvent.from_context(
                self.context,
                response=result.response,
                metrics=result.metrics,
                termination_reason=result.termination_reason,
            ),
        )

    async def _fail_run(self, error: Exception) -> None:
        """Handle run failure."""
        if self.context.session_runtime.trace_runtime is not None:
            await self.context.session_runtime.trace_runtime.on_run_failed(
                error,
                run_id=self.context.run_id,
            )
        await self.context.session_runtime.append_run_log_entries(
            [
                build_run_failed_entry(
                    self.context,
                    sequence=await self.context.session_runtime.allocate_sequence(),
                    error=error,
                )
            ]
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

    async def _set_termination_reason(
        self,
        reason: TerminationReason,
        *,
        phase: str,
        source: str,
    ) -> None:
        if self.context.ledger.termination_reason == reason:
            return
        set_termination_reason(self.context, reason)
        entry = build_termination_decided_entry(
            self.context,
            sequence=await self.context.session_runtime.allocate_sequence(),
            termination_reason=reason,
            phase=phase,
            source=source,
        )
        await self.context.session_runtime.append_run_log_entries([entry])
        await self.context.session_runtime.publish(
            TerminationDecidedEvent.from_context(
                self.context,
                termination_reason=reason,
                phase=phase,
                source=source,
            )
        )

    async def _prepare_run_context(
        self,
        user_input: UserInput,
        system_prompt: str,
    ) -> tuple[StepView, int]:
        """Build all state needed before the main loop starts."""
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
        before_run_hook_result = await self.context.hooks.before_run(
            user_input, self.context
        )
        memories: list[MemoryRecord] = []
        if user_input is not None:
            memories = await self.context.hooks.memory_retrieve(
                user_input, self.context
            )

        run_log_storage = self.context.session_runtime.run_log_storage
        user_step = StepView.user(
            self.context,
            sequence=await self.context.session_runtime.allocate_sequence(),
            user_input=user_input,
        )
        self.context.ledger.run_start_seq = user_step.sequence
        last_compact = await self.context.session_runtime.get_latest_compact_metadata(
            self.context.agent_id
        )
        compact_start_seq = last_compact.end_seq + 1 if last_compact is not None else 0
        existing_steps = await run_log_storage.list_step_views(
            session_id=self.context.session_id,
            agent_id=self.context.agent_id,
            start_seq=compact_start_seq if compact_start_seq > 0 else None,
        )
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
        await self.context.session_runtime.append_run_log_entries(
            [
                build_context_assembled_entry(
                    self.context,
                    sequence=await self.context.session_runtime.allocate_sequence(),
                    messages=self.context.snapshot_messages(),
                    memory_count=len(memories),
                )
            ]
        )
        set_tool_schemas(self.context, tool_schemas)
        record_compaction_metadata(self.context, last_compact)

        return user_step, compact_start_seq

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
            failure_count = increment_compaction_failure(self.context)
            err = result.error or ""
            await self.context.hooks.compaction_failed(
                self.context.run_id, err, failure_count, self.context
            )
            logger.warning(
                "compaction_failed",
                run_id=self.context.run_id,
                error=err,
                failure_count=failure_count,
            )
            if failure_count >= 3:
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
        replace_messages(
            self.context,
            apply_steering_messages(
                self.context.snapshot_messages(),
                self.context.session_runtime.steering_queue,
            ),
        )
        modified = await self.context.hooks.before_llm_call(
            self.context.snapshot_messages(),
            self.context,
        )
        if modified is not None:
            replace_messages(self.context, modified)
        await self.context.session_runtime.append_run_log_entries(
            [
                build_llm_call_started_entry(
                    self.context,
                    sequence=await self.context.session_runtime.allocate_sequence(),
                    messages=self.context.snapshot_messages(),
                    tools=self.context.copy_tool_schemas(),
                )
            ]
        )

        step, llm_context = await stream_assistant_step(
            self.runtime.model,
            self.context,
            self.runtime.abort_signal,
        )
        await commit_step(self.context, step, llm=llm_context)
        await self.context.session_runtime.append_run_log_entries(
            [
                build_llm_call_completed_entry(
                    self.context,
                    sequence=await self.context.session_runtime.allocate_sequence(),
                    step=step,
                    llm=llm_context,
                )
            ]
        )

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

            await self.context.hooks.after_tool_call(
                call_id,
                result.tool_name,
                result.input_args or {},
                result,
                self.context,
            )

            content = batch.process_result(result)

            tool_step = StepView.tool(
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
                await self._set_termination_reason(
                    result.termination_reason,
                    phase="tool_result",
                    source=result.tool_name,
                )
                terminated = True

        outcome = await batch.finalize()
        if outcome.applied:
            replace_messages(self.context, outcome.messages)
            rebuilt_entry = build_messages_rebuilt_entry(
                self.context,
                sequence=await self.context.session_runtime.allocate_sequence(),
                reason="retrospect",
                messages=self.context.snapshot_messages(),
            )
            retrospect_entry = build_retrospect_applied_entry(
                self.context,
                sequence=await self.context.session_runtime.allocate_sequence(),
                affected_sequences=outcome.affected_sequences,
                affected_step_ids=outcome.affected_step_ids,
                feedback=outcome.feedback,
                replacement=outcome.replacement,
            )
            await self.context.session_runtime.append_run_log_entries(
                [rebuilt_entry, retrospect_entry]
            )
            await self.context.session_runtime.publish(
                MessagesRebuiltEvent.from_context(
                    self.context,
                    reason="retrospect",
                    message_count=len(self.context.snapshot_messages()),
                )
            )
            await self.context.session_runtime.publish(
                RetrospectAppliedEvent.from_context(
                    self.context,
                    affected_sequences=outcome.affected_sequences,
                    affected_step_ids=outcome.affected_step_ids,
                    feedback=outcome.feedback,
                    replacement=outcome.replacement,
                    trigger=None,
                )
            )

        return terminated or self.context.is_terminal


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

    orchestrator = RunEngine(context, runtime)
    return await orchestrator.execute_run(
        user_input,
        system_prompt,
        pending_tool_calls,
    )


RunLoopOrchestrator = RunEngine


__all__ = ["execute_run", "RunEngine", "RunLoopOrchestrator"]
