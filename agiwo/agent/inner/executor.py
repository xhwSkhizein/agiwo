"""
Agent Executor - LLM agent execution loop with tool calling.

This module implements the core agent execution loop:
- Streams LLM responses and accumulates tool calls
- Executes tools in parallel
- Tracks metrics and state
- Handles termination and summary generation
"""

import asyncio
import time
from collections.abc import Sequence

from agiwo.agent.compact_types import CompactMetadata
from agiwo.agent.hooks import AgentHooks
from agiwo.agent.input import ChannelContext
from agiwo.agent.inner.compaction.runtime import CompactionRuntime
from agiwo.agent.runtime_tools import RuntimeToolLike
from agiwo.agent.memory_types import MemoryRecord
from agiwo.llm.limits import (
    resolve_max_context_window,
    resolve_max_input_tokens_per_call,
)
from agiwo.agent.inner.event_emitter import EventEmitter
from agiwo.agent.inner.llm_handler import LLMStreamHandler
from agiwo.agent.inner.message_assembler import MessageAssembler
from agiwo.agent.inner.tool_runtime import ToolRuntime
from agiwo.agent.inner.summarizer import (
    DEFAULT_TERMINATION_USER_PROMPT,
    _format_termination_reason,
)
from agiwo.agent.inner.run_state import RunState
from agiwo.agent.options import AgentOptions
from agiwo.agent.runtime import (
    LLMCallContext,
    RunOutput,
    StepRecord,
    TerminationReason,
)
from agiwo.agent.scheduler_port import StepObserver
from agiwo.agent.storage.base import RunStepStorage
from agiwo.agent.storage.session import SessionStorage
from agiwo.config.settings import settings
from agiwo.llm.base import Model
from agiwo.llm.helper import parse_json_tool_args
from agiwo.agent.execution_context import ExecutionContext
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger


logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Agent Executor
# ═══════════════════════════════════════════════════════════════════════════


class AgentExecutor:
    """
    Agent execution loop coordinator.

    Responsibilities: LLM calls, tool execution, limit checks, termination summary.

    Components:
    - LLMStreamHandler: LLM streaming
    - ToolRuntime: Tool execution
    - Built-in limit checks and summary generation
    """

    SUMMARY_REASONS = frozenset(
        {
            TerminationReason.MAX_STEPS,
            TerminationReason.TIMEOUT,
            TerminationReason.MAX_RUN_COST,
            TerminationReason.ERROR_WITH_CONTEXT,
        }
    )

    def __init__(
        self,
        model: Model,
        tools: list[RuntimeToolLike],
        emitter: EventEmitter,
        options: AgentOptions | None = None,
        hooks: AgentHooks | None = None,
        run_step_storage: RunStepStorage | None = None,
        session_storage: SessionStorage | None = None,
        root_path: str | None = None,
        step_observers: Sequence[StepObserver] | None = None,
    ):
        self.model = model
        self.emitter = emitter
        self.options = options or AgentOptions()
        self.hooks = hooks or AgentHooks()
        self.run_step_storage = run_step_storage
        self.session_storage = session_storage
        self.root_path = root_path or settings.root_path
        self._step_observers = list(step_observers or [])
        self.max_context_window = resolve_max_context_window(model)
        self.max_input_tokens_per_call = resolve_max_input_tokens_per_call(
            self.options.max_input_tokens_per_call, model
        )

        self.llm_handler = LLMStreamHandler(model)
        self.tool_runtime = ToolRuntime(tools=tools)

        self._tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": tool.get_definition().name,
                    "description": tool.get_definition().description,
                    "parameters": tool.get_definition().parameters,
                },
            }
            for tool in tools
        ] if tools else None

        # Compactor (created only when session storage is enabled)
        self._compactor: CompactionRuntime | None = None
        if session_storage is not None:
            self._compactor = CompactionRuntime(
                llm_handler=self.llm_handler,
                emitter=emitter,
                session_storage=session_storage,
                compact_prompt=self.options.compact_prompt,
                root_path=self.root_path,
            )

    # ───────────────────────────────────────────────────────────────────
    # Public API
    # ───────────────────────────────────────────────────────────────────

    async def execute(
        self,
        system_prompt: str,
        user_step: StepRecord,
        context: ExecutionContext,
        *,
        memories: list[MemoryRecord] | None = None,
        before_run_hook_result: str | None = None,
        channel_context: ChannelContext | None = None,
        pending_tool_calls: list[dict] | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> RunOutput:
        """
        Execute agent loop.

        Args:
            system_prompt: System prompt for the agent.
            user_step: Current user step (history loaded internally).
            context: Execution context.
            memories: Retrieved memory records.
            before_run_hook_result: Result from before-run hook.
            channel_context: Channel context metadata.
            pending_tool_calls: Pending tool calls from previous run.
            abort_signal: Abort signal for cancellation.

        Returns:
            RunOutput with response and metrics.
        """
        # Query CompactMetadata and load history
        last_compact: CompactMetadata | None = None
        compact_start_seq = 0
        existing_steps: list[StepRecord] = []

        if self.session_storage is not None:
            last_compact = await self.session_storage.get_latest_compact_metadata(
                context.session_id, context.agent_id
            )
            if last_compact:
                compact_start_seq = last_compact.end_seq + 1

        # Load historical steps (filtered by compact metadata)
        # Note: user_step is already persisted by Agent, so it's included in the query result
        if self.run_step_storage is not None:
            existing_steps = await self.run_step_storage.get_steps(
                session_id=context.session_id,
                agent_id=context.agent_id,
                start_seq=compact_start_seq if compact_start_seq > 0 else None,
            )
        else:
            # No storage: use user_step directly
            existing_steps = [user_step]

        # Assemble messages from steps
        messages = MessageAssembler.assemble(
            system_prompt,
            existing_steps,
            memories,
            before_run_hook_result,
            channel_context=channel_context,
        )

        state = RunState(
            context=context,
            config=self.options,
            messages=messages,
            tool_schemas=self._tool_schemas,
            last_compact_metadata=last_compact,
            compact_start_seq=compact_start_seq,
        )

        try:
            await self._run_loop(state, pending_tool_calls, abort_signal)
        except asyncio.CancelledError:
            state.termination_reason = TerminationReason.CANCELLED
            logger.info("agent_execution_cancelled", run_id=state.context.run_id)
        except Exception as e:
            state.termination_reason = (
                TerminationReason.ERROR_WITH_CONTEXT
                if state.assistant_steps_count > 0
                else TerminationReason.ERROR
            )
            logger.error(
                "agent_execution_failed",
                run_id=state.context.run_id,
                error=str(e),
                error_type=type(e).__name__,
                steps_completed=state.steps_count,
                termination_reason=state.termination_reason,
                exc_info=True,
            )
        finally:
            await self._maybe_generate_summary(state, abort_signal)

        return state.build_output()

    # ───────────────────────────────────────────────────────────────────
    # Core Loop
    # ───────────────────────────────────────────────────────────────────

    async def _run_loop(
        self,
        state: RunState,
        pending_tool_calls: list[dict] | None,
        abort_signal: AbortSignal | None,
    ) -> None:
        if await self._resume_pending_tool_calls(state, pending_tool_calls, abort_signal):
            return

        while state.termination_reason is None:
            if not await self._run_cycle(state, abort_signal):
                return

    async def _resume_pending_tool_calls(
        self,
        state: RunState,
        pending_tool_calls: list[dict] | None,
        abort_signal: AbortSignal | None,
    ) -> bool:
        if not pending_tool_calls:
            return False

        await self._execute_tools(state, pending_tool_calls, abort_signal)
        return state.termination_reason is not None

    async def _run_cycle(
        self,
        state: RunState,
        abort_signal: AbortSignal | None,
    ) -> bool:
        reason = self._check_non_recoverable_limits(state)
        if reason is not None:
            state.termination_reason = reason
            return False

        if await self._handle_compaction(state, abort_signal):
            return state.termination_reason is None

        step, llm_context = await self._stream_next_step(state, abort_signal)
        if self._apply_post_llm_limits(state, step, llm_context):
            return False
        return await self._handle_step_tools(state, step, abort_signal)

    async def _stream_next_step(
        self,
        state: RunState,
        abort_signal: AbortSignal | None,
    ) -> tuple[StepRecord, LLMCallContext]:
        self._drain_steering_queue(state)

        state.current_step += 1
        if self.hooks.on_before_llm_call:
            modified = await self.hooks.on_before_llm_call(state.messages)
            if modified is not None:
                state.messages = modified

        step, llm_context = await self.llm_handler.stream_assistant_step(
            state, self.emitter.emit_step_delta, abort_signal
        )
        await self.emitter.emit_step_completed(step, llm=llm_context)
        state.track_step(step)
        state.add_token_cost(step.metrics.token_cost if step.metrics else 0.0)

        if self.hooks.on_after_llm_call:
            await self.hooks.on_after_llm_call(step)
        if self.hooks.on_step:
            await self.hooks.on_step(step)
        await self._notify_step_observers(step)
        return step, llm_context

    def _apply_post_llm_limits(
        self,
        state: RunState,
        step: StepRecord,
        llm_context: LLMCallContext,
    ) -> bool:
        reason = self._check_post_llm_limits(state, step, llm_context)
        if reason is None:
            return False
        state.termination_reason = reason
        return True

    async def _handle_step_tools(
        self,
        state: RunState,
        step: StepRecord,
        abort_signal: AbortSignal | None,
    ) -> bool:
        if step.tool_calls:
            await self._execute_tools(state, step.tool_calls, abort_signal)
            return state.termination_reason is None

        state.termination_reason = TerminationReason.COMPLETED
        return False

    async def _handle_compaction(
        self,
        state: RunState,
        abort_signal: AbortSignal | None,
    ) -> bool:
        if not await self._maybe_compact(state, abort_signal):
            return False

        logger.info(
            "compact_triggered",
            run_id=state.context.run_id,
            before_messages=len(state.messages),
        )
        if (
            self.options.max_run_cost is not None
            and state.token_cost >= self.options.max_run_cost
        ):
            state.termination_reason = TerminationReason.MAX_RUN_COST
        return True

    # ───────────────────────────────────────────────────────────────────
    # Helpers
    # ───────────────────────────────────────────────────────────────────

    def _drain_steering_queue(self, state: RunState) -> None:
        """Consume all pending steering messages and inject them into the current messages.

        Injection rules:
        - If the last message is role=user or role=tool: append <system-steering> tag to its content
        - Otherwise: append a new user message with the steering text
        """
        queue = state.context.steering_queue
        if queue is None or queue.empty():
            return

        parts: list[str] = []
        while not queue.empty():
            try:
                parts.append(str(queue.get_nowait()))
            except asyncio.QueueEmpty:
                break

        if not parts:
            return

        steering_text = "\n".join(parts)
        tag = f"\n\n<system-steering>{steering_text}</system-steering>"

        last_msg = state.messages[-1] if state.messages else None
        if last_msg and last_msg.get("role") in ("user", "tool"):
            content = last_msg.get("content", "")
            if isinstance(content, str):
                last_msg["content"] = content + tag
            elif isinstance(content, list):
                last_msg["content"].append({"type": "text", "text": tag})
            else:
                last_msg["content"] = tag
        else:
            state.messages.append({"role": "user", "content": steering_text})

    async def _execute_tools(
        self,
        state: RunState,
        tool_calls: list[dict],
        abort_signal: AbortSignal | None,
    ) -> None:
        # Before-tool hooks
        for tc in tool_calls:
            if self.hooks.on_before_tool_call:
                toolcall_id = tc.get("id", "unknown")
                fn: dict = tc.get("function", {})
                tool_name = fn.get("name", "unknown")
                args = parse_json_tool_args(fn.get("arguments", {}))
                modified = await self.hooks.on_before_tool_call(
                    toolcall_id, tool_name, args
                )
                if modified is not None:
                    tc["function"]["arguments"] = modified

        outcomes = await self.tool_runtime.execute_batch(
            tool_calls, context=state.context, abort_signal=abort_signal
        )

        for tc, outcome in zip(tool_calls, outcomes):
            result = outcome.result
            # After-tool hook
            if self.hooks.on_after_tool_call:
                fn = tc.get("function", {})

                await self.hooks.on_after_tool_call(
                    tc.get("id", ""),
                    fn.get("name", "unknown"),
                    parse_json_tool_args(fn.get("arguments", {})),
                    result,
                )

            seq = await state.next_sequence()
            step = StepRecord.tool(
                state.context,
                sequence=seq,
                tool_call_id=result.tool_call_id,
                name=result.tool_name,
                content=result.content,
                content_for_user=result.content_for_user,
            )
            await self.emitter.emit_step_completed(step)
            state.track_step(step)
            if self.hooks.on_step:
                await self.hooks.on_step(step)
            await self._notify_step_observers(step)

        for outcome in outcomes:
            if outcome.termination_reason is not None:
                state.termination_reason = outcome.termination_reason
                return

    def _check_non_recoverable_limits(self, state: RunState) -> TerminationReason | None:
        """Check limits that cannot be recovered by compact (steps, timeout)."""
        if state.current_step >= self.options.max_steps:
            logger.warning(
                "limit_hit_max_steps",
                current_step=state.current_step,
                max_steps=self.options.max_steps,
                run_id=state.context.run_id,
            )
            return TerminationReason.MAX_STEPS

        if self.options.run_timeout and state.elapsed > self.options.run_timeout:
            logger.warning(
                "limit_hit_timeout",
                elapsed=state.elapsed,
                run_timeout=self.options.run_timeout,
                run_id=state.context.run_id,
            )
            return TerminationReason.TIMEOUT

        if state.context.timeout_at and time.time() >= state.context.timeout_at:
            logger.warning(
                "limit_hit_context_timeout",
                timeout_at=state.context.timeout_at,
                run_id=state.context.run_id,
            )
            return TerminationReason.TIMEOUT

        return None

    def _check_post_llm_limits(
        self,
        state: RunState,
        step: StepRecord,
        llm_context: LLMCallContext,
    ) -> TerminationReason | None:
        """Unified post-LLM-call limit checks (input tokens, run cost, output limit)."""
        input_tokens = step.metrics.input_tokens if step.metrics else 0
        if input_tokens and input_tokens > self.max_input_tokens_per_call:
            logger.warning(
                "limit_hit_max_input_tokens_per_call",
                input_tokens=input_tokens,
                max_input_tokens_per_call=self.max_input_tokens_per_call,
                run_id=state.context.run_id,
            )
            return TerminationReason.MAX_INPUT_TOKENS_PER_CALL

        if self.options.max_run_cost is not None and state.token_cost >= self.options.max_run_cost:
            logger.warning(
                "limit_hit_max_run_cost",
                token_cost=state.token_cost,
                max_run_cost=self.options.max_run_cost,
                run_id=state.context.run_id,
            )
            return TerminationReason.MAX_RUN_COST

        finish_reason = llm_context.finish_reason
        if finish_reason and finish_reason.strip().lower() in {"length", "max_tokens"}:
            logger.warning(
                "limit_hit_max_output_tokens",
                finish_reason=finish_reason,
                run_id=state.context.run_id,
            )
            return TerminationReason.MAX_OUTPUT_TOKENS

        return None

    async def _maybe_generate_summary(
        self,
        state: RunState,
        abort_signal: AbortSignal | None,
    ) -> None:
        if not self.options.enable_termination_summary:
            return

        if state.termination_reason not in self.SUMMARY_REASONS:
            return

        # Build summary user prompt
        prompt_template = (
            self.options.termination_summary_prompt or DEFAULT_TERMINATION_USER_PROMPT
        )
        termination_reason_str = _format_termination_reason(state.termination_reason)
        user_prompt = (
            prompt_template % termination_reason_str
            if "%s" in prompt_template
            else prompt_template
        )

        # Record Summary UserStep (the summary request)
        user_seq = await state.next_sequence()
        summary_user_step = StepRecord.user(
            state.context,
            sequence=user_seq,
            content=user_prompt,
            name="summary_request",
        )
        await self.emitter.emit_step_completed(summary_user_step)
        state.track_step(summary_user_step, append_message=True)

        # Call LLM for summary
        step, llm_context = await self.llm_handler.stream_assistant_step(
            state,
            self.emitter.emit_step_delta,
            abort_signal,
            messages=state.messages,
            tools=None,
        )
        step.name = "summary"
        await self.emitter.emit_step_completed(step, llm=llm_context)
        state.track_step(step, append_message=False)
        state.add_token_cost(step.metrics.token_cost if step.metrics else 0.0)

        logger.info(
            "summary_generated",
            tokens=step.metrics.total_tokens if step.metrics else 0,
        )

    async def _notify_step_observers(self, step: StepRecord) -> None:
        for observer in self._step_observers:
            await observer(step)

    # ───────────────────────────────────────────────────────────────────
    # Compact
    # ───────────────────────────────────────────────────────────────────

    async def _maybe_compact(
        self,
        state: RunState,
        abort_signal: AbortSignal | None,
    ) -> bool:
        """
        Check and perform compact if needed.

        Returns:
            True if compact was performed, False otherwise.
        """
        if self._compactor is None:
            return False

        if not self._compactor.should_compact(
            state.messages, self.max_context_window
        ):
            return False

        # Retry logic
        retry_count = settings.compact_retry_count
        last_error: Exception | None = None

        for attempt in range(retry_count + 1):
            try:
                result = await self._compactor.compact(state, abort_signal)
                if result.step is not None:
                    step_cost = result.step.metrics.token_cost if result.step.metrics else 0.0
                    state.add_token_cost(step_cost)

                # Update state with compacted messages
                state.messages = result.compacted_messages

                logger.info(
                    "compact_success",
                    run_id=state.context.run_id,
                    start_seq=result.metadata.start_seq,
                    end_seq=result.metadata.end_seq,
                    before_tokens=result.metadata.before_token_estimate,
                    after_tokens=result.metadata.after_token_estimate,
                    attempt=attempt + 1,
                )
                return True

            except Exception as error:  # noqa: BLE001 - compaction retries guard the runtime boundary
                last_error = error
                logger.warning(
                    "compact_attempt_failed",
                    run_id=state.context.run_id,
                    attempt=attempt + 1,
                    max_attempts=retry_count + 1,
                    error=str(error),
                )
                if attempt < retry_count:
                    continue

        # All retries failed, log and continue without compact
        logger.error(
            "compact_failed_all_retries",
            run_id=state.context.run_id,
            error=str(last_error),
        )
        return False
