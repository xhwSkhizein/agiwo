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

from agiwo.agent.inner.event_emitter import EventEmitter
from agiwo.agent.inner.llm_handler import LLMStreamHandler
from agiwo.agent.inner.summarizer import build_termination_messages
from agiwo.agent.inner.run_state import RunState
from agiwo.agent.schema import RunOutput, StepRecord, TerminationReason
from agiwo.agent.options import AgentOptions
from agiwo.agent.hooks import AgentHooks
from agiwo.llm.base import Model
from agiwo.llm.helper import parse_json_tool_args
from agiwo.utils.abort_signal import AbortSignal
from agiwo.agent.execution_context import ExecutionContext
from agiwo.tool.base import BaseTool
from agiwo.tool.executor import ToolExecutor
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
    - ToolExecutor: Tool execution
    - Built-in limit checks and summary generation
    """

    SUMMARY_REASONS = frozenset(
        {
            TerminationReason.MAX_STEPS,
            TerminationReason.TIMEOUT,
            TerminationReason.ERROR_WITH_CONTEXT,
        }
    )

    def __init__(
        self,
        model: Model,
        tools: list[BaseTool],
        emitter: EventEmitter,
        options: AgentOptions | None = None,
        hooks: AgentHooks | None = None,
    ):
        self.model = model
        self.emitter = emitter
        self.options = options or AgentOptions()
        self.hooks = hooks or AgentHooks()
        self.cache_hit_price = float(getattr(model, "cache_hit_price", 0.0) or 0.0)
        self.input_price = float(getattr(model, "input_price", 0.0) or 0.0)
        self.output_price = float(getattr(model, "output_price", 0.0) or 0.0)

        self.llm_handler = LLMStreamHandler(model)
        self.tool_executor = ToolExecutor(tools=tools)

        self._tool_schemas = [t.to_openai_schema() for t in tools] if tools else None

    # ───────────────────────────────────────────────────────────────────
    # Public API
    # ───────────────────────────────────────────────────────────────────

    async def execute(
        self,
        messages: list[dict],
        context: ExecutionContext,
        *,
        pending_tool_calls: list[dict] | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> RunOutput:
        state = RunState(
            context=context,
            config=self.options,
            messages=messages,
            tool_schemas=self._tool_schemas,
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
        # Resume pending tools from previous run
        if pending_tool_calls:
            await self._execute_tools(state, pending_tool_calls, abort_signal)

        # Main agent loop
        while True:
            if reason := self._check_pre_limits(state):
                state.termination_reason = reason
                break

            # LLM call (with hooks)
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
            state.add_token_cost(self._estimate_token_cost(step))

            if self.hooks.on_after_llm_call:
                await self.hooks.on_after_llm_call(step)
            if self.hooks.on_step:
                await self.hooks.on_step(step)

            had_tool_calls = bool(step.tool_calls)
            if had_tool_calls:
                await self._execute_tools(state, step.tool_calls or [], abort_signal)
                if state.termination_reason is not None:
                    return

            if reason := self._check_post_limits(state, llm_context.finish_reason, step):
                state.termination_reason = reason
                return

            if not had_tool_calls:
                state.termination_reason = TerminationReason.COMPLETED
                return  # Normal completion

    # ───────────────────────────────────────────────────────────────────
    # Helpers
    # ───────────────────────────────────────────────────────────────────

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

        results = await self.tool_executor.execute_batch(
            tool_calls, context=state.context, abort_signal=abort_signal
        )

        for tc, result in zip(tool_calls, results):
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

        for result in results:
            if result.termination_reason is not None:
                state.termination_reason = result.termination_reason
                return

    def _check_pre_limits(self, state: RunState) -> TerminationReason | None:
        """Check limits before making a new LLM request."""
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

        if (
            self.options.max_tokens_per_run is not None
            and state.total_tokens >= self.options.max_tokens_per_run
        ):
            logger.warning(
                "limit_hit_max_tokens_per_run",
                total_tokens=state.total_tokens,
                max_tokens=self.options.max_tokens_per_run,
                run_id=state.context.run_id,
            )
            return TerminationReason.MAX_TOKENS_PER_RUN

        if (
            self.options.max_run_token_cost is not None
            and state.token_cost >= self.options.max_run_token_cost
        ):
            logger.warning(
                "limit_hit_max_run_token_cost",
                token_cost=state.token_cost,
                max_cost=self.options.max_run_token_cost,
                run_id=state.context.run_id,
            )
            return TerminationReason.MAX_RUN_TOKEN_COST

        return None

    def _check_post_limits(
        self,
        state: RunState,
        finish_reason: str | None,
        step: StepRecord,
    ) -> TerminationReason | None:
        """Check limits after one LLM step and any tool execution for that step."""
        if self._is_model_output_limit_hit(finish_reason):
            logger.warning(
                "limit_hit_max_output_tokens_per_call",
                finish_reason=finish_reason,
                run_id=state.context.run_id,
            )
            return TerminationReason.MAX_OUTPUT_TOKENS_PER_CALL

        if (
            self.options.max_context_window_tokens is not None
            and step.metrics
            and step.metrics.total_tokens is not None
            and step.metrics.total_tokens >= self.options.max_context_window_tokens
        ):
            logger.warning(
                "limit_hit_max_context_window_tokens",
                total_tokens=step.metrics.total_tokens,
                max_tokens=self.options.max_context_window_tokens,
                run_id=state.context.run_id,
            )
            return TerminationReason.MAX_CONTEXT_WINDOW_TOKENS

        if (
            self.options.max_tokens_per_run is not None
            and state.total_tokens >= self.options.max_tokens_per_run
        ):
            logger.warning(
                "limit_hit_max_tokens_per_run_post",
                total_tokens=state.total_tokens,
                max_tokens=self.options.max_tokens_per_run,
                run_id=state.context.run_id,
            )
            return TerminationReason.MAX_TOKENS_PER_RUN

        if (
            self.options.max_run_token_cost is not None
            and state.token_cost >= self.options.max_run_token_cost
        ):
            logger.warning(
                "limit_hit_max_run_token_cost_post",
                token_cost=state.token_cost,
                max_cost=self.options.max_run_token_cost,
                run_id=state.context.run_id,
            )
            return TerminationReason.MAX_RUN_TOKEN_COST

        return None

    @staticmethod
    def _is_model_output_limit_hit(finish_reason: str | None) -> bool:
        if finish_reason is None:
            return False

        normalized = finish_reason.strip().lower()
        return normalized in {"length", "max_tokens"}

    def _estimate_token_cost(self, step: StepRecord) -> float:
        """Estimate step cost in USD from token usage and model prices (per 1M tokens)."""
        if not step.metrics:
            return 0.0

        input_tokens = step.metrics.input_tokens or 0
        output_tokens = step.metrics.output_tokens or 0
        cache_hit_tokens = step.metrics.cache_read_tokens or 0
        cache_miss_tokens = max(input_tokens - cache_hit_tokens, 0)

        if input_tokens == 0 and output_tokens == 0:
            return 0.0

        return (
            (cache_hit_tokens * self.cache_hit_price)
            + (cache_miss_tokens * self.input_price)
            + (output_tokens * self.output_price)
        ) / 1_000_000.0

    async def _maybe_generate_summary(
        self,
        state: RunState,
        abort_signal: AbortSignal | None,
    ) -> None:
        if not self.options.enable_termination_summary:
            return

        if state.termination_reason not in self.SUMMARY_REASONS:
            return

        summary_messages = build_termination_messages(
            messages=state.messages,
            termination_reason=state.termination_reason,
            pending_tool_calls=state.pending_tool_calls,
            custom_prompt=self.options.termination_summary_prompt,
        )

        step, llm_context = await self.llm_handler.stream_assistant_step(
            state,
            self.emitter.emit_step_delta,
            abort_signal,
            messages=summary_messages,
            tools=None,
        )
        await self.emitter.emit_step_completed(step, llm=llm_context)
        state.track_step(step, append_message=False)

        logger.info(
            "summary_generated",
            tokens=step.metrics.total_tokens if step.metrics else 0,
        )
