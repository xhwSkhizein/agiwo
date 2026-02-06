"""
Agent Executor - LLM agent execution loop with tool calling.

This module implements the core agent execution loop:
- Streams LLM responses and accumulates tool calls
- Executes tools in parallel
- Tracks metrics and state
- Handles termination and summary generation
"""

from __future__ import annotations

import asyncio
import time

from agiwo.agent.schema import RunOutput, StepRecord, TerminationReason
from agiwo.agent.options import AgentOptions
from agiwo.agent.llm_handler import LLMStreamHandler
from agiwo.agent.summarizer import build_termination_messages
from agiwo.llm.base import Model
from agiwo.utils.abort_signal import AbortSignal
from agiwo.agent.execution_context import ExecutionContext
from agiwo.tool.base import BaseTool
from agiwo.tool.executor import ToolExecutor
from agiwo.agent.run_state import RunState
from agiwo.utils.logging import get_logger

from agiwo.agent.side_effect_io import SideEffectIO


logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Agent Executor
# ═══════════════════════════════════════════════════════════════════════════


class AgentExecutor:
    """
    重构后的 AgentExecutor - 使用职责分离的组件。

    职责：协调 LLM 调用、工具执行、限制检查、摘要生成

    组件：
    - LLMStreamHandler: LLM 调用和流式处理
    - ToolExecutor: 工具执行
    - 内置限制检查与摘要生成
    """

    SUMMARY_REASONS = frozenset(
        {
            TerminationReason.MAX_STEPS,
            TerminationReason.TIMEOUT,
            TerminationReason.MAX_TOKENS,
            TerminationReason.ERROR_WITH_CONTEXT,
        }
    )

    def __init__(
        self,
        model: Model,
        tools: list[BaseTool],
        run_io: SideEffectIO,
        options: AgentOptions | None = None,
    ):
        self.run_io = run_io
        self.options = options or AgentOptions()

        # 初始化各个组件
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
            # 检查限制
            if reason := self._check_limits(state):
                state.termination_reason = reason
                break

            # LLM 调用
            state.current_step += 1
            step, llm_context = await self.llm_handler.stream_assistant_step(
                state, self.run_io, abort_signal
            )
            await self.run_io.commit_step(step, llm=llm_context)
            state.track_step(step)

            if not step.tool_calls:
                state.termination_reason = TerminationReason.COMPLETED
                return  # Normal completion

            await self._execute_tools(state, step.tool_calls, abort_signal)

    # ───────────────────────────────────────────────────────────────────
    # Helpers
    # ───────────────────────────────────────────────────────────────────

    async def _execute_tools(
        self,
        state: RunState,
        tool_calls: list[dict],
        abort_signal: AbortSignal | None,
    ) -> None:
        results = await self.tool_executor.execute_batch(
            tool_calls, context=state.context, abort_signal=abort_signal
        )

        for result in results:
            seq = await self.run_io.allocate_sequence()
            step = StepRecord.tool(
                state.context,
                sequence=seq,
                tool_call_id=result.tool_call_id,
                name=result.tool_name,
                content=result.content,
                content_for_user=result.content_for_user,
            )
            await self.run_io.commit_step(step)
            state.track_step(step)

    def _check_limits(self, state: RunState) -> TerminationReason | None:
        """检查所有执行限制，返回终止原因或 None。"""
        if state.current_step >= self.options.max_steps:
            return TerminationReason.MAX_STEPS

        if self.options.run_timeout and state.elapsed > self.options.run_timeout:
            return TerminationReason.TIMEOUT

        if state.context.timeout_at and time.time() >= state.context.timeout_at:
            return TerminationReason.TIMEOUT

        if (
            self.options.max_output_tokens
            and state.total_tokens >= self.options.max_output_tokens
        ):
            return TerminationReason.MAX_TOKENS

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

        summary_messages = build_termination_messages(
            messages=state.messages,
            termination_reason=state.termination_reason,
            pending_tool_calls=state.pending_tool_calls,
            custom_prompt=self.options.termination_summary_prompt,
        )

        step, llm_context = await self.llm_handler.stream_assistant_step(
            state,
            self.run_io,
            abort_signal,
            messages=summary_messages,
            tools=None,
        )
        await self.run_io.commit_step(step, llm=llm_context)
        state.track_step(step, append_message=False)

        logger.info(
            "summary_generated",
            tokens=step.metrics.total_tokens if step.metrics else 0,
        )
