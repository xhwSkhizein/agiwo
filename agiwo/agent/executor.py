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

from agiwo.agent.schema import RunOutput
from agiwo.agent.config_options import AgentConfigOptions
from agiwo.agent.limit_checker import ExecutionLimitChecker
from agiwo.agent.llm_handler import LLMStreamHandler
from agiwo.agent.summary_generator import SummaryGenerator
from agiwo.llm.base import Model
from agiwo.utils.abort_signal import AbortSignal
from agiwo.tool.permission.manager import PermissionManager
from agiwo.agent.execution_context import ExecutionContext
from agiwo.tool.base import BaseTool
from agiwo.tool.executor import ToolExecutor
from agiwo.agent.run_state import RunState
from agiwo.utils.logging import get_logger

from agiwo.agent.side_effect_processor import SideEffectProcessor


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
    - ExecutionLimitChecker: 限制检查
    - SummaryGenerator: 摘要生成
    """

    def __init__(
        self,
        model: Model,
        tools: list[BaseTool],
        side_effect_processor: SideEffectProcessor,
        options: AgentConfigOptions | None = None,
        permission_manager: PermissionManager | None = None,
    ):
        self.side_effect_processor = side_effect_processor
        self.options = options or AgentConfigOptions()

        # 初始化各个组件
        self.llm_handler = LLMStreamHandler(model)
        self.tool_executor = ToolExecutor(tools=tools)
        self.limit_checker = ExecutionLimitChecker(self.options)
        self.summary_generator = SummaryGenerator(
            llm_handler=self.llm_handler,
            enabled=self.options.enable_termination_summary,
            custom_prompt=self.options.termination_summary_prompt,
        )

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
        state = RunState.create(
            context,
            self.options,
            messages,
            self.side_effect_processor,
            self._tool_schemas,
        )

        try:
            await self._run_loop(state, pending_tool_calls, abort_signal)
        except asyncio.CancelledError:
            state.termination_reason = "cancelled"
            logger.info("agent_execution_cancelled", run_id=state.context.run_id)
        except Exception as e:
            state.termination_reason = (
                "error_with_context"
                if state.tracker.assistant_steps_count > 0
                else "error"
            )
            logger.error(
                "agent_execution_failed",
                run_id=state.context.run_id,
                error=str(e),
                error_type=type(e).__name__,
                steps_completed=state.tracker.steps_count,
                termination_reason=state.termination_reason,
                exc_info=True,
            )
        finally:
            await self.summary_generator.maybe_generate_summary(state, abort_signal)
            await state.cleanup()

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
            if reason := self.limit_checker.check_limits(state):
                state.termination_reason = reason
                break

            # LLM 调用
            state.current_step += 1
            step = await self.llm_handler.stream_assistant_step(state, abort_signal)

            if not step.tool_calls:
                state.termination_reason = "completed"
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
            seq = await self._allocate_sequence()
            step = state.sf.tool_step(
                sequence=seq,
                tool_call_id=result.tool_call_id,
                name=result.tool_name,
                content=result.content,
                content_for_user=result.content_for_user,
            )
            await state.record_step(step)

    async def _allocate_sequence(self) -> int:
        return await self.side_effect_processor.allocate_sequence()
