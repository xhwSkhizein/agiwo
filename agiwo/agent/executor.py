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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agiwo.agent.base import AgentConfigOptions
from agiwo.agent.limit_checker import ExecutionLimitChecker
from agiwo.agent.llm_handler import LLMStreamHandler
from agiwo.agent.summary_generator import SummaryGenerator
from agiwo.agent.schema import RunMetrics, Step, StepAdapter, StepDelta, RunOutput
from agiwo.llm.base import Model
from agiwo.utils.abort_signal import AbortSignal
from agiwo.tool.permission.manager import PermissionManager
from agiwo.agent.execution_context import ExecutionContext
from agiwo.agent.step_factory import StepFactory
from agiwo.tool.base import BaseTool
from agiwo.tool.executor import ToolExecutor
from agiwo.utils.logging import get_logger

if TYPE_CHECKING:
    from typing import Self

from agiwo.agent.side_effect_processor import SideEffectProcessor


logger = get_logger(__name__)


# ToolCallAccumulator moved to agio.agent.step_builder


class MetricsTracker:
    """Internal metrics tracker for aggregating execution statistics."""

    def __init__(self) -> None:
        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.steps_count = 0
        self.tool_calls_count = 0
        self.assistant_steps_count = 0
        self.pending_tool_calls = None
        self.response_content = None

    def track(self, step: Step) -> None:
        self.steps_count += 1

        if step.metrics:
            if step.metrics.total_tokens:
                self.total_tokens += step.metrics.total_tokens
            if step.metrics.input_tokens:
                self.input_tokens += step.metrics.input_tokens
            if step.metrics.output_tokens:
                self.output_tokens += step.metrics.output_tokens

        if step.is_assistant_step():
            self.assistant_steps_count += 1
            if step.content:
                self.response_content = step.content
            if step.tool_calls:
                self.tool_calls_count += len(step.tool_calls)
                self.pending_tool_calls = step.tool_calls
        elif step.role.value == "tool":
            self.pending_tool_calls = None


# ═══════════════════════════════════════════════════════════════════════════
# Run State
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class RunState:
    """Encapsulates all mutable state for a single execution run."""

    context: ExecutionContext
    config: AgentConfigOptions
    messages: list[dict]
    pipeline: SideEffectProcessor
    tracker: MetricsTracker
    sf: StepFactory
    tool_schemas: list[dict] | None = None
    start_time: float = field(default_factory=time.time)
    current_step: int = 0
    termination_reason: str | None = None

    @classmethod
    def create(
        cls,
        context: ExecutionContext,
        config: AgentConfigOptions,
        messages: list[dict],
        pipeline: SideEffectProcessor,
        tool_schemas: list[dict] | None = None,
    ) -> Self:
        return cls(
            context=context,
            config=config,
            messages=messages,
            pipeline=pipeline,
            tracker=MetricsTracker(),
            sf=StepFactory(context),
            tool_schemas=tool_schemas,
        )

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    async def record_step(self, step: Step, *, append_message: bool = True) -> None:
        """Queue for persistence, track metrics, emit event, optionally append to messages."""
        await self.pipeline.commit_step(step)
        self.tracker.track(step)
        if append_message:
            self.messages.append(StepAdapter.to_llm_message(step))

    async def emit_delta(self, step_id: str, delta: StepDelta) -> None:
        await self.pipeline.emit_step_delta(step_id, delta)

    def build_output(self) -> RunOutput:
        return RunOutput(
            response=self.tracker.response_content,
            run_id=self.context.run_id,
            session_id=self.context.session_id,
            metrics=RunMetrics(
                duration=self.elapsed,
                total_tokens=self.tracker.total_tokens,
                input_tokens=self.tracker.input_tokens,
                output_tokens=self.tracker.output_tokens,
                steps_count=self.tracker.steps_count,
                tool_calls_count=self.tracker.tool_calls_count,
            ),
            termination_reason=self.termination_reason,
        )

    async def cleanup(self) -> None:
        pass


# StepBuilder moved to agio.agent.step_builder


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
        pipeline: SideEffectProcessor,
        config: AgentConfigOptions | None = None,
        permission_manager: PermissionManager | None = None,
    ):
        self.pipeline = pipeline
        self.config = config or AgentConfigOptions()

        # 初始化各个组件
        self.llm_handler = LLMStreamHandler(model)
        self.tool_executor = ToolExecutor(
            tools,
            permission_manager=permission_manager,
            default_timeout=self.config.timeout_per_step,
        )
        self.limit_checker = ExecutionLimitChecker(self.config)
        self.summary_generator = SummaryGenerator(
            llm_handler=self.llm_handler,
            enabled=self.config.enable_termination_summary,
            custom_prompt=self.config.termination_summary_prompt,
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
            context, self.config, messages, self.pipeline, self._tool_schemas
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
        return await self.pipeline.allocate_sequence()
