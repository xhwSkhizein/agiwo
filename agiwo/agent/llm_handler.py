"""
LLM Stream Handler - Handles LLM streaming and Step building.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone


from agiwo.agent.step_builder import StepBuilder
from agiwo.agent.schema import StepMetrics, LLMCallContext, StepRecord
from agiwo.agent.run_state import RunState
from agiwo.agent.side_effect_io import SideEffectIO
from agiwo.llm.base import Model
from agiwo.utils.abort_signal import AbortSignal


class LLMStreamHandler:
    """LLM 流式调用处理器"""

    def __init__(self, model: Model) -> None:
        self.model = model

    async def stream_assistant_step(
        self,
        state: RunState,
        run_io: SideEffectIO,
        abort_signal: AbortSignal | None,
        *,
        messages: list[dict] | None = None,
        tools: list[dict] | None = ...,  # sentinel: use default
    ):
        """流式调用 LLM 并构建 Step"""
        messages = messages if messages is not None else state.messages
        tools = state.tool_schemas if tools is ... else tools

        llm_context = LLMCallContext(
            messages=list(messages),
            tools=list(tools) if tools else None,
            request_params=self._get_request_params(),
        )

        builder = await self._create_step_builder(run_io, state)

        async for chunk in self.model.arun_stream(messages, tools=tools):
            self._check_abort(abort_signal)
            await builder.process_chunk(chunk)

        step = builder.finalize()
        return step, llm_context

    async def _create_step_builder(
        self,
        run_io: SideEffectIO,
        state: RunState,
    ) -> StepBuilder:
        """创建 StepBuilder"""
        seq = await run_io.allocate_sequence()
        step = StepRecord.assistant(
            state.context,
            sequence=seq,
            content="",
            tool_calls=None,
            metrics=StepMetrics(
                start_at=datetime.now(timezone.utc),
                model_name=getattr(self.model, "model_name", None),
                provider=getattr(self.model, "provider", None),
            ),
        )
        return StepBuilder(step=step, emit_delta=run_io.emit_step_delta)

    def _get_request_params(self) -> dict | None:
        """获取 LLM 请求参数"""
        if hasattr(self.model, "temperature"):
            return {
                "temperature": self.model.temperature,
                "max_tokens": self.model.max_tokens,
                "top_p": self.model.top_p,
            }
        return None

    @staticmethod
    def _check_abort(abort_signal: AbortSignal | None) -> None:
        """检查中止信号"""
        if abort_signal and abort_signal.is_aborted():
            raise asyncio.CancelledError(abort_signal.reason)
