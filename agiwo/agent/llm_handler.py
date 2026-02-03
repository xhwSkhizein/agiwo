"""
LLM Stream Handler - Handles LLM streaming and Step building.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone


from agiwo.agent.step_builder import StepBuilder
from agiwo.agent.schema import StepMetrics
from agiwo.agent.run_state import RunState
from agiwo.llm.base import Model
from agiwo.utils.abort_signal import AbortSignal


class LLMStreamHandler:
    """LLM 流式调用处理器"""

    def __init__(self, model: Model) -> None:
        self.model = model

    async def stream_assistant_step(
        self,
        state: RunState,
        abort_signal: AbortSignal | None,
        *,
        messages: list[dict] | None = None,
        tools: list[dict] | None = ...,  # sentinel: use default
        append_message: bool = True,
    ):
        """流式调用 LLM 并构建 Step"""
        messages = messages if messages is not None else state.messages
        tools = state.tool_schemas if tools is ... else tools

        builder = await self._create_step_builder(state, messages, tools)

        async for chunk in self.model.arun_stream(messages, tools=tools):
            self._check_abort(abort_signal)
            await builder.process_chunk(chunk)

        step = builder.finalize()
        await state.record_step(step, append_message=append_message)
        return step

    async def _create_step_builder(
        self, state: RunState, messages: list[dict], tools: list[dict] | None
    ) -> StepBuilder:
        """创建 StepBuilder"""
        seq = await state.pipeline.allocate_sequence()
        step = state.sf.assistant_step(
            sequence=seq,
            content="",
            tool_calls=None,
            llm_messages=messages.copy(),
            llm_tools=tools.copy() if tools else None,
            llm_request_params=self._get_request_params(),
            metrics=StepMetrics(
                start_at=datetime.now(timezone.utc),
                model_name=getattr(self.model, "model_name", None),
                provider=getattr(self.model, "provider", None),
            ),
        )
        return StepBuilder(step=step, state=state)

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
