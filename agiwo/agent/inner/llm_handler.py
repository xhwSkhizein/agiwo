"""
LLM Stream Handler - Handles LLM streaming and Step building.
"""

import asyncio
from datetime import datetime, timezone

from agiwo.agent.inner.step_builder import StepBuilder
from agiwo.agent.inner.run_state import RunState
from agiwo.agent.inner.side_effect_io import SideEffectIO

from agiwo.agent.schema import StepMetrics, LLMCallContext, StepRecord
from agiwo.llm.base import Model
from agiwo.utils.abort_signal import AbortSignal


class LLMStreamHandler:
    """LLM streaming call handler."""

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
        """Stream call LLM and build Step."""
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
        """Create StepBuilder."""
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
        """Get LLM request parameters."""
        if hasattr(self.model, "temperature"):
            return {
                "temperature": self.model.temperature,
                "max_tokens": self.model.max_tokens,
                "top_p": self.model.top_p,
            }
        return None

    @staticmethod
    def _check_abort(abort_signal: AbortSignal | None) -> None:
        """Check abort signal."""
        if abort_signal and abort_signal.is_aborted():
            raise asyncio.CancelledError(abort_signal.reason)
