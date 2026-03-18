"""
LLM Stream Handler - Handles LLM streaming and Step building.
"""

import asyncio
from datetime import datetime, timezone

from agiwo.agent.inner.run_recorder import RunRecorder
from agiwo.agent.inner.step_builder import StepBuilder
from agiwo.agent.inner.run_state import RunState
from agiwo.agent.runtime import StepMetrics, LLMCallContext, StepRecord
from agiwo.llm.base import Model
from agiwo.llm.usage_resolver import ModelUsageEstimator, UsageEstimate
from agiwo.utils.abort_signal import AbortSignal


class LLMStreamHandler:
    """LLM streaming call handler."""

    def __init__(self, model: Model) -> None:
        self.model = model
        self.metrics_resolver = ModelUsageEstimator(model)

    async def stream_assistant_step(
        self,
        state: RunState,
        run_recorder: RunRecorder,
        abort_signal: AbortSignal | None,
        *,
        messages: list[dict] | None = None,
        tools: list[dict] | None = ...,  # sentinel: use default
    ) -> tuple[StepRecord, LLMCallContext]:
        """Stream call LLM and build Step.

        Automatically estimates request tokens before the call and resolves
        usage metrics + cost after the call. Callers receive a fully resolved step.
        """
        messages = messages if messages is not None else state.messages
        tools = state.tool_schemas if tools is ... else tools

        request_estimate = self.metrics_resolver.estimate_request(messages, tools)

        llm_context = LLMCallContext(
            messages=list(messages),
            tools=list(tools) if tools else None,
            request_params=self._get_request_params(),
        )

        builder = await self._create_step_builder(run_recorder)

        async for chunk in self.model.arun_stream(messages, tools=tools):
            self._check_abort(abort_signal)
            await builder.process_chunk(chunk)

        step = builder.finalize()
        self._resolve_step_metrics(step, request_estimate)
        llm_context.finish_reason = builder.finish_reason
        return step, llm_context

    async def _create_step_builder(
        self,
        run_recorder: RunRecorder,
    ) -> StepBuilder:
        """Create StepBuilder."""
        step = await run_recorder.create_assistant_step()
        if step.metrics is None:
            step.metrics = StepMetrics(start_at=datetime.now(timezone.utc))
        step.metrics.model_name = getattr(self.model, "model_name", None)
        step.metrics.provider = getattr(self.model, "provider", None)
        return StepBuilder(step=step, emit_delta=run_recorder.publish_delta)

    def _resolve_step_metrics(
        self,
        step: StepRecord,
        request_estimate: UsageEstimate | None,
    ) -> None:
        """Fill missing metrics on StepRecord and compute cost."""
        if step.metrics is None:
            return

        had_provider_usage = any(
            value is not None
            for value in (
                step.metrics.input_tokens,
                step.metrics.output_tokens,
                step.metrics.total_tokens,
                step.metrics.cache_read_tokens,
                step.metrics.cache_creation_tokens,
            )
        )

        estimated_output = self.metrics_resolver.estimate_assistant_output(
            content=step.content if isinstance(step.content, str) else None,
            reasoning_content=step.reasoning_content,
            tool_calls=step.tool_calls,
        )

        if step.metrics.input_tokens is None and request_estimate is not None:
            step.metrics.input_tokens = request_estimate.input_tokens
        if step.metrics.output_tokens is None:
            step.metrics.output_tokens = estimated_output
        if step.metrics.total_tokens is None:
            step.metrics.total_tokens = (step.metrics.input_tokens or 0) + (
                step.metrics.output_tokens or 0
            )
        if step.metrics.cache_read_tokens is None:
            step.metrics.cache_read_tokens = (
                request_estimate.cache_read_tokens if request_estimate else 0
            )
        if step.metrics.cache_creation_tokens is None:
            step.metrics.cache_creation_tokens = (
                request_estimate.cache_creation_tokens if request_estimate else 0
            )

        if had_provider_usage:
            if request_estimate is not None and (
                step.metrics.input_tokens == request_estimate.input_tokens
                or step.metrics.output_tokens == estimated_output
            ):
                step.metrics.usage_source = "mixed"
            else:
                step.metrics.usage_source = "provider"
        else:
            step.metrics.usage_source = "estimated"

        step.metrics.token_cost = self.metrics_resolver.compute_cost(
            input_tokens=step.metrics.input_tokens,
            output_tokens=step.metrics.output_tokens,
            cache_read_tokens=step.metrics.cache_read_tokens,
            cache_creation_tokens=step.metrics.cache_creation_tokens,
        )

    def _get_request_params(self) -> dict:
        """Get LLM request parameters."""
        params: dict = {
            "model_id": getattr(self.model, "id", None),
            "model_name": getattr(self.model, "name", None),
        }
        if hasattr(self.model, "temperature"):
            params["temperature"] = self.model.temperature
            params["max_output_tokens"] = self.model.max_output_tokens
            params["top_p"] = self.model.top_p
        return params

    @staticmethod
    def _check_abort(abort_signal: AbortSignal | None) -> None:
        """Check abort signal."""
        if abort_signal and abort_signal.is_aborted():
            raise asyncio.CancelledError(abort_signal.reason)
