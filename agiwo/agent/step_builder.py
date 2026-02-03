"""
Step Builder - Handles streaming accumulation for Steps.

This module provides the StepBuilder class that accumulates streaming chunks
from LLM responses into complete Step objects.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from agiwo.agent.schema import Step, StepDelta
from agiwo.llm.helper import normalize_usage_metrics

from agiwo.agent.executor import RunState


class ToolCallAccumulator:
    """Accumulate streaming tool calls."""

    def __init__(self) -> None:
        self._calls: dict[int, dict] = {}

    def accumulate(self, delta_calls: list[dict]) -> None:
        for tc in delta_calls:
            idx = tc.get("index", 0)

            if idx not in self._calls:
                self._calls[idx] = {
                    "id": None,
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }

            acc = self._calls[idx]

            if tc.get("id"):
                acc["id"] = tc["id"]

            if tc.get("type"):
                acc["type"] = tc["type"]

            if tc.get("function"):
                fn = tc["function"]
                if fn.get("name"):
                    acc["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    acc["function"]["arguments"] += fn["arguments"]

    def finalize(self) -> list[dict]:
        return [call for call in self._calls.values() if call["id"] is not None]


@dataclass
class StepBuilder:
    """Accumulates streaming chunks into a complete Step."""

    step: Step
    state: RunState
    step_start_time: float = field(default_factory=time.time)
    tool_accumulator: ToolCallAccumulator = field(default_factory=ToolCallAccumulator)
    first_token_received: bool = False

    async def process_chunk(self, chunk) -> None:
        """Process a single stream chunk."""
        delta = StepDelta()
        has_content = chunk.content or chunk.reasoning_content or chunk.tool_calls

        # Track first token latency
        if has_content and not self.first_token_received:
            self.first_token_received = True
            if self.step.metrics:
                self.step.metrics.first_token_latency_ms = (
                    time.time() - self.step_start_time
                ) * 1000

        # Accumulate content
        if chunk.content:
            self.step.content = (self.step.content or "") + chunk.content
            delta.content = chunk.content

        if chunk.reasoning_content:
            self.step.reasoning_content = (
                self.step.reasoning_content or ""
            ) + chunk.reasoning_content
            delta.reasoning_content = chunk.reasoning_content

        if chunk.tool_calls:
            self.tool_accumulator.accumulate(chunk.tool_calls)
            delta.tool_calls = chunk.tool_calls

        # Usage (typically only in final chunk)
        if chunk.usage and self.step.metrics:
            normalized = normalize_usage_metrics(chunk.usage)
            self.step.metrics.input_tokens = normalized["input_tokens"]
            self.step.metrics.output_tokens = normalized["output_tokens"]
            self.step.metrics.total_tokens = normalized["total_tokens"]
            delta.usage = normalized

        # Emit delta
        if has_content or delta.usage:
            await self.state.emit_delta(self.step.id, delta)

    def finalize(self) -> Step:
        """Finalize step with accumulated data and metrics."""
        self.step.content = self.step.content or None
        self.step.reasoning_content = self.step.reasoning_content or None
        self.step.tool_calls = self.tool_accumulator.finalize() or None

        if self.step.metrics:
            self.step.metrics.exec_end_at = datetime.now(timezone.utc)
            self.step.metrics.duration_ms = (time.time() - self.step_start_time) * 1000

        return self.step


__all__ = ["StepBuilder", "ToolCallAccumulator"]
