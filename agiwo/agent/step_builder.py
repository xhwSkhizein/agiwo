"""
Step Builder - Handles streaming accumulation for Steps.

This module provides the StepBuilder class that accumulates streaming chunks
from LLM responses into complete Step objects.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable
from datetime import datetime, timezone

from agiwo.agent.schema import StepRecord, StepDelta
from agiwo.llm.base import StreamChunk
from agiwo.llm.helper import normalize_usage_metrics


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

    step: StepRecord
    emit_delta: Callable[[str, StepDelta], Awaitable[None]]
    step_start_time: float = field(default_factory=time.time)
    tool_accumulator: ToolCallAccumulator = field(default_factory=ToolCallAccumulator)
    first_token_received: bool = False

    async def process_chunk(self, chunk: StreamChunk) -> None:
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

        if chunk.content:
            self._append_content(delta, chunk.content)

        if chunk.reasoning_content:
            self._append_reasoning(delta, chunk.reasoning_content)

        if chunk.tool_calls:
            self._append_tool_calls(delta, chunk.tool_calls)

        # Usage (typically only in final chunk)
        if chunk.usage and self.step.metrics:
            normalized = normalize_usage_metrics(chunk.usage)
            self.step.metrics.input_tokens = normalized["input_tokens"]
            self.step.metrics.output_tokens = normalized["output_tokens"]
            self.step.metrics.total_tokens = normalized["total_tokens"]
            delta.usage = normalized

        # Emit delta
        if has_content or delta.usage:
            await self.emit_delta(self.step.id, delta)

    def finalize(self) -> StepRecord:
        """Finalize step with accumulated data and metrics."""
        self.step.content = self.step.content or None
        self.step.reasoning_content = self.step.reasoning_content or None
        self.step.tool_calls = self.tool_accumulator.finalize() or None

        if self.step.metrics:
            self.step.metrics.end_at = datetime.now(timezone.utc)
            self.step.metrics.duration_ms = (time.time() - self.step_start_time) * 1000

        return self.step

    def _append_content(self, delta: StepDelta, content: str) -> None:
        self.step.content = (self.step.content or "") + content
        delta.content = content

    def _append_reasoning(self, delta: StepDelta, reasoning: str) -> None:
        self.step.reasoning_content = (self.step.reasoning_content or "") + reasoning
        delta.reasoning_content = reasoning

    def _append_tool_calls(self, delta: StepDelta, tool_calls: list[dict]) -> None:
        self.tool_accumulator.accumulate(tool_calls)
        delta.tool_calls = tool_calls


__all__ = ["StepBuilder", "ToolCallAccumulator"]
