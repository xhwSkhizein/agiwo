"""Token usage estimation, fallback resolution, and step metrics resolution."""

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import tiktoken

from agiwo.agent.runtime import StepMetrics, StepRecord
from agiwo.llm.base import Model


DEFAULT_TOKENIZER_ENCODING = "cl100k_base"


@dataclass(frozen=True)
class UsageEstimate:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    source: Literal["provider", "estimated", "mixed"] = "estimated"
    confidence: Literal["high", "low"] = "low"


@lru_cache(maxsize=128)
def _resolve_encoding(model_name: str):
    if model_name:
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            pass
    return tiktoken.get_encoding(DEFAULT_TOKENIZER_ENCODING)


class TiktokenUsageEstimator:
    """Estimate token usage from shared message/tool payloads."""

    HIGH_CONFIDENCE_PROVIDERS = frozenset(
        {"openai", "openai-compatible", "deepseek", "nvidia"}
    )

    def __init__(self, model: Model) -> None:
        self._encoding = _resolve_encoding(
            getattr(model, "id", "") or getattr(model, "name", "")
        )
        self._provider = str(getattr(model, "provider", "") or "")

    def estimate_request(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> UsageEstimate:
        input_tokens = self.count_messages(messages) + self.count_tools(tools)
        return UsageEstimate(
            input_tokens=input_tokens,
            total_tokens=input_tokens,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            confidence=self._resolve_confidence(),
        )

    def estimate_assistant_output(
        self,
        *,
        content: str | None,
        reasoning_content: str | None,
        tool_calls: list[dict] | None,
    ) -> int:
        total = self.count_text(content or "")
        total += self.count_text(reasoning_content or "")
        if tool_calls:
            total += self.count_object(tool_calls)
        return total

    def fill_step_usage_metrics(
        self,
        step: StepRecord,
        *,
        request_estimate: UsageEstimate | None,
    ) -> None:
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

        if step.metrics.input_tokens is None and request_estimate is not None:
            step.metrics.input_tokens = request_estimate.input_tokens

        if step.metrics.output_tokens is None:
            step.metrics.output_tokens = self.estimate_assistant_output(
                content=step.content if isinstance(step.content, str) else None,
                reasoning_content=step.reasoning_content,
                tool_calls=step.tool_calls,
            )

        if step.metrics.total_tokens is None:
            step.metrics.total_tokens = (
                (step.metrics.input_tokens or 0) + (step.metrics.output_tokens or 0)
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
            if (
                request_estimate is not None
                and (
                    step.metrics.input_tokens == request_estimate.input_tokens
                    or step.metrics.output_tokens
                    == self.estimate_assistant_output(
                        content=step.content if isinstance(step.content, str) else None,
                        reasoning_content=step.reasoning_content,
                        tool_calls=step.tool_calls,
                    )
                )
            ):
                step.metrics.usage_source = "mixed"
            else:
                step.metrics.usage_source = "provider"
        else:
            step.metrics.usage_source = "estimated"

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def count_object(self, payload: Any) -> int:
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        )
        return self.count_text(serialized)

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        return sum(self.count_object(message) for message in messages)

    def count_tools(self, tools: list[dict[str, Any]] | None) -> int:
        if not tools:
            return 0
        return self.count_object(tools)

    def _resolve_confidence(self) -> Literal["high", "low"]:
        if self._provider in self.HIGH_CONFIDENCE_PROVIDERS:
            return "high"
        return "low"


class StepMetricsResolver:
    """Resolves token usage and cost for completed LLM steps.

    Consolidates:
    - Pre-call request token estimation
    - Post-call usage backfill (provider → estimated fallback)
    - Cost calculation from model prices
    """

    def __init__(self, model: Model) -> None:
        self._estimator = TiktokenUsageEstimator(model)
        self._cache_hit_price = float(getattr(model, "cache_hit_price", 0.0) or 0.0)
        self._input_price = float(getattr(model, "input_price", 0.0) or 0.0)
        self._output_price = float(getattr(model, "output_price", 0.0) or 0.0)

    def estimate_request(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> UsageEstimate:
        """Estimate input tokens before an LLM call."""
        return self._estimator.estimate_request(messages, tools)

    def estimate_messages_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Estimate token count for messages (used by compact threshold check)."""
        return self._estimator.estimate_request(messages, None).input_tokens or 0

    def resolve(self, step: StepRecord, request_estimate: UsageEstimate | None) -> None:
        """Fill missing usage metrics and compute cost on step.metrics in-place."""
        self._estimator.fill_step_usage_metrics(step, request_estimate=request_estimate)
        if step.metrics is not None:
            step.metrics.token_cost = self._compute_cost(step.metrics)

    def _compute_cost(self, metrics: StepMetrics | None) -> float:
        """Compute step cost in USD from token usage and model prices (per 1M tokens)."""
        if metrics is None:
            return 0.0

        input_tokens = metrics.input_tokens or 0
        output_tokens = metrics.output_tokens or 0
        cache_hit_tokens = metrics.cache_read_tokens or 0
        cache_creation_tokens = metrics.cache_creation_tokens or 0
        cache_miss_tokens = max(
            input_tokens - cache_hit_tokens - cache_creation_tokens,
            0,
        )
        paid_input_tokens = cache_creation_tokens + cache_miss_tokens

        if input_tokens == 0 and output_tokens == 0:
            return 0.0

        return (
            (cache_hit_tokens * self._cache_hit_price)
            + (paid_input_tokens * self._input_price)
            + (output_tokens * self._output_price)
        ) / 1_000_000.0


__all__ = ["UsageEstimate", "TiktokenUsageEstimator", "StepMetricsResolver"]
