"""Step domain: StepRecord, StepDelta, StepMetrics, and message helpers."""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agiwo.agent.input import ChannelContext, MessageContent, UserInput, UserMessage
from agiwo.agent.input_codec import (
    extract_text,
    normalize_to_message,
    to_message_content,
)
from agiwo.agent.runtime.core import AgentContext, MessageRole
from agiwo.utils.serialization import serialize_optional_datetime


@dataclass
class StepDelta:
    """Incremental update to a StepRecord for streaming."""

    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[dict] | None = None
    usage: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "reasoning_content": self.reasoning_content,
            "tool_calls": self.tool_calls,
            "usage": self.usage,
        }


@dataclass
class StepMetrics:
    """Metrics for a single StepRecord."""

    start_at: datetime | None = None
    end_at: datetime | None = None
    duration_ms: float | None = None
    total_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    token_cost: float | None = None
    usage_source: str | None = None
    model_name: str | None = None
    provider: str | None = None
    first_token_latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_at": serialize_optional_datetime(self.start_at),
            "end_at": serialize_optional_datetime(self.end_at),
            "duration_ms": self.duration_ms,
            "total_tokens": self.total_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "token_cost": self.token_cost,
            "usage_source": self.usage_source,
            "model_name": self.model_name,
            "provider": self.provider,
            "first_token_latency_ms": self.first_token_latency_ms,
        }


@dataclass
class StepRecord:
    """Unified Step record for persistence and streaming."""

    session_id: str
    run_id: str
    sequence: int
    role: MessageRole
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str | None = None
    content: MessageContent | None = None
    content_for_user: str | None = None
    reasoning_content: str | None = None
    user_input: UserInput | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    metrics: StepMetrics | None = None
    created_at: datetime = field(default_factory=datetime.now)
    parent_run_id: str | None = None
    depth: int = 0

    def is_user_step(self) -> bool:
        return self.role == MessageRole.USER

    def is_assistant_step(self) -> bool:
        return self.role == MessageRole.ASSISTANT

    def is_tool_step(self) -> bool:
        return self.role == MessageRole.TOOL

    def get_llm_content(self) -> MessageContent | None:
        if self.content is not None:
            return self.content
        if self.user_input is not None:
            return to_message_content(normalize_to_message(self.user_input).content)
        return None

    def get_display_text(self) -> str:
        if self.user_input is not None:
            return extract_text(self.user_input)
        if self.content is not None:
            if isinstance(self.content, str):
                return self.content
            return json.dumps(self.content, ensure_ascii=False)
        return ""

    def get_channel_context(self) -> ChannelContext | None:
        if isinstance(self.user_input, UserMessage):
            return self.user_input.context
        return None

    @classmethod
    def user(
        cls,
        ctx: AgentContext,
        *,
        sequence: int,
        user_input: UserInput | None = None,
        content: MessageContent | None = None,
        **overrides: Any,
    ) -> "StepRecord":
        context_attrs = _build_step_context_attrs(ctx, overrides)
        if user_input is not None:
            derived_content = to_message_content(
                normalize_to_message(user_input).content
            )
            return cls(
                sequence=sequence,
                role=MessageRole.USER,
                user_input=user_input,
                content=derived_content,
                **context_attrs,
            )
        if content is not None:
            return cls(
                sequence=sequence,
                role=MessageRole.USER,
                content=content,
                **context_attrs,
            )
        raise ValueError("StepRecord.user() requires either user_input or content")

    @classmethod
    def assistant(
        cls,
        ctx: AgentContext,
        *,
        sequence: int,
        content: str | None = None,
        tool_calls: list[dict] | None = None,
        reasoning_content: str | None = None,
        metrics: StepMetrics | None = None,
        **overrides: Any,
    ) -> "StepRecord":
        context_attrs = _build_step_context_attrs(ctx, overrides)
        return cls(
            sequence=sequence,
            role=MessageRole.ASSISTANT,
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            metrics=metrics,
            **context_attrs,
        )

    @classmethod
    def tool(
        cls,
        ctx: AgentContext,
        *,
        sequence: int,
        tool_call_id: str,
        name: str,
        content: str,
        content_for_user: str | None = None,
        metrics: StepMetrics | None = None,
        **overrides: Any,
    ) -> "StepRecord":
        context_attrs = _build_step_context_attrs(ctx, overrides)
        return cls(
            sequence=sequence,
            role=MessageRole.TOOL,
            content=content,
            content_for_user=content_for_user,
            tool_call_id=tool_call_id,
            name=name,
            metrics=metrics,
            **context_attrs,
        )


def _build_step_context_attrs(
    ctx: AgentContext,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    return {
        "session_id": ctx.session_id,
        "run_id": ctx.run_id,
        "agent_id": overrides.get("agent_id", ctx.agent_id),
        "parent_run_id": overrides.get("parent_run_id", ctx.parent_run_id),
        "depth": overrides.get("depth", ctx.depth),
    }


def step_to_message(step: StepRecord) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": step.role.value}
    if step.content is not None:
        msg["content"] = step.content
    if step.reasoning_content is not None:
        msg["reasoning_content"] = step.reasoning_content
    if step.tool_calls is not None:
        msg["tool_calls"] = step.tool_calls
    if step.tool_call_id is not None:
        msg["tool_call_id"] = step.tool_call_id
    if step.name is not None:
        msg["name"] = step.name
    return msg


def steps_to_messages(steps: list[StepRecord]) -> list[dict[str, Any]]:
    return [step_to_message(step) for step in steps]


__all__ = [
    "StepDelta",
    "StepMetrics",
    "StepRecord",
    "step_to_message",
    "steps_to_messages",
]
