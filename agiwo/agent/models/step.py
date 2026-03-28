"""Step-scoped models for agent execution and LLM interaction."""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from agiwo.agent.models._serialization import fields_to_dict
from agiwo.agent.models.input import (
    ChannelContext,
    MessageContent,
    UserInput,
    UserMessage,
)

if TYPE_CHECKING:
    from agiwo.agent.runtime.context import RunContext


def _serialize_user_input_structured(
    value: UserInput | None,
) -> str | dict[str, Any] | list[Any] | None:
    return UserMessage.to_structured_payload(value)


def _build_step_context_attrs(
    ctx: "RunContext",
    overrides: dict[str, Any],
) -> dict[str, Any]:
    return {
        "session_id": ctx.session_id,
        "run_id": ctx.run_id,
        "agent_id": overrides.get("agent_id", ctx.agent_id),
        "parent_run_id": overrides.get("parent_run_id", ctx.parent_run_id),
        "depth": overrides.get("depth", ctx.depth),
    }


class MessageRole(str, Enum):
    """Standard LLM message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class StepDelta:
    """Incremental update to a StepRecord for streaming."""

    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[dict] | None = None
    usage: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return fields_to_dict(self)


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
        return fields_to_dict(self)


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

    def __post_init__(self) -> None:
        if (
            self.role == MessageRole.USER
            and self.user_input is not None
            and self.content is None
        ):
            self.content = UserMessage.from_value(self.user_input).to_message_content()

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
            return UserMessage.from_value(self.user_input).to_message_content()
        return None

    def get_display_text(self) -> str:
        if self.user_input is not None:
            return UserMessage.from_value(self.user_input).extract_text()
        if self.content is not None:
            if isinstance(self.content, str):
                return self.content
            return json.dumps(self.content, ensure_ascii=False)
        return ""

    def get_channel_context(self) -> ChannelContext | None:
        if isinstance(self.user_input, UserMessage):
            return self.user_input.context
        return None

    def to_dict(self) -> dict[str, Any]:
        payload = fields_to_dict(self)
        payload["user_input"] = _serialize_user_input_structured(self.user_input)
        payload["metrics"] = self.metrics.to_dict() if self.metrics else None
        return payload

    def to_message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": self.role.value}
        if self.content is not None:
            msg["content"] = self.content
        if self.reasoning_content is not None:
            msg["reasoning_content"] = self.reasoning_content
        if self.tool_calls is not None:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            msg["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            msg["name"] = self.name
        return msg

    @classmethod
    def user(
        cls,
        ctx: "RunContext",
        *,
        sequence: int,
        user_input: UserInput | None = None,
        content: MessageContent | None = None,
        name: str | None = None,
        **overrides: Any,
    ) -> "StepRecord":
        context_attrs = _build_step_context_attrs(ctx, overrides)
        if user_input is not None:
            derived_content = UserMessage.from_value(user_input).to_message_content()
            return cls(
                sequence=sequence,
                role=MessageRole.USER,
                user_input=user_input,
                content=derived_content,
                name=name,
                **context_attrs,
            )
        if content is not None:
            return cls(
                sequence=sequence,
                role=MessageRole.USER,
                content=content,
                name=name,
                **context_attrs,
            )
        raise ValueError("StepRecord.user() requires either user_input or content")

    @classmethod
    def assistant(
        cls,
        ctx: "RunContext",
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
        ctx: "RunContext",
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


@dataclass
class LLMCallContext:
    """LLM call context for observability (not persisted)."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    request_params: dict[str, Any] | None = None
    finish_reason: str | None = None


__all__ = [
    "LLMCallContext",
    "MessageRole",
    "StepDelta",
    "StepMetrics",
    "StepRecord",
]
