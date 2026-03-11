"""Compatibility facade for agent-domain models and codecs."""

from agiwo.agent.compact_types import CompactMetadata, CompactResult
from agiwo.agent.input import (
    ChannelContext,
    ContentPart,
    ContentType,
    MessageContent,
    UserInput,
    UserMessage,
)
from agiwo.agent.input_codec import (
    deserialize_user_input,
    extract_text,
    normalize_to_message,
    serialize_user_input,
    to_message_content,
)
from agiwo.agent.memory_types import MemoryRecord
from agiwo.agent.runtime import (
    EventType,
    LLMCallContext,
    MessageRole,
    Run,
    RunMetrics,
    RunOutput,
    RunStatus,
    StepDelta,
    StepMetrics,
    StepRecord,
    StreamEvent,
    TerminationReason,
    step_to_message,
    steps_to_messages,
)

__all__ = [
    "ChannelContext",
    "CompactMetadata",
    "CompactResult",
    "ContentPart",
    "ContentType",
    "EventType",
    "LLMCallContext",
    "MemoryRecord",
    "MessageContent",
    "MessageRole",
    "Run",
    "RunMetrics",
    "RunOutput",
    "RunStatus",
    "StepDelta",
    "StepMetrics",
    "StepRecord",
    "StreamEvent",
    "TerminationReason",
    "UserInput",
    "UserMessage",
    "deserialize_user_input",
    "extract_text",
    "normalize_to_message",
    "serialize_user_input",
    "step_to_message",
    "steps_to_messages",
    "to_message_content",
]
