"""
Generic channel data models and protocols.

These models represent common messaging concepts shared across all channel
implementations (Feishu, Slack, DingTalk, etc.).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol


@dataclass
class Attachment:
    type: str
    key: str
    name: str = ""


@dataclass
class InboundMessage:
    channel_instance_id: str
    event_id: str
    message_id: str
    chat_id: str
    chat_type: str
    sender_id: str
    sender_name: str
    text: str
    event_time_ms: int
    raw_payload: dict[str, Any]
    message_type: str = "text"
    thread_id: str | None = None
    mentions: list[str] = field(default_factory=list)
    is_at_bot: bool = False
    attachments: list[Attachment] = field(default_factory=list)


@dataclass
class SessionRuntime:
    session_key: str
    agiwo_session_id: str
    runtime_agent_id: str
    scheduler_state_id: str
    base_agent_id: str
    chat_id: str
    chat_type: str
    trigger_user_id: str
    updated_at: datetime


@dataclass
class BatchContext:
    session_key: str
    chat_id: str
    chat_type: str
    trigger_user_id: str
    trigger_message_id: str
    base_agent_id: str


@dataclass
class BatchPayload:
    context: BatchContext
    messages: list[InboundMessage]
    rendered_user_input: str


class SessionRuntimeStore(Protocol):
    async def get_session_runtime(self, session_key: str) -> SessionRuntime | None: ...
    async def upsert_session_runtime(self, runtime: SessionRuntime) -> None: ...
