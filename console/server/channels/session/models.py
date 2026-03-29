"""
Generic channel data models and protocols.

These models represent common messaging concepts shared across all channel
implementations (Feishu, Slack, DingTalk, etc.).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from agiwo.agent import UserMessage

if TYPE_CHECKING:
    from server.channels.session.binding import SessionMutationPlan


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
class ChannelChatContext:
    scope_id: str
    channel_instance_id: str
    chat_id: str
    chat_type: str
    user_open_id: str
    base_agent_id: str
    current_session_id: str
    created_at: datetime
    updated_at: datetime


@dataclass
class Session:
    id: str
    chat_context_scope_id: str
    base_agent_id: str
    runtime_agent_id: str
    scheduler_state_id: str
    created_by: str
    created_at: datetime
    updated_at: datetime
    current_task_id: str | None = None
    task_message_count: int = 0
    source_session_id: str | None = None
    source_task_id: str | None = None
    fork_context_summary: str | None = None


@dataclass
class SessionWithContext:
    session: Session
    chat_context: ChannelChatContext


@dataclass
class SessionSwitchResult:
    previous_session: Session | None
    current_session: Session
    chat_context: ChannelChatContext


@dataclass
class UserSessionItem:
    session: Session
    chat_context: ChannelChatContext
    is_current: bool
    in_current_context: bool


@dataclass
class SessionCreateResult:
    chat_context: ChannelChatContext
    session: Session


@dataclass
class BatchContext:
    chat_context_scope_id: str
    channel_instance_id: str
    chat_id: str
    chat_type: str
    trigger_user_id: str
    trigger_message_id: str
    base_agent_id: str


@dataclass
class BatchPayload:
    context: BatchContext
    messages: list[InboundMessage]
    user_message: UserMessage


class ChannelChatSessionStore(Protocol):
    async def get_chat_context(self, scope_id: str) -> ChannelChatContext | None: ...
    async def upsert_chat_context(self, chat_context: ChannelChatContext) -> None: ...
    async def get_session(self, session_id: str) -> Session | None: ...
    async def get_session_with_context(
        self,
        session_id: str,
    ) -> SessionWithContext | None: ...
    async def upsert_session(self, session: Session) -> None: ...
    async def apply_session_mutation(
        self,
        mutation: "SessionMutationPlan",
    ) -> None: ...
    async def list_sessions_by_user(
        self, user_open_id: str
    ) -> list[SessionWithContext]: ...
    async def list_sessions_by_chat_context(
        self, chat_context_scope_id: str
    ) -> list[Session]: ...
