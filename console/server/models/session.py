"""Console session and channel runtime models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from agiwo.agent import UserMessage


@dataclass(slots=True)
class Attachment:
    type: str
    key: str
    name: str = ""


@dataclass(slots=True)
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


@dataclass(slots=True)
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


@dataclass(slots=True)
class SessionRuntimeBinding:
    base_agent_id: str
    runtime_agent_id: str = ""
    scheduler_state_id: str = ""


@dataclass(slots=True)
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

    @property
    def runtime_binding(self) -> SessionRuntimeBinding:
        return SessionRuntimeBinding(
            base_agent_id=self.base_agent_id,
            runtime_agent_id=self.runtime_agent_id,
            scheduler_state_id=self.scheduler_state_id,
        )


@dataclass(slots=True)
class SessionWithContext:
    session: Session
    chat_context: ChannelChatContext


@dataclass(slots=True)
class SessionSwitchResult:
    previous_session: Session | None
    current_session: Session
    chat_context: ChannelChatContext


@dataclass(slots=True)
class UserSessionItem:
    session: Session
    chat_context: ChannelChatContext
    is_current: bool
    in_current_context: bool


@dataclass(slots=True)
class SessionCreateResult:
    chat_context: ChannelChatContext
    session: Session


@dataclass(slots=True)
class BatchContext:
    chat_context_scope_id: str
    channel_instance_id: str
    chat_id: str
    chat_type: str
    trigger_user_id: str
    trigger_message_id: str
    base_agent_id: str


@dataclass(slots=True)
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
    async def list_sessions_by_user(
        self, user_open_id: str
    ) -> list[SessionWithContext]: ...
    async def list_sessions_by_chat_context(
        self, chat_context_scope_id: str
    ) -> list[Session]: ...


def bind_runtime_agent(session: Session, agent_id: str) -> None:
    """Bind a runtime agent to a session and default scheduler state to it."""
    session.runtime_agent_id = agent_id
    session.scheduler_state_id = agent_id


def bind_scheduler_state(session: Session, state_id: str) -> None:
    """Bind the latest scheduler state to the session."""
    session.scheduler_state_id = state_id


def reset_runtime_binding(session: Session) -> None:
    """Clear runtime and scheduler bindings while keeping the base agent."""
    session.runtime_agent_id = ""
    session.scheduler_state_id = ""


def start_task(session: Session, *, task_id: str) -> None:
    """Start a new task for the session."""
    session.current_task_id = task_id
    session.task_message_count = 0


def append_task_message(session: Session) -> None:
    """Increment the message count for the active session task."""
    session.task_message_count += 1
