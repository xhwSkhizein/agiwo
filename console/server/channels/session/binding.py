"""Session domain operations for channel runtime state."""

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from server.channels.session.models import ChannelChatContext, Session


@dataclass(frozen=True)
class SessionMutationPlan:
    chat_context: ChannelChatContext
    current_session: Session
    previous_session: Session | None = None
    retired_runtime_agent_id: str | None = None


class SessionContextError(RuntimeError):
    """Base error for channel session-domain operations."""


class ChatContextNotFoundError(SessionContextError):
    def __init__(self, scope_id: str) -> None:
        super().__init__(f"Chat context not found for scope '{scope_id}'")
        self.scope_id = scope_id


class SessionNotFoundError(SessionContextError):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session not found: {session_id}")
        self.session_id = session_id


class SessionNotInChatContextError(SessionContextError):
    def __init__(self, session_id: str, chat_context_id: str) -> None:
        super().__init__(
            f"Session '{session_id}' does not belong to chat context '{chat_context_id}'"
        )
        self.session_id = session_id
        self.chat_context_id = chat_context_id


class BaseAgentNotFoundError(SessionContextError):
    def __init__(self, base_agent_id: str) -> None:
        super().__init__(f"Base agent not found: {base_agent_id}")
        self.base_agent_id = base_agent_id


def open_initial_session(
    *,
    chat_context_scope_id: str,
    channel_instance_id: str,
    chat_id: str,
    chat_type: str,
    user_open_id: str,
    base_agent_id: str,
    created_by: str,
    now: datetime,
) -> SessionMutationPlan:
    chat_context_id = str(uuid4())
    session_id = str(uuid4())
    chat_context = ChannelChatContext(
        id=chat_context_id,
        scope_id=chat_context_scope_id,
        channel_instance_id=channel_instance_id,
        chat_id=chat_id,
        chat_type=chat_type,
        user_open_id=user_open_id,
        base_agent_id=base_agent_id,
        current_session_id=session_id,
        created_at=now,
        updated_at=now,
    )
    session = Session(
        id=session_id,
        chat_context_id=chat_context_id,
        base_agent_id=base_agent_id,
        runtime_agent_id="",
        scheduler_state_id="",
        created_by=created_by,
        created_at=now,
        updated_at=now,
    )
    return SessionMutationPlan(
        chat_context=chat_context,
        current_session=session,
    )


def open_new_session(
    chat_context: ChannelChatContext,
    *,
    base_agent_id: str,
    created_by: str,
    now: datetime,
) -> SessionMutationPlan:
    session = Session(
        id=str(uuid4()),
        chat_context_id=chat_context.id,
        base_agent_id=base_agent_id,
        runtime_agent_id="",
        scheduler_state_id="",
        created_by=created_by,
        created_at=now,
        updated_at=now,
    )
    _set_chat_context_base_agent(chat_context, base_agent_id=base_agent_id, now=now)
    _set_current_session(chat_context, session.id, now=now)
    return SessionMutationPlan(
        chat_context=chat_context,
        current_session=session,
    )


def switch_session(
    chat_context: ChannelChatContext,
    previous_session: Session | None,
    target_session: Session,
    *,
    now: datetime,
) -> SessionMutationPlan:
    _set_current_session(chat_context, target_session.id, now=now)
    target_session.updated_at = now
    return SessionMutationPlan(
        chat_context=chat_context,
        current_session=target_session,
        previous_session=previous_session,
    )


def repair_missing_base_agent(
    chat_context: ChannelChatContext,
    session: Session,
    *,
    default_agent_id: str,
    now: datetime,
) -> SessionMutationPlan:
    old_runtime_agent_id = session.runtime_agent_id
    new_runtime_agent_id = f"{default_agent_id}-rebind"
    retired_runtime_agent_id = None
    if old_runtime_agent_id and old_runtime_agent_id != new_runtime_agent_id:
        retired_runtime_agent_id = old_runtime_agent_id

    session.base_agent_id = default_agent_id
    assign_runtime_identity(session, new_runtime_agent_id)
    session.updated_at = now
    _set_chat_context_base_agent(chat_context, base_agent_id=default_agent_id, now=now)

    return SessionMutationPlan(
        chat_context=chat_context,
        current_session=session,
        retired_runtime_agent_id=retired_runtime_agent_id,
    )


def sync_chat_context_base_agent(
    chat_context: ChannelChatContext,
    *,
    base_agent_id: str,
    now: datetime,
) -> ChannelChatContext:
    _set_chat_context_base_agent(chat_context, base_agent_id=base_agent_id, now=now)
    return chat_context


def assign_runtime_identity(session: Session, runtime_agent_id: str) -> None:
    session.runtime_agent_id = runtime_agent_id
    session.scheduler_state_id = runtime_agent_id


def assign_scheduler_state(session: Session, scheduler_state_id: str) -> None:
    session.scheduler_state_id = scheduler_state_id


def _set_chat_context_base_agent(
    chat_context: ChannelChatContext,
    *,
    base_agent_id: str,
    now: datetime,
) -> None:
    chat_context.base_agent_id = base_agent_id
    chat_context.updated_at = now


def _set_current_session(
    chat_context: ChannelChatContext,
    session_id: str,
    *,
    now: datetime,
) -> None:
    chat_context.current_session_id = session_id
    chat_context.updated_at = now


__all__ = [
    "BaseAgentNotFoundError",
    "ChatContextNotFoundError",
    "SessionContextError",
    "SessionMutationPlan",
    "SessionNotFoundError",
    "SessionNotInChatContextError",
    "assign_runtime_identity",
    "assign_scheduler_state",
    "open_initial_session",
    "open_new_session",
    "repair_missing_base_agent",
    "switch_session",
    "sync_chat_context_base_agent",
]
