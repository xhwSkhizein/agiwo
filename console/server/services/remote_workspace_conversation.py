"""Shared conversation service for session-scoped, scheduler-first message routing."""

from datetime import datetime, timezone
from uuid import uuid4

from server.channels.session.binding import (
    append_message_to_current_task,
    mark_session_task_started,
)
from server.channels.session.models import ChannelChatSessionStore, Session


class RemoteWorkspaceConversationService:
    """Accept session-scoped user input, implicitly resolve/create task, route through scheduler."""

    def __init__(self, *, session_service, executor) -> None:
        self._session_service = session_service
        self._executor = executor

    async def send_message(self, *, agent, chat_context_scope_id: str, user_message):
        _chat_context, session = await self._session_service.resolve_current_session(
            chat_context_scope_id=chat_context_scope_id,
        )
        return await self._send_to_session(
            agent=agent, session=session, user_message=user_message
        )

    async def send_message_to_session(
        self, *, agent, session: Session, user_message, store: ChannelChatSessionStore
    ):
        """Send a message to a known session (used by channel adapters that already resolved the session)."""
        result = await self._send_to_session(
            agent=agent, session=session, user_message=user_message
        )
        await store.upsert_session(session)
        return result

    async def _send_to_session(self, *, agent, session: Session, user_message):
        if session.current_task_id is None:
            mark_session_task_started(session, task_id=str(uuid4()))
        append_message_to_current_task(session)
        session.updated_at = datetime.now(timezone.utc)
        return await self._executor.execute(agent, session, user_message)
