"""Shared session application service for Console + Feishu remote workspace."""

from dataclasses import dataclass
from datetime import datetime, timezone

from server.channels.session.binding import fork_session, switch_session
from server.channels.session.models import (
    ChannelChatContext,
    ChannelChatSessionStore,
    Session,
    SessionSwitchResult,
)


@dataclass
class WorkspaceForkResult:
    chat_context: ChannelChatContext
    session: Session


class RemoteWorkspaceSessionService:
    def __init__(self, *, store: ChannelChatSessionStore) -> None:
        self._store = store

    async def switch_session(
        self,
        *,
        chat_context_scope_id: str,
        target_session_id: str,
    ) -> SessionSwitchResult:
        chat_context = await self._store.get_chat_context(chat_context_scope_id)
        if chat_context is None:
            raise RuntimeError(f"Chat context not found: {chat_context_scope_id}")
        previous = await self._store.get_session(chat_context.current_session_id)
        target = await self._store.get_session(target_session_id)
        if target is None:
            raise RuntimeError(f"Session not found: {target_session_id}")
        mutation = switch_session(
            chat_context,
            previous,
            target,
            now=datetime.now(timezone.utc),
        )
        await self._store.apply_session_mutation(mutation)
        return SessionSwitchResult(
            previous_session=mutation.previous_session,
            current_session=mutation.current_session,
            chat_context=mutation.chat_context,
        )

    async def fork_session(
        self,
        *,
        chat_context_scope_id: str,
        context_summary: str,
        created_by: str,
    ) -> WorkspaceForkResult:
        chat_context = await self._store.get_chat_context(chat_context_scope_id)
        if chat_context is None:
            raise RuntimeError(f"Chat context not found: {chat_context_scope_id}")
        source = await self._store.get_session(chat_context.current_session_id)
        if source is None:
            raise RuntimeError(
                f"Current session not found: {chat_context.current_session_id}"
            )
        mutation = fork_session(
            chat_context,
            source,
            created_by=created_by,
            context_summary=context_summary,
            now=datetime.now(timezone.utc),
        )
        await self._store.apply_session_mutation(mutation)
        return WorkspaceForkResult(
            chat_context=mutation.chat_context,
            session=mutation.current_session,
        )
