"""In-memory Feishu channel metadata store."""

from server.channels.session.models import ChannelChatContext, Session, SessionWithContext
from server.channels.session.binding import SessionMutationPlan


class InMemoryFeishuChannelStore:
    def __init__(self) -> None:
        self._event_dedup: set[str] = set()
        self._chat_context_map: dict[str, ChannelChatContext] = {}
        self._chat_context_scope_by_id: dict[str, str] = {}
        self._session_map: dict[str, Session] = {}
        self._session_ids_by_context: dict[str, set[str]] = {}

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def claim_event(self, channel_instance_id: str, event_id: str) -> bool:
        dedup_key = f"{channel_instance_id}:{event_id}"
        if dedup_key in self._event_dedup:
            return False
        self._event_dedup.add(dedup_key)
        return True

    async def get_chat_context(self, scope_id: str) -> ChannelChatContext | None:
        return self._chat_context_map.get(scope_id)

    async def get_chat_context_by_id(
        self,
        chat_context_id: str,
    ) -> ChannelChatContext | None:
        scope_id = self._chat_context_scope_by_id.get(chat_context_id)
        if scope_id is None:
            return None
        return self._chat_context_map.get(scope_id)

    async def upsert_chat_context(self, chat_context: ChannelChatContext) -> None:
        previous = self._chat_context_map.get(chat_context.scope_id)
        if previous is not None and previous.id != chat_context.id:
            self._chat_context_scope_by_id.pop(previous.id, None)
        self._chat_context_map[chat_context.scope_id] = chat_context
        self._chat_context_scope_by_id[chat_context.id] = chat_context.scope_id

    async def get_session(self, session_id: str) -> Session | None:
        return self._session_map.get(session_id)

    async def get_session_with_context(
        self,
        session_id: str,
    ) -> SessionWithContext | None:
        session = self._session_map.get(session_id)
        if session is None:
            return None
        chat_context = await self.get_chat_context_by_id(session.chat_context_id)
        if chat_context is None:
            return None
        return SessionWithContext(session=session, chat_context=chat_context)

    async def upsert_session(self, session: Session) -> None:
        previous = self._session_map.get(session.id)
        if previous is not None and previous.chat_context_id != session.chat_context_id:
            previous_ids = self._session_ids_by_context.get(previous.chat_context_id)
            if previous_ids is not None:
                previous_ids.discard(session.id)
        self._session_map[session.id] = session
        ids = self._session_ids_by_context.setdefault(session.chat_context_id, set())
        ids.add(session.id)

    async def apply_session_mutation(self, mutation: SessionMutationPlan) -> None:
        await self.upsert_chat_context(mutation.chat_context)
        await self.upsert_session(mutation.current_session)

    async def list_sessions_by_user(self, user_open_id: str) -> list[SessionWithContext]:
        items: list[SessionWithContext] = []
        for session in self._session_map.values():
            chat_context = await self.get_chat_context_by_id(session.chat_context_id)
            if chat_context is None or chat_context.user_open_id != user_open_id:
                continue
            items.append(SessionWithContext(session=session, chat_context=chat_context))
        items.sort(key=lambda item: item.session.updated_at, reverse=True)
        return items

    async def list_sessions_by_chat_context(self, chat_context_id: str) -> list[Session]:
        session_ids = self._session_ids_by_context.get(chat_context_id, set())
        sessions = [
            self._session_map[session_id]
            for session_id in session_ids
            if session_id in self._session_map
        ]
        sessions.sort(key=lambda session: session.updated_at, reverse=True)
        return sessions


__all__ = ["InMemoryFeishuChannelStore"]
