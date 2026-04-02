"""Unit tests for channel-side SessionContextService behavior."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from server.models.session import ChannelChatContext, Session, SessionWithContext
from server.services.agent_registry import AgentRegistry
from server.services.runtime.session_service import SessionContextService


class InMemorySessionStore:
    def __init__(self) -> None:
        now = datetime.now(timezone.utc)
        self.chat_context = ChannelChatContext(
            scope_id="scope-1",
            channel_instance_id="console-web",
            chat_id="chat-1",
            chat_type="dm",
            user_open_id="user-1",
            base_agent_id="agent-1",
            current_session_id="sess-1",
            created_at=now,
            updated_at=now,
        )
        self.sessions: dict[str, Session] = {
            "sess-1": Session(
                id="sess-1",
                chat_context_scope_id="scope-1",
                base_agent_id="agent-1",
                created_by="AUTO",
                created_at=now,
                updated_at=now,
            ),
        }

    async def get_chat_context(self, scope_id: str) -> ChannelChatContext | None:
        if self.chat_context is None:
            return None
        return self.chat_context if scope_id == self.chat_context.scope_id else None

    async def get_session(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)

    async def get_session_with_context(self, session_id: str):
        session = self.sessions.get(session_id)
        if session is None or self.chat_context is None:
            return None
        if session.chat_context_scope_id != self.chat_context.scope_id:
            return None
        return SessionWithContext(session=session, chat_context=self.chat_context)

    async def list_sessions_by_chat_context(
        self, chat_context_scope_id: str
    ) -> list[Session]:
        return [
            session
            for session in self.sessions.values()
            if session.chat_context_scope_id == chat_context_scope_id
        ]

    async def list_sessions_by_base_agent(self, base_agent_id: str) -> list[Session]:
        return [
            session
            for session in self.sessions.values()
            if session.base_agent_id == base_agent_id
        ]

    async def upsert_chat_context(self, chat_context: ChannelChatContext) -> None:
        self.chat_context = chat_context

    async def upsert_session(self, session: Session) -> None:
        self.sessions[session.id] = session

    async def list_sessions_by_user(self, user_open_id: str):
        del user_open_id
        return []

    async def list_sessions(self) -> list[Session]:
        return list(self.sessions.values())


def _make_service(store: InMemorySessionStore) -> SessionContextService:
    registry = AsyncMock(spec=AgentRegistry)
    registry.get_agent_by_name = AsyncMock(return_value=None)
    registry.get_agent = AsyncMock(return_value=None)
    return SessionContextService(
        store=store,
        agent_registry=registry,
        default_agent_name="",
    )


@pytest.mark.asyncio
async def test_fork_session_creates_new_current_session_with_lineage() -> None:
    store = InMemorySessionStore()
    service = _make_service(store)

    result = await service.fork_session(
        chat_context_scope_id="scope-1",
        context_summary="Extract follow-up task B",
        created_by="CONSOLE_FORK",
    )

    assert result.session.source_session_id == "sess-1"
    assert result.session.fork_context_summary == "Extract follow-up task B"
    assert result.chat_context.current_session_id == result.session.id


@pytest.mark.asyncio
async def test_switch_session_updates_current_pointer_only() -> None:
    store = InMemorySessionStore()
    now = datetime.now(timezone.utc)
    store.sessions["sess-2"] = Session(
        id="sess-2",
        chat_context_scope_id="scope-1",
        base_agent_id="agent-1",
        created_by="AUTO",
        created_at=now,
        updated_at=now,
    )
    service = _make_service(store)

    result = await service.switch_session(
        chat_context_scope_id="scope-1",
        target_session_id="sess-2",
    )

    assert result.current_session.id == "sess-2"
    assert result.chat_context.current_session_id == "sess-2"


@pytest.mark.asyncio
async def test_create_session_establishes_new_chat_context_when_none_exists() -> None:
    store = InMemorySessionStore()
    store.chat_context = None  # type: ignore[assignment]
    service = _make_service(store)

    result = await service.create_new_session(
        chat_context_scope_id="scope-new",
        channel_instance_id="console-web",
        chat_id="scope-new",
        chat_type="dm",
        user_open_id="user-1",
        base_agent_id="agent-1",
        created_by="CONSOLE_CREATE",
    )

    assert result.session.base_agent_id == "agent-1"
    assert result.chat_context.current_session_id == result.session.id


@pytest.mark.asyncio
async def test_list_sessions_returns_all_sessions_for_scope() -> None:
    store = InMemorySessionStore()
    service = _make_service(store)

    result = await service.list_sessions(chat_context_scope_id="scope-1")
    assert len(result) == 1
    assert result[0].id == "sess-1"


@pytest.mark.asyncio
async def test_fork_session_by_id_for_console_can_skip_chat_context_update() -> None:
    store = InMemorySessionStore()
    service = _make_service(store)

    result = await service.fork_session_by_id(
        session_id="sess-1",
        context_summary="branching off",
        created_by="CONSOLE_FORK",
        update_chat_context=False,
    )

    assert result.session.source_session_id == "sess-1"
    assert result.session.chat_context_scope_id is None
    assert result.chat_context is not None
    assert result.chat_context.current_session_id == "sess-1"
