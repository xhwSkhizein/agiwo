"""Unit tests for SessionContextService session management (fork, switch, create, list)."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from server.models.session import ChannelChatContext, Session
from server.services.runtime.session_service import SessionContextService
from server.services.agent_registry import AgentRegistry


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
                runtime_agent_id="runtime-1",
                scheduler_state_id="state-1",
                created_by="AUTO",
                created_at=now,
                updated_at=now,
                current_task_id="task-1",
                task_message_count=1,
            ),
        }

    async def get_chat_context(self, scope_id: str) -> ChannelChatContext | None:
        if self.chat_context is None:
            return None
        return self.chat_context if scope_id == self.chat_context.scope_id else None

    async def get_session(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)

    async def list_sessions_by_chat_context(
        self, chat_context_scope_id: str
    ) -> list[Session]:
        return [
            s
            for s in self.sessions.values()
            if s.chat_context_scope_id == chat_context_scope_id
        ]

    async def upsert_chat_context(self, chat_context: ChannelChatContext) -> None:
        self.chat_context = chat_context

    async def upsert_session(self, session: Session) -> None:
        self.sessions[session.id] = session

    async def list_sessions_by_user(self, user_open_id: str):
        return []


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
async def test_fork_session_creates_new_current_session_with_weak_lineage() -> None:
    store = InMemorySessionStore()
    service = _make_service(store)

    result = await service.fork_session(
        chat_context_scope_id="scope-1",
        context_summary="Extract follow-up task B",
        created_by="CONSOLE_FORK",
    )

    assert result.session.source_session_id == "sess-1"
    assert result.session.source_task_id == "task-1"
    assert result.session.current_task_id is None
    assert result.chat_context.current_session_id == result.session.id


@pytest.mark.asyncio
async def test_switch_session_keeps_task_metadata_intact() -> None:
    store = InMemorySessionStore()
    now = datetime.now(timezone.utc)
    store.sessions["sess-2"] = Session(
        id="sess-2",
        chat_context_scope_id="scope-1",
        base_agent_id="agent-1",
        runtime_agent_id="runtime-2",
        scheduler_state_id="state-2",
        created_by="AUTO",
        created_at=now,
        updated_at=now,
        current_task_id="task-2",
        task_message_count=3,
    )
    service = _make_service(store)

    result = await service.switch_session(
        chat_context_scope_id="scope-1",
        target_session_id="sess-2",
    )

    assert result.current_session.id == "sess-2"
    assert result.current_session.current_task_id == "task-2"
    assert result.current_session.task_message_count == 3


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
    assert result.session.current_task_id is None
    assert result.chat_context.current_session_id == result.session.id


@pytest.mark.asyncio
async def test_create_session_adds_to_existing_chat_context() -> None:
    store = InMemorySessionStore()
    service = _make_service(store)

    result = await service.create_new_session(
        chat_context_scope_id="scope-1",
        channel_instance_id="console-web",
        chat_id="scope-1",
        chat_type="dm",
        user_open_id="user-1",
        base_agent_id="agent-1",
        created_by="CONSOLE_CREATE",
    )

    assert result.session.id != "sess-1"
    assert result.chat_context.current_session_id == result.session.id


@pytest.mark.asyncio
async def test_list_sessions_returns_empty_for_unknown_scope() -> None:
    store = InMemorySessionStore()
    service = _make_service(store)

    result = await service.list_sessions(chat_context_scope_id="unknown-scope")
    assert result == []


@pytest.mark.asyncio
async def test_list_sessions_returns_all_sessions_for_scope() -> None:
    store = InMemorySessionStore()
    service = _make_service(store)

    result = await service.list_sessions(chat_context_scope_id="scope-1")
    assert len(result) == 1
    assert result[0].id == "sess-1"


@pytest.mark.asyncio
async def test_fork_session_by_id() -> None:
    store = InMemorySessionStore()
    service = _make_service(store)

    result = await service.fork_session_by_id(
        session_id="sess-1",
        context_summary="branching off",
        created_by="CONSOLE_FORK",
    )

    assert result.session.source_session_id == "sess-1"
    assert result.session.source_task_id == "task-1"
    assert result.session.fork_context_summary == "branching off"
    assert result.chat_context.current_session_id == result.session.id


@pytest.mark.asyncio
async def test_fork_session_by_id_raises_for_missing_session() -> None:
    store = InMemorySessionStore()
    service = _make_service(store)

    with pytest.raises(RuntimeError, match="Session not found"):
        await service.fork_session_by_id(
            session_id="nonexistent",
            context_summary="nope",
            created_by="CONSOLE_FORK",
        )
