"""Unit tests for RemoteWorkspaceSessionService."""

from datetime import datetime, timezone

import pytest

from server.channels.session.binding import SessionMutationPlan
from server.channels.session.models import ChannelChatContext, Session
from server.services.remote_workspace_session import RemoteWorkspaceSessionService


class InMemorySessionStore:
    def __init__(self) -> None:
        now = datetime.now(timezone.utc)
        self.chat_context = ChannelChatContext(
            id="ctx-1",
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
                chat_context_id="ctx-1",
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
        return self.chat_context if scope_id == self.chat_context.scope_id else None

    async def get_session(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)

    async def apply_session_mutation(self, mutation: SessionMutationPlan) -> None:
        self.chat_context = mutation.chat_context
        self.sessions[mutation.current_session.id] = mutation.current_session
        if mutation.previous_session is not None:
            self.sessions[mutation.previous_session.id] = mutation.previous_session


@pytest.mark.asyncio
async def test_fork_session_creates_new_current_session_with_weak_lineage() -> None:
    store = InMemorySessionStore()
    service = RemoteWorkspaceSessionService(store=store)

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
        chat_context_id="ctx-1",
        base_agent_id="agent-1",
        runtime_agent_id="runtime-2",
        scheduler_state_id="state-2",
        created_by="AUTO",
        created_at=now,
        updated_at=now,
        current_task_id="task-2",
        task_message_count=3,
    )
    service = RemoteWorkspaceSessionService(store=store)

    result = await service.switch_session(
        chat_context_scope_id="scope-1",
        target_session_id="sess-2",
    )

    assert result.current_session.id == "sess-2"
    assert result.current_session.current_task_id == "task-2"
    assert result.current_session.task_message_count == 3
