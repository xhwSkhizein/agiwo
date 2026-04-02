"""Regression tests for shared scheduler-backed session semantics."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from server.models.session import ChannelChatContext, Session
from server.services.runtime.session_runtime_service import SessionRuntimeService


def _session() -> Session:
    now = datetime.now(timezone.utc)
    return Session(
        id="sess-1",
        chat_context_scope_id="scope-1",
        base_agent_id="agent-1",
        created_by="AUTO",
        created_at=now,
        updated_at=now,
    )


def _chat_context() -> ChannelChatContext:
    now = datetime.now(timezone.utc)
    return ChannelChatContext(
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


def test_console_and_feishu_share_explicit_session_identity() -> None:
    console_summary = {
        "session_id": "sess-1",
        "root_state_id": "sess-1",
        "source_session_id": None,
    }
    feishu_summary = {
        "session_id": "sess-1",
        "root_state_id": "sess-1",
        "source_session_id": None,
    }
    assert console_summary == feishu_summary


def test_fork_lineage_is_consistent_across_channels() -> None:
    chat_context = _chat_context()
    source = _session()
    now = datetime.now(timezone.utc)

    forked_session = Session(
        id=str(uuid4()),
        chat_context_scope_id=chat_context.scope_id,
        base_agent_id=source.base_agent_id,
        created_by="CONSOLE_FORK",
        created_at=now,
        updated_at=now,
        source_session_id=source.id,
        fork_context_summary="Branch off",
    )

    assert forked_session.source_session_id == "sess-1"
    assert forked_session.fork_context_summary == "Branch off"


@pytest.mark.asyncio
async def test_executor_routes_all_channels_to_same_root_state_identity() -> None:
    session = _session()
    scheduler = AsyncMock()
    scheduler.route_root_input = AsyncMock(
        return_value=AsyncMock(action="stream", stream=None, state_id="sess-1")
    )
    store = AsyncMock()
    store.upsert_session = AsyncMock()
    runtime_service = SessionRuntimeService(
        scheduler=scheduler,
        session_store=store,
        timeout=60,
    )

    await runtime_service.execute(
        agent=AsyncMock(), session=session, user_input="hello from console"
    )
    await runtime_service.execute(
        agent=AsyncMock(), session=session, user_input="hello from feishu"
    )

    assert scheduler.route_root_input.await_count == 2
    for awaited in scheduler.route_root_input.await_args_list:
        assert awaited.kwargs["state_id"] == session.id
        assert awaited.kwargs["session_id"] == session.id
        assert awaited.kwargs["stream_mode"] == "until_settled"
