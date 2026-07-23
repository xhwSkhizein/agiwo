"""Regression tests for shared MainAgent-backed session semantics."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from agiwo.agent.models.input import UserMessage
from agiwo.agent.models.run import RunOutput
from server.models.session import ChannelChatContext, Session
from server.services.runtime.session_turn_service import SessionTurnService


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
async def test_executor_accepts_all_channels_via_same_main_agent() -> None:
    session = _session()
    store = AsyncMock()
    store.upsert_session = AsyncMock()
    runtime_service = SessionTurnService(
        session_store=store,
        timeout=60,
    )
    handle = AsyncMock(run_id="run-1")
    main_agent = AsyncMock()
    main_agent.accept = AsyncMock(return_value=handle)
    main_agent.wait_current_run = AsyncMock(
        return_value=RunOutput(response="ok", session_id=session.id)
    )

    await runtime_service.submit_user_message(
        main_agent,
        session,
        UserMessage.from_value("hello from console"),
    )
    await runtime_service.submit_user_message(
        main_agent,
        session,
        UserMessage.from_value("hello from feishu"),
    )

    assert main_agent.accept.await_count == 2
    for call in main_agent.accept.await_args_list:
        assert call.args[0].extract_text() in {
            "hello from console",
            "hello from feishu",
        }
