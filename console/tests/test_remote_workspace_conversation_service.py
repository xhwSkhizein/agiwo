"""Unit tests for MainAgent-backed session execution."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from agiwo.agent.models.input import UserMessage
from agiwo.agent.models.run import RunOutput
from server.models.session import Session
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


def _make_runtime_service() -> SessionTurnService:
    store = AsyncMock()
    store.upsert_session = AsyncMock()
    return SessionTurnService(session_store=store, timeout=60)


@pytest.mark.asyncio
async def test_submit_user_message_accepts_via_main_agent() -> None:
    session = _session()
    runtime_service = _make_runtime_service()
    handle = AsyncMock(run_id="run-1")
    main_agent = AsyncMock()
    main_agent.accept = AsyncMock(return_value=handle)
    main_agent.wait_current_run = AsyncMock(
        return_value=RunOutput(response="hello", session_id=session.id)
    )

    run_id, output = await runtime_service.submit_user_message(
        main_agent,
        session,
        UserMessage.from_value("hello"),
    )

    assert run_id == "run-1"
    assert output.response == "hello"
    main_agent.accept.assert_awaited_once()
    accepted = main_agent.accept.await_args.args[0]
    assert accepted.extract_text() == "hello"
    main_agent.wait_current_run.assert_awaited_once()
    runtime_service._session_store.upsert_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_submit_user_message_reuses_same_main_agent_for_follow_up() -> None:
    session = _session()
    runtime_service = _make_runtime_service()
    handle = AsyncMock(run_id="run-1")
    main_agent = AsyncMock()
    main_agent.accept = AsyncMock(return_value=handle)
    main_agent.wait_current_run = AsyncMock(
        return_value=RunOutput(response="ok", session_id=session.id)
    )

    await runtime_service.submit_user_message(
        main_agent,
        session,
        UserMessage.from_value("first"),
    )
    await runtime_service.submit_user_message(
        main_agent,
        session,
        UserMessage.from_value("follow up"),
    )

    assert main_agent.accept.await_count == 2
    assert main_agent.wait_current_run.await_count == 2
