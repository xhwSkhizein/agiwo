"""Unit tests for scheduler-backed session execution."""

from datetime import datetime, timezone
from unittest.mock import ANY, AsyncMock

import pytest

from agiwo.scheduler.commands import RouteStreamMode
from server.models.session import Session
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


def _make_runtime_service() -> SessionRuntimeService:
    scheduler = AsyncMock()
    scheduler.route_root_input = AsyncMock(
        return_value=AsyncMock(action="stream", stream=None, state_id="sess-1")
    )
    store = AsyncMock()
    store.upsert_session = AsyncMock()
    return SessionRuntimeService(scheduler=scheduler, session_store=store, timeout=60)


@pytest.mark.asyncio
async def test_execute_routes_to_session_root_state_on_first_dispatch() -> None:
    session = _session()
    runtime_service = _make_runtime_service()

    await runtime_service.execute(
        agent=AsyncMock(), session=session, user_input="hello"
    )

    runtime_service._scheduler.route_root_input.assert_awaited_once_with(
        "hello",
        agent=ANY,
        state_id="sess-1",
        session_id="sess-1",
        persistent=True,
        timeout=60,
        stream_mode=RouteStreamMode.UNTIL_SETTLED,
    )


@pytest.mark.asyncio
async def test_execute_reuses_same_root_state_for_follow_up_message() -> None:
    session = _session()
    runtime_service = _make_runtime_service()

    await runtime_service.execute(
        agent=AsyncMock(), session=session, user_input="first"
    )
    await runtime_service.execute(
        agent=AsyncMock(), session=session, user_input="follow up"
    )

    assert runtime_service._scheduler.route_root_input.await_count == 2
    for call in runtime_service._scheduler.route_root_input.await_args_list:
        assert call.kwargs["state_id"] == session.id
        assert call.kwargs["session_id"] == session.id
