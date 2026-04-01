"""Unit tests for SessionRuntimeService task-tracking integration.

These tests verify that SessionRuntimeService.execute() handles implicit task
creation and message counting (previously in RemoteWorkspaceConversationService).
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from server.models.session import Session
from server.services.runtime.session_runtime_service import SessionRuntimeService


def _session(
    *, current_task_id: str | None = None, task_message_count: int = 0
) -> Session:
    now = datetime.now(timezone.utc)
    return Session(
        id="sess-1",
        chat_context_scope_id="scope-1",
        base_agent_id="agent-1",
        runtime_agent_id="runtime-1",
        scheduler_state_id="state-1",
        created_by="AUTO",
        created_at=now,
        updated_at=now,
        current_task_id=current_task_id,
        task_message_count=task_message_count,
    )


def _make_runtime_service() -> SessionRuntimeService:
    scheduler = AsyncMock()
    scheduler.route_root_input = AsyncMock(
        return_value=AsyncMock(action="stream", stream=None, state_id="state-1")
    )
    store = AsyncMock()
    store.upsert_session = AsyncMock()
    return SessionRuntimeService(scheduler=scheduler, session_store=store, timeout=60)


@pytest.mark.asyncio
async def test_execute_creates_implicit_task_before_first_dispatch() -> None:
    session = _session()
    runtime_service = _make_runtime_service()

    await runtime_service.execute(
        agent=AsyncMock(), session=session, user_input="hello"
    )

    assert session.current_task_id is not None
    assert session.task_message_count == 1


@pytest.mark.asyncio
async def test_execute_reuses_current_task_for_follow_up_message() -> None:
    session = _session(current_task_id="task-1", task_message_count=1)
    runtime_service = _make_runtime_service()

    await runtime_service.execute(
        agent=AsyncMock(),
        session=session,
        user_input="follow up",
    )

    assert session.current_task_id == "task-1"
    assert session.task_message_count == 2
