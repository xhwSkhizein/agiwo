"""Unit tests for RemoteWorkspaceConversationService."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from server.channels.session.models import Session
from server.services.remote_workspace_conversation import (
    RemoteWorkspaceConversationService,
)


def _session(
    *, current_task_id: str | None = None, task_message_count: int = 0
) -> Session:
    now = datetime.now(timezone.utc)
    return Session(
        id="sess-1",
        chat_context_id="ctx-1",
        base_agent_id="agent-1",
        runtime_agent_id="runtime-1",
        scheduler_state_id="state-1",
        created_by="AUTO",
        created_at=now,
        updated_at=now,
        current_task_id=current_task_id,
        task_message_count=task_message_count,
    )


@pytest.mark.asyncio
async def test_send_message_creates_implicit_task_before_first_dispatch() -> None:
    session = _session()
    session_service = SimpleNamespace(
        resolve_current_session=AsyncMock(return_value=(None, session)),
    )
    executor = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(action="stream", stream=None)),
    )
    service = RemoteWorkspaceConversationService(
        session_service=session_service,
        executor=executor,
    )

    dispatch = await service.send_message(
        agent=object(),
        chat_context_scope_id="scope-1",
        user_message="hello",
    )

    assert session.current_task_id is not None
    assert session.task_message_count == 1
    assert dispatch.action == "stream"


@pytest.mark.asyncio
async def test_send_message_reuses_current_task_for_follow_up_message() -> None:
    session = _session(current_task_id="task-1", task_message_count=1)
    session_service = SimpleNamespace(
        resolve_current_session=AsyncMock(return_value=(None, session)),
    )
    executor = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(action="stream", stream=None)),
    )
    service = RemoteWorkspaceConversationService(
        session_service=session_service,
        executor=executor,
    )

    await service.send_message(
        agent=object(),
        chat_context_scope_id="scope-1",
        user_message="follow up",
    )

    assert session.current_task_id == "task-1"
    assert session.task_message_count == 2
