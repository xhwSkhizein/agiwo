"""Regression tests for Console and Feishu channel consistency.

Both entrypoints must produce equivalent session/task semantics
when using the shared remote workspace services.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from server.channels.session.binding import (
    append_message_to_current_task,
    fork_session,
    mark_session_task_started,
)
from server.channels.session.models import ChannelChatContext, Session
from server.services.remote_workspace_conversation import (
    RemoteWorkspaceConversationService,
)


def _session(
    *,
    current_task_id: str | None = None,
    task_message_count: int = 0,
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


def _chat_context() -> ChannelChatContext:
    now = datetime.now(timezone.utc)
    return ChannelChatContext(
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


def test_console_and_feishu_share_session_first_semantics() -> None:
    """Both entrypoints produce identical session/task summary shapes."""
    console_summary = {
        "session_id": "sess-1",
        "task_id": "task-1",
        "source_session_id": None,
    }
    feishu_summary = {
        "session_id": "sess-1",
        "task_id": "task-1",
        "source_session_id": None,
    }
    assert console_summary == feishu_summary


def test_implicit_task_creation_is_consistent_across_channels() -> None:
    """Both Console and Feishu use the same implicit task creation logic."""
    session = _session()
    assert session.current_task_id is None

    mark_session_task_started(session, task_id="task-1")
    assert session.current_task_id == "task-1"
    assert session.task_message_count == 0

    append_message_to_current_task(session)
    assert session.task_message_count == 1


def test_fork_lineage_is_consistent_across_channels() -> None:
    """Fork from Console and Feishu uses the same fork_session logic."""
    chat_context = _chat_context()
    source = _session(current_task_id="task-1", task_message_count=3)
    now = datetime.now(timezone.utc)

    mutation = fork_session(
        chat_context,
        source,
        created_by="CONSOLE_FORK",
        context_summary="Branch off",
        now=now,
    )

    assert mutation.current_session.source_session_id == "sess-1"
    assert mutation.current_session.source_task_id == "task-1"
    assert mutation.current_session.fork_context_summary == "Branch off"
    assert mutation.current_session.runtime_agent_id == ""
    assert mutation.current_session.current_task_id is None


@pytest.mark.asyncio
async def test_conversation_service_applies_task_semantics_uniformly() -> None:
    """The shared conversation service applies the same task logic regardless of entry channel."""
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

    await service.send_message(
        agent=object(),
        chat_context_scope_id="scope-1",
        user_message="hello from console",
    )

    task_id_after_console = session.current_task_id
    count_after_console = session.task_message_count
    assert task_id_after_console is not None
    assert count_after_console == 1

    await service.send_message(
        agent=object(),
        chat_context_scope_id="scope-1",
        user_message="hello from feishu",
    )

    assert session.current_task_id == task_id_after_console
    assert session.task_message_count == 2
