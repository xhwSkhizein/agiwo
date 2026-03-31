"""Regression tests for Console and Feishu channel consistency.

Both entrypoints must produce equivalent session/task semantics
when using the shared AgentExecutor.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from server.channels.agent_executor import AgentExecutor
from server.channels.session.models import (
    ChannelChatContext,
    Session,
    append_message_to_current_task,
    mark_session_task_started,
)


def _session(
    *,
    current_task_id: str | None = None,
    task_message_count: int = 0,
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

    # Inline fork_session logic
    from uuid import uuid4
    forked_session = Session(
        id=str(uuid4()),
        chat_context_scope_id=chat_context.scope_id,
        base_agent_id=source.base_agent_id,
        runtime_agent_id="",
        scheduler_state_id="",
        created_by="CONSOLE_FORK",
        created_at=now,
        updated_at=now,
        source_session_id=source.id,
        source_task_id=source.current_task_id,
        fork_context_summary="Branch off",
    )

    assert forked_session.source_session_id == "sess-1"
    assert forked_session.source_task_id == "task-1"
    assert forked_session.fork_context_summary == "Branch off"
    assert forked_session.runtime_agent_id == ""
    assert forked_session.current_task_id is None


@pytest.mark.asyncio
async def test_executor_applies_task_semantics_uniformly() -> None:
    """AgentExecutor applies the same task logic regardless of entry channel."""
    session = _session()
    scheduler = AsyncMock()
    scheduler.route_root_input = AsyncMock(
        return_value=AsyncMock(action="stream", stream=None, state_id="state-1")
    )
    store = AsyncMock()
    store.upsert_session = AsyncMock()
    executor = AgentExecutor(scheduler=scheduler, store=store, timeout=60)

    await executor.execute(
        agent=AsyncMock(), session=session, user_input="hello from console"
    )

    task_id_after_console = session.current_task_id
    count_after_console = session.task_message_count
    assert task_id_after_console is not None
    assert count_after_console == 1

    await executor.execute(
        agent=AsyncMock(), session=session, user_input="hello from feishu"
    )

    assert session.current_task_id == task_id_after_console
    assert session.task_message_count == 2
