from datetime import datetime, timezone

from server.channels.session.binding import (
    assign_runtime_identity,
    assign_scheduler_state,
    fork_session,
    mark_session_task_started,
    open_initial_session,
    open_new_session,
    repair_missing_base_agent,
    switch_session,
)
from server.channels.session.models import ChannelChatContext, Session


def _chat_context() -> ChannelChatContext:
    now = datetime.now(timezone.utc)
    return ChannelChatContext(
        id="ctx-1",
        scope_id="scope-1",
        channel_instance_id="feishu-main",
        chat_id="chat-1",
        chat_type="p2p",
        user_open_id="user-1",
        base_agent_id="agent-1",
        current_session_id="sess-1",
        created_at=now,
        updated_at=now,
    )


def _session() -> Session:
    now = datetime.now(timezone.utc)
    return Session(
        id="sess-1",
        chat_context_id="ctx-1",
        base_agent_id="agent-1",
        runtime_agent_id="runtime-1",
        scheduler_state_id="runtime-1",
        created_by="AUTO",
        created_at=now,
        updated_at=now,
    )


def test_open_initial_session_builds_aligned_session_plan() -> None:
    now = datetime.now(timezone.utc)

    mutation = open_initial_session(
        chat_context_scope_id="scope-1",
        channel_instance_id="feishu-main",
        chat_id="chat-1",
        chat_type="p2p",
        user_open_id="user-1",
        base_agent_id="agent-1",
        created_by="AUTO",
        now=now,
    )

    assert mutation.chat_context.current_session_id == mutation.current_session.id
    assert mutation.current_session.base_agent_id == "agent-1"
    assert mutation.current_session.runtime_agent_id == ""
    assert mutation.current_session.scheduler_state_id == ""


def test_open_new_session_uses_explicit_base_agent_and_rebases_context() -> None:
    chat_context = _chat_context()
    now = datetime.now(timezone.utc)

    mutation = open_new_session(
        chat_context,
        base_agent_id="agent-2",
        created_by="COMMAND_NEW",
        now=now,
    )

    assert mutation.chat_context.base_agent_id == "agent-2"
    assert mutation.chat_context.current_session_id == mutation.current_session.id
    assert mutation.current_session.base_agent_id == "agent-2"


def test_repair_missing_base_agent_tracks_retired_runtime_agent() -> None:
    chat_context = _chat_context()
    session = _session()
    now = datetime.now(timezone.utc)

    mutation = repair_missing_base_agent(
        chat_context,
        session,
        default_agent_id="agent-default",
        now=now,
    )

    assert mutation.current_session.base_agent_id == "agent-default"
    assert mutation.current_session.runtime_agent_id == "agent-default-rebind"
    assert mutation.current_session.scheduler_state_id == "agent-default-rebind"
    assert mutation.chat_context.base_agent_id == "agent-default"
    assert mutation.retired_runtime_agent_id == "runtime-1"


def test_runtime_and_scheduler_identity_updates_flow_through_domain_helpers() -> None:
    session = _session()

    assign_runtime_identity(session, "runtime-2")
    assert session.runtime_agent_id == "runtime-2"
    assert session.scheduler_state_id == "runtime-2"

    assign_scheduler_state(session, "state-2")
    assert session.runtime_agent_id == "runtime-2"
    assert session.scheduler_state_id == "state-2"


def test_switch_session_returns_updated_binding() -> None:
    chat_context = _chat_context()
    previous_session = _session()
    target_session = Session(
        id="sess-2",
        chat_context_id="ctx-1",
        base_agent_id="agent-1",
        runtime_agent_id="runtime-2",
        scheduler_state_id="state-2",
        created_by="AUTO",
        created_at=previous_session.created_at,
        updated_at=previous_session.updated_at,
    )
    now = datetime.now(timezone.utc)

    mutation = switch_session(
        chat_context,
        previous_session,
        target_session,
        now=now,
    )

    assert mutation.chat_context.current_session_id == "sess-2"
    assert mutation.current_session.updated_at == now
    assert mutation.previous_session is not None
    assert mutation.previous_session.id == "sess-1"


def test_mark_session_task_started_initializes_implicit_task() -> None:
    session = _session()

    mark_session_task_started(session, task_id="task-2")

    assert session.current_task_id == "task-2"
    assert session.task_message_count == 0


def test_fork_session_copies_lineage_but_resets_runtime_identity() -> None:
    chat_context = _chat_context()
    source_session = _session()
    source_session.current_task_id = "task-1"
    now = datetime.now(timezone.utc)

    mutation = fork_session(
        chat_context,
        source_session,
        created_by="COMMAND_FORK",
        context_summary="Continue with follow-up task B",
        now=now,
    )

    assert mutation.current_session.id != source_session.id
    assert mutation.current_session.source_session_id == "sess-1"
    assert mutation.current_session.source_task_id == "task-1"
    assert mutation.current_session.fork_context_summary == "Continue with follow-up task B"
    assert mutation.current_session.runtime_agent_id == ""
    assert mutation.current_session.scheduler_state_id == ""
