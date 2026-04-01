"""Tests for Feishu session commands using SessionContextService."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from server.channels.feishu.commands.base import CommandContext
from server.channels.feishu.commands.session import build_session_command_specs
from server.models.session import (
    ChannelChatContext,
    Session,
    SessionCreateResult,
    SessionSwitchResult,
)


def _ctx(*, current_session: Session | None = None) -> CommandContext:
    return CommandContext(
        chat_context_scope_id="scope-1",
        channel_instance_id="feishu-main",
        chat_id="chat-1",
        chat_type="p2p",
        trigger_user_open_id="user-1",
        trigger_message_id="msg-1",
        base_agent_id="agent-1",
        chat_context=None,
        current_session=current_session,
    )


def _session(session_id: str = "sess-1") -> Session:
    now = datetime.now(timezone.utc)
    return Session(
        id=session_id,
        chat_context_scope_id="scope-1",
        base_agent_id="agent-1",
        runtime_agent_id="runtime-1",
        scheduler_state_id="state-1",
        created_by="AUTO",
        created_at=now,
        updated_at=now,
        current_task_id="task-1",
        task_message_count=2,
    )


def _chat_context() -> ChannelChatContext:
    now = datetime.now(timezone.utc)
    return ChannelChatContext(
        scope_id="scope-1",
        channel_instance_id="feishu-main",
        chat_id="chat-1",
        chat_type="p2p",
        user_open_id="user-1",
        base_agent_id="agent-1",
        current_session_id="sess-new",
        created_at=now,
        updated_at=now,
    )


def _build_specs(service, session_manager=None, scheduler=None):
    sm = session_manager or SimpleNamespace(reset_chat_context=lambda _: None)
    sc = scheduler or SimpleNamespace(get_state=AsyncMock(return_value=None))
    return {spec.name: spec for spec in build_session_command_specs(service, sm, sc)}


@pytest.mark.asyncio
async def test_new_command_uses_session_context_service() -> None:
    service = SimpleNamespace(
        create_new_session=AsyncMock(
            return_value=SessionCreateResult(
                chat_context=_chat_context(),
                session=_session("sess-new"),
            )
        )
    )
    specs = _build_specs(service)
    result = await specs["new"].execute(_ctx(), "")

    assert "sess-new" in result.text
    service.create_new_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_switch_command_uses_session_context_service() -> None:
    service = SimpleNamespace(
        switch_session=AsyncMock(
            return_value=SessionSwitchResult(
                previous_session=_session("sess-1"),
                current_session=_session("sess-2"),
                chat_context=_chat_context(),
            )
        )
    )
    specs = _build_specs(service)
    result = await specs["switch"].execute(_ctx(current_session=_session()), "sess-2")

    assert "sess-2" in result.text
    service.switch_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_fork_command_creates_forked_session() -> None:
    forked = _session("sess-forked")
    forked.source_session_id = "sess-1"
    service = SimpleNamespace(
        fork_session=AsyncMock(
            return_value=SessionCreateResult(
                chat_context=_chat_context(),
                session=forked,
            )
        )
    )
    specs = _build_specs(service)
    ctx = _ctx(current_session=_session())
    result = await specs["fork"].execute(ctx, "继续处理子任务B")

    assert "sess-forked" in result.text
    assert "继续处理子任务B" in result.text
    service.fork_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_fork_command_requires_context_summary() -> None:
    service = SimpleNamespace()
    specs = _build_specs(service)
    result = await specs["fork"].execute(_ctx(current_session=_session()), "")

    assert "用法" in result.text


@pytest.mark.asyncio
async def test_fork_command_requires_active_session() -> None:
    service = SimpleNamespace()
    specs = _build_specs(service)
    result = await specs["fork"].execute(_ctx(current_session=None), "something")

    assert "没有活跃会话" in result.text


@pytest.mark.asyncio
async def test_list_command_shows_task_and_fork_info() -> None:
    session = _session()
    session.source_session_id = "sess-0"
    service = SimpleNamespace(
        list_sessions=AsyncMock(return_value=[session]),
    )
    scheduler = SimpleNamespace(get_state=AsyncMock(return_value=None))
    specs = _build_specs(service, scheduler=scheduler)
    result = await specs["list"].execute(_ctx(current_session=session), "")

    assert result.post_content is not None
