from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from server.channels.feishu.commands import build_feishu_command_registry
from server.channels.feishu.commands.base import (
    CommandContext,
    CommandResult,
    CommandSpec,
    build_command_registry,
)
from server.channels.models import Session
from server.channels.session_binding import SessionNotFoundError
from server.config import ConsoleConfig


def _command_context(*, current_session: Session | None = None) -> CommandContext:
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


def _session(session_id: str) -> Session:
    now = datetime.now(timezone.utc)
    return Session(
        id=session_id,
        chat_context_id="ctx-1",
        base_agent_id="agent-1",
        runtime_agent_id="runtime-1",
        scheduler_state_id="state-1",
        created_by="TEST",
        created_at=now,
        updated_at=now,
    )


@pytest.mark.asyncio
async def test_build_command_registry_wraps_specs_and_adds_help() -> None:
    execute = AsyncMock(return_value=CommandResult(text="pong"))
    registry = build_command_registry(
        [
            CommandSpec(
                name="ping",
                description="返回 pong",
                execute=execute,
            )
        ]
    )
    ctx = _command_context()

    parsed = registry.try_parse("/ping hello")

    assert parsed is not None
    handler, args = parsed
    result = await handler.execute(ctx, args)
    assert result.text == "pong"
    execute.assert_awaited_once_with(ctx, "hello")

    help_parsed = registry.try_parse("/help")
    assert help_parsed is not None
    help_handler, help_args = help_parsed
    help_result = await help_handler.execute(ctx, help_args)
    assert "/help — 显示可用命令列表" in help_result.text
    assert "/ping — 返回 pong" in help_result.text


def test_build_feishu_command_registry_includes_expected_commands() -> None:
    registry = build_feishu_command_registry(
        runtime_mgr=SimpleNamespace(),
        session_manager=SimpleNamespace(),
        scheduler=SimpleNamespace(),
        agent_registry=SimpleNamespace(),
        console_config=ConsoleConfig(),
    )

    assert set(registry.handlers) == {
        "agents",
        "cancel",
        "context",
        "detail",
        "help",
        "list",
        "new",
        "resume",
        "status",
        "steer",
        "switch",
    }


@pytest.mark.asyncio
async def test_switch_command_maps_known_errors_from_runtime_manager() -> None:
    session_manager = SimpleNamespace(reset_chat_context=Mock())
    runtime_mgr = SimpleNamespace(
        switch_session=AsyncMock(side_effect=SessionNotFoundError("sess-2")),
        terminate_session_runtime=AsyncMock(),
        create_new_session=AsyncMock(),
        list_user_sessions=AsyncMock(),
        runtime_agents={},
        get_or_create_runtime_agent=AsyncMock(),
    )
    registry = build_feishu_command_registry(
        runtime_mgr=runtime_mgr,
        session_manager=session_manager,
        scheduler=SimpleNamespace(),
        agent_registry=SimpleNamespace(),
        console_config=ConsoleConfig(),
    )

    parsed = registry.try_parse("/switch sess-2")

    assert parsed is not None
    handler, args = parsed
    result = await handler.execute(
        _command_context(current_session=_session("sess-1")),
        args,
    )

    assert result.text == "会话不存在: sess-2"
    session_manager.reset_chat_context.assert_not_called()
    runtime_mgr.terminate_session_runtime.assert_not_called()
