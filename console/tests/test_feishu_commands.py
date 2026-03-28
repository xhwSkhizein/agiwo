from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from agiwo.agent import ContentPart, ContentType, UserMessage
from agiwo.scheduler.models import AgentState, AgentStateStatus
from agiwo.scheduler.store.memory import InMemoryAgentStateStorage
from server.channels.feishu.commands import build_feishu_command_registry
from server.channels.feishu.commands.base import (
    CommandContext,
    CommandResult,
    CommandSpec,
    build_command_registry,
)
from server.channels.feishu.commands.scheduler import (
    _content_parts_to_string,
    _execute_agents,
    _execute_detail,
    _user_input_to_preview,
    _user_input_to_string,
)
from server.channels.session.models import Session
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
    mock_workspace_service = SimpleNamespace()
    registry = build_feishu_command_registry(
        session_service=SimpleNamespace(
            as_remote_workspace_service=lambda: mock_workspace_service,
        ),
        agent_pool=SimpleNamespace(),
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
        "fork",
        "help",
        "list",
        "new",
        "resume",
        "status",
        "steer",
        "switch",
    }


@pytest.mark.asyncio
async def test_switch_command_maps_known_errors_from_session_service() -> None:
    session_manager = SimpleNamespace(reset_chat_context=Mock())
    workspace_service = SimpleNamespace(
        switch_session=AsyncMock(side_effect=RuntimeError("Session not found: sess-2")),
    )
    registry = build_feishu_command_registry(
        session_service=SimpleNamespace(
            as_remote_workspace_service=lambda: workspace_service,
        ),
        agent_pool=SimpleNamespace(
            runtime_agents={}, get_or_create_runtime_agent=AsyncMock()
        ),
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

    assert "切换失败" in result.text
    assert "sess-2" in result.text
    session_manager.reset_chat_context.assert_not_called()


# ========== UserInput conversion tests ==========


def test_user_input_to_string_with_plain_string() -> None:
    """Test converting a plain string UserInput."""
    assert _user_input_to_string("Hello world") == "Hello world"


def test_user_input_to_string_with_user_message() -> None:
    """Test converting a UserMessage object."""
    msg = UserMessage(content=[ContentPart(type=ContentType.TEXT, text="Test message")])
    assert _user_input_to_string(msg) == "Test message"


def test_user_input_to_string_with_content_parts_list() -> None:
    """Test converting a list of ContentPart objects."""
    parts = [
        ContentPart(type=ContentType.TEXT, text="First"),
        ContentPart(type=ContentType.TEXT, text="Second"),
    ]
    assert _user_input_to_string(parts) == "First Second"


def test_user_input_to_string_with_multimodal_content() -> None:
    """Test converting multimodal content with images and files."""
    parts = [
        ContentPart(type=ContentType.TEXT, text="Check this image"),
        ContentPart(type=ContentType.IMAGE, url="https://example.com/img.png"),
        ContentPart(type=ContentType.FILE, metadata={"name": "doc.pdf"}),
    ]
    result = _user_input_to_string(parts)
    assert "Check this image" in result
    assert "[图片]" in result
    assert "[文件]" in result


def test_user_input_to_preview_truncates_long_text() -> None:
    """Test that preview truncates long text correctly."""
    long_text = "A" * 100
    preview = _user_input_to_preview(long_text, max_len=50)
    assert len(preview) <= 50
    assert preview.endswith("...")


def test_user_input_to_preview_short_text_untouched() -> None:
    """Test that short text is not truncated."""
    short_text = "Short text"
    assert _user_input_to_preview(short_text, max_len=50) == "Short text"


def test_content_parts_to_string_with_empty_list() -> None:
    """Test converting empty content parts list."""
    assert _content_parts_to_string([]) == "(无内容)"


def test_content_parts_to_string_with_audio_video() -> None:
    """Test converting audio and video content parts."""
    parts = [
        ContentPart(type=ContentType.AUDIO, url="https://example.com/audio.mp3"),
        ContentPart(type=ContentType.VIDEO, url="https://example.com/video.mp4"),
    ]
    result = _content_parts_to_string(parts)
    assert "[音频]" in result
    assert "[视频]" in result


# ========== Integration tests for commands with UserInput tasks ==========


@pytest.mark.asyncio
async def test_agents_command_handles_user_message_task() -> None:
    """Test that /agents command correctly handles UserMessage as task."""
    storage = InMemoryAgentStateStorage()
    # InMemory storage doesn't need connect()

    # Create a state with UserMessage as task (the problematic case)
    task_msg = UserMessage(
        content=[ContentPart(type=ContentType.TEXT, text="Do something")]
    )
    state = AgentState(
        id="test-agent-1",
        session_id="sess-1",
        status=AgentStateStatus.IDLE,
        task=task_msg,  # This is a UserMessage, not a string!
        is_persistent=True,
        depth=0,
        wake_count=1,
    )
    await storage.save_state(state)

    scheduler = SimpleNamespace(list_states=storage.list_states)

    ctx = _command_context()
    result = await _execute_agents(scheduler, ctx, "")

    # Should return a post content result, not raise TypeError
    assert result.is_post()
    assert result.post_content is not None


@pytest.mark.asyncio
async def test_detail_command_handles_user_message_task() -> None:
    """Test that /detail command correctly handles UserMessage as task."""
    storage = InMemoryAgentStateStorage()
    # InMemory storage doesn't need connect()

    # Create a state with UserMessage as task
    task_msg = UserMessage(
        content=[
            ContentPart(type=ContentType.TEXT, text="This is a complex task"),
            ContentPart(type=ContentType.TEXT, text="with multiple parts"),
        ]
    )
    state = AgentState(
        id="test-agent-2",
        session_id="sess-1",
        status=AgentStateStatus.RUNNING,
        task=task_msg,
        is_persistent=False,
        depth=1,
        wake_count=0,
        parent_id="parent-1",
    )
    await storage.save_state(state)

    scheduler = SimpleNamespace(
        list_states=storage.list_states,
        get_state=storage.get_state,
    )

    ctx = _command_context()
    result = await _execute_detail(scheduler, ctx, "test-agent-2")

    # Should return a post content result with full task text
    assert result.is_post()
    assert result.post_content is not None
    # The task text should be in the post content
    zh_cn = result.post_content.get("zh_cn", {})
    content_lines = zh_cn.get("content", [])
    all_text = " ".join(str(line) for line in content_lines)
    assert "complex task" in all_text or "multiple parts" in all_text
