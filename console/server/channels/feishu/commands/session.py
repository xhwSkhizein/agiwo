"""
Session management commands: /new, /list and /switch.
"""

from datetime import datetime
from functools import partial

from agiwo.scheduler.scheduler import Scheduler

from server.channels.agent_executor import AgentExecutor
from server.channels.feishu.commands.base import CommandContext, CommandResult, CommandSpec
from server.channels.feishu.commands.post_builder import (
    bold,
    build_post_content,
    code,
    new_line,
    separator_line,
    text_element,
)
from server.channels.feishu.commands.status_text import format_scheduler_status
from server.channels.session import SessionContextService, SessionManager
from server.channels.session.binding import (
    ChatContextNotFoundError,
    SessionNotFoundError,
    SessionNotInChatContextError,
)
from server.channels.session.models import Session


def build_session_command_specs(
    session_service: SessionContextService,
    executor: AgentExecutor,
    session_manager: SessionManager,
    scheduler: Scheduler,
) -> list[CommandSpec]:
    return [
        CommandSpec(
            name="new",
            description="创建新会话，重置当前对话上下文",
            execute=partial(
                _execute_new_session,
                session_service,
                executor,
                session_manager,
            ),
        ),
        CommandSpec(
            name="list",
            description="列出历史会话和概览",
            execute=partial(_execute_list_sessions, session_service, scheduler),
        ),
        CommandSpec(
            name="switch",
            description="切换当前会话 — /switch <session_id>",
            execute=partial(
                _execute_switch_session,
                session_service,
                executor,
                session_manager,
            ),
        ),
    ]


async def _execute_new_session(
    session_service: SessionContextService,
    executor: AgentExecutor,
    session_manager: SessionManager,
    ctx: CommandContext,
    args: str,
) -> CommandResult:
    del args

    current_session = ctx.current_session
    if current_session is None and not ctx.base_agent_id:
        return CommandResult(
            text="默认 Agent 不存在，无法创建会话。请先检查默认 Agent 配置。"
        )

    cleanup_error: str | None = None
    if current_session is not None:
        try:
            await executor.cancel_if_active(current_session, "用户执行 /new 重置会话")
        except Exception as exc:  # noqa: BLE001
            cleanup_error = str(exc)

    created = await session_service.create_new_session(
        chat_context_scope_id=ctx.chat_context_scope_id,
        channel_instance_id=ctx.channel_instance_id,
        chat_id=ctx.chat_id,
        chat_type=ctx.chat_type,
        user_open_id=ctx.trigger_user_open_id,
        base_agent_id=ctx.base_agent_id,
        created_by="COMMAND_NEW",
    )
    session_manager.reset_chat_context(ctx.chat_context_scope_id)
    if cleanup_error is not None:
        return CommandResult(
            text=(
                f"新会话已创建: {created.session.id}\n"
                f"警告: 旧会话清理失败: {cleanup_error}"
            )
        )
    return CommandResult(text=f"新会话已创建: {created.session.id}")


async def _execute_switch_session(
    session_service: SessionContextService,
    executor: AgentExecutor,
    session_manager: SessionManager,
    ctx: CommandContext,
    args: str,
) -> CommandResult:
    target_session_id = args.strip()
    if not target_session_id:
        return CommandResult(text="用法: /switch <session_id>")

    if ctx.current_session is not None and ctx.current_session.id == target_session_id:
        return CommandResult(text=f"当前已在会话 {target_session_id}。")

    try:
        switched = await session_service.switch_session(
            chat_context_scope_id=ctx.chat_context_scope_id,
            target_session_id=target_session_id,
        )
    except (
        ChatContextNotFoundError,
        SessionNotFoundError,
        SessionNotInChatContextError,
    ) as exc:
        return CommandResult(text=_switch_session_error_text(exc, target_session_id))

    cleanup_error: str | None = None
    previous = switched.previous_session
    if previous is not None and previous.id != switched.current_session.id:
        try:
            await executor.cancel_if_active(previous, "用户执行 /switch 切换会话")
        except Exception as exc:  # noqa: BLE001
            cleanup_error = str(exc)

    session_manager.reset_chat_context(ctx.chat_context_scope_id)
    if cleanup_error is not None:
        return CommandResult(
            text=(
                f"已切换到会话: {switched.current_session.id}\n"
                f"警告: 旧会话清理失败: {cleanup_error}"
            )
        )
    return CommandResult(text=f"已切换到会话: {switched.current_session.id}")


async def _execute_list_sessions(
    session_service: SessionContextService,
    scheduler: Scheduler,
    ctx: CommandContext,
    args: str,
) -> CommandResult:
    del args

    items = await session_service.list_user_sessions(
        user_open_id=ctx.trigger_user_open_id,
        current_chat_context_scope_id=ctx.chat_context_scope_id,
    )
    if not items:
        return CommandResult(text="暂无会话记录。")

    content: list[list[dict]] = []

    # Title line
    content.append([bold(f"📋 会话列表 (共 {len(items)} 个)")])
    content.append(new_line())

    for i, item in enumerate(items, 1):
        chat_context = item.chat_context
        session = item.session

        # Chat type indicator
        is_p2p = chat_context.chat_type == "p2p"
        chat_icon = "👤" if is_p2p else "👥"
        chat_label = "私聊" if is_p2p else f"群聊"

        # Status badge
        status_text = await _resolve_status_text(scheduler, session)
        status_emoji = _status_to_emoji(status_text)

        # Header line with number and badges
        header_parts = [
            bold(f"{i}. "),
            text_element(f"{chat_icon} {chat_label}"),
        ]
        if item.is_current:
            header_parts.append(text_element("  [当前会话]", ["bold", "italic"]))
        if item.in_current_context:
            header_parts.append(text_element("  [当前上下文]", ["italic"]))
        content.append(header_parts)

        # Detail lines
        content.append([
            text_element("   状态: "),
            text_element(f"{status_emoji} {status_text}"),
        ])
        content.append([
            text_element("   会话ID: "),
            code(session.id),
        ])
        if session.scheduler_state_id:
            content.append([
                text_element("   调度ID: "),
                code(session.scheduler_state_id),
            ])
        content.append([
            text_element("   更新于: "),
            text_element(_format_time(session.updated_at)),
        ])

        # Separator between items (except last)
        if i < len(items):
            content.append(separator_line())

    # Tips
    content.append(new_line())
    content.append([text_element("💡 提示: 使用 /switch <会话ID> 切换会话")])

    post_content = build_post_content("会话列表", content)
    return CommandResult(post_content=post_content)


def _status_to_emoji(status: str) -> str:
    """Convert status text to emoji."""
    status_map = {
        "运行中": "🟢",
        "等待中": "⏳",
        "队列中": "📋",
        "闲置": "⚪",
        "已完成": "✅",
        "失败": "❌",
        "取消": "🚫",
        "未启动": "⚪",
    }
    for key, emoji in status_map.items():
        if key in status:
            return emoji
    return "⚪"


def _switch_session_error_text(
    error: Exception,
    target_session_id: str,
) -> str:
    if isinstance(error, ChatContextNotFoundError):
        return "当前聊天上下文不存在，请先发送一条消息。"
    if isinstance(error, SessionNotFoundError):
        return f"会话不存在: {target_session_id}"
    if isinstance(error, SessionNotInChatContextError):
        return "只能切换到当前聊天上下文下的历史会话。"
    return f"切换失败: {error}"


async def _resolve_status_text(
    scheduler: Scheduler,
    session: Session,
) -> str:
    if not session.scheduler_state_id:
        return "未启动"

    state = await scheduler.get_state(session.scheduler_state_id)
    if state is None:
        return "未启动"
    return format_scheduler_status(state.status)


def _format_time(dt: datetime) -> str:
    local = dt.astimezone()
    return local.strftime("%m-%d %H:%M")
