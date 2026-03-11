"""
Session management commands: /new, /list and /switch.
"""

from datetime import datetime
from functools import partial

from agiwo.scheduler.scheduler import Scheduler

from server.channels.agent_runtime import AgentRuntimeManager
from server.channels.session_binding import (
    ChatContextNotFoundError,
    SessionNotFoundError,
    SessionNotInChatContextError,
)
from server.channels.feishu.commands.base import CommandContext, CommandResult, CommandSpec
from server.channels.feishu.commands.status_text import format_scheduler_status
from server.channels.models import Session
from server.channels.session_manager import SessionManager


def build_session_command_specs(
    runtime_mgr: AgentRuntimeManager,
    session_manager: SessionManager,
    scheduler: Scheduler,
) -> list[CommandSpec]:
    return [
        CommandSpec(
            name="new",
            description="创建新会话，重置当前对话上下文",
            execute=partial(_execute_new_session, runtime_mgr, session_manager),
        ),
        CommandSpec(
            name="list",
            description="列出历史会话和概览",
            execute=partial(_execute_list_sessions, runtime_mgr, scheduler),
        ),
        CommandSpec(
            name="switch",
            description="切换当前会话 — /switch <session_id>",
            execute=partial(_execute_switch_session, runtime_mgr, session_manager),
        ),
    ]


async def _execute_new_session(
    runtime_mgr: AgentRuntimeManager,
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
            await runtime_mgr.terminate_session_runtime(
                current_session,
                "用户执行 /new 重置会话",
            )
        except Exception as exc:  # noqa: BLE001
            cleanup_error = str(exc)

    created = await runtime_mgr.create_new_session(
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
    runtime_mgr: AgentRuntimeManager,
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
        switched = await runtime_mgr.switch_session(
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
            await runtime_mgr.terminate_session_runtime(
                previous,
                "用户执行 /switch 切换会话",
            )
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
    runtime_mgr: AgentRuntimeManager,
    scheduler: Scheduler,
    ctx: CommandContext,
    args: str,
) -> CommandResult:
    del args

    items = await runtime_mgr.list_user_sessions(
        user_open_id=ctx.trigger_user_open_id,
        current_chat_context_scope_id=ctx.chat_context_scope_id,
    )
    if not items:
        return CommandResult(text="暂无会话记录。")

    lines: list[str] = [f"会话列表 (共 {len(items)} 个):\n"]
    for i, item in enumerate(items, 1):
        marker_current = " [当前]" if item.is_current else ""
        marker_context = " [当前上下文]" if item.in_current_context else ""
        chat_context = item.chat_context
        session = item.session
        chat_label = (
            "私聊"
            if chat_context.chat_type == "p2p"
            else f"群聊 {chat_context.chat_id}..."
        )
        status_text = await _resolve_status_text(scheduler, session)
        lines.append(
            f"{i}. {chat_label}{marker_current}{marker_context}\n"
            f"   状态: {status_text}\n"
            f"   会话ID: {session.id}\n"
            f"   in_current_context: {'true' if item.in_current_context else 'false'}\n"
            f"   更新于: {_format_time(session.updated_at)}"
        )
    return CommandResult(text="\n".join(lines))


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
