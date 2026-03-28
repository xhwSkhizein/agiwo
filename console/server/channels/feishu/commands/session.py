"""
Session management commands: /new, /list, /switch and /fork.
"""

from datetime import datetime
from functools import partial

from agiwo.scheduler.scheduler import Scheduler

from server.channels.feishu.commands.base import (
    CommandContext,
    CommandResult,
    CommandSpec,
)
from server.channels.feishu.commands.post_builder import (
    bold,
    build_post_content,
    code,
    new_line,
    separator_line,
    text_element,
)
from server.channels.feishu.commands.status_text import format_scheduler_status
from server.channels.session import SessionManager
from server.channels.session.models import Session
from server.services.remote_workspace_session import RemoteWorkspaceSessionService


def build_session_command_specs(
    workspace_session_service: RemoteWorkspaceSessionService,
    session_manager: SessionManager,
    scheduler: Scheduler,
) -> list[CommandSpec]:
    return [
        CommandSpec(
            name="new",
            description="创建新会话，重置当前对话上下文",
            execute=partial(
                _execute_new_session,
                workspace_session_service,
                session_manager,
            ),
        ),
        CommandSpec(
            name="list",
            description="列出历史会话和概览",
            execute=partial(
                _execute_list_sessions,
                workspace_session_service,
                scheduler,
            ),
        ),
        CommandSpec(
            name="switch",
            description="切换当前会话 — /switch <session_id>",
            execute=partial(
                _execute_switch_session,
                workspace_session_service,
                session_manager,
            ),
        ),
        CommandSpec(
            name="fork",
            description="从当前会话分叉新会话 — /fork <上下文描述>",
            execute=partial(
                _execute_fork_session,
                workspace_session_service,
                session_manager,
            ),
        ),
    ]


async def _execute_new_session(
    service: RemoteWorkspaceSessionService,
    session_manager: SessionManager,
    ctx: CommandContext,
    args: str,
) -> CommandResult:
    del args

    if ctx.current_session is None and not ctx.base_agent_id:
        return CommandResult(
            text="默认 Agent 不存在，无法创建会话。请先检查默认 Agent 配置。"
        )

    result = await service.create_session(
        chat_context_scope_id=ctx.chat_context_scope_id,
        channel_instance_id=ctx.channel_instance_id,
        chat_id=ctx.chat_id,
        chat_type=ctx.chat_type,
        user_open_id=ctx.trigger_user_open_id,
        base_agent_id=ctx.base_agent_id,
        created_by="COMMAND_NEW",
    )
    session_manager.reset_chat_context(ctx.chat_context_scope_id)
    return CommandResult(
        text=f"新会话已创建: {result.session.id}，之前的任务将在后台继续运行。"
    )


async def _execute_switch_session(
    service: RemoteWorkspaceSessionService,
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
        switched = await service.switch_session(
            chat_context_scope_id=ctx.chat_context_scope_id,
            target_session_id=target_session_id,
        )
    except RuntimeError as exc:
        return CommandResult(text=f"切换失败: {exc}")

    session_manager.reset_chat_context(ctx.chat_context_scope_id)
    return CommandResult(
        text=f"已切换到会话: {switched.current_session.id}，之前的任务将在后台继续运行。"
    )


async def _execute_fork_session(
    service: RemoteWorkspaceSessionService,
    session_manager: SessionManager,
    ctx: CommandContext,
    args: str,
) -> CommandResult:
    context_summary = args.strip()
    if not context_summary:
        return CommandResult(
            text="用法: /fork <上下文描述>\n例如: /fork 继续处理子任务B"
        )

    if ctx.current_session is None:
        return CommandResult(text="当前没有活跃会话，请先发送消息创建会话。")

    try:
        result = await service.fork_session(
            chat_context_scope_id=ctx.chat_context_scope_id,
            context_summary=context_summary,
            created_by="COMMAND_FORK",
        )
    except RuntimeError as exc:
        return CommandResult(text=f"分叉失败: {exc}")

    session_manager.reset_chat_context(ctx.chat_context_scope_id)
    return CommandResult(
        text=(
            f"已从会话 {ctx.current_session.id} 分叉新会话: {result.session.id}\n"
            f"上下文: {context_summary}"
        )
    )


async def _execute_list_sessions(
    service: RemoteWorkspaceSessionService,
    scheduler: Scheduler,
    ctx: CommandContext,
    args: str,
) -> CommandResult:
    del args

    sessions = await service.list_sessions(
        chat_context_scope_id=ctx.chat_context_scope_id,
    )
    if not sessions:
        return CommandResult(text="暂无会话记录。")

    current_session_id = (
        ctx.current_session.id if ctx.current_session is not None else None
    )

    content: list[list[dict]] = []
    content.append([bold(f"📋 会话列表 (共 {len(sessions)} 个)")])
    content.append(new_line())

    for i, session in enumerate(sessions, 1):
        status_text = await _resolve_status_text(scheduler, session)
        status_emoji = _status_to_emoji(status_text)

        header_parts: list[dict] = [bold(f"{i}. ")]
        if session.id == current_session_id:
            header_parts.append(text_element("[当前] ", ["bold", "italic"]))
        content.append(header_parts)

        content.append(
            [
                text_element("   状态: "),
                text_element(f"{status_emoji} {status_text}"),
            ]
        )
        content.append(
            [
                text_element("   会话ID: "),
                code(session.id),
            ]
        )
        if session.current_task_id:
            content.append(
                [
                    text_element("   任务: "),
                    code(session.current_task_id),
                    text_element(f"  ({session.task_message_count} 条消息)"),
                ]
            )
        if session.source_session_id:
            content.append(
                [
                    text_element("   分叉自: "),
                    code(session.source_session_id),
                ]
            )
        if session.scheduler_state_id:
            content.append(
                [
                    text_element("   调度ID: "),
                    code(session.scheduler_state_id),
                ]
            )
        content.append(
            [
                text_element("   更新于: "),
                text_element(_format_time(session.updated_at)),
            ]
        )

        if i < len(sessions):
            content.append(separator_line())

    content.append(new_line())
    content.append([text_element("💡 提示: /switch <会话ID> 切换 | /fork <描述> 分叉")])

    post_content = build_post_content("会话列表", content)
    return CommandResult(post_content=post_content)


def _status_to_emoji(status: str) -> str:
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
