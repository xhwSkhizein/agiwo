"""
Scheduler control commands for Feishu channel.

Provides /agents, /detail, /steer, /cancel, /resume commands
for managing scheduler agents directly from Feishu chat.
"""

from functools import partial

from agiwo.agent import ContentPart, UserInput, UserMessage
from agiwo.scheduler.engine import Scheduler

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
from server.channels.feishu.commands.status_text import (
    format_scheduler_status,
    status_to_emoji,
)
from server.config import ConsoleConfig
from server.services.agent_registry import AgentRegistry
from server.services.agent_lifecycle import resume_persistent_agent


def build_scheduler_command_specs(
    scheduler: Scheduler,
    registry: AgentRegistry,
    console_config: ConsoleConfig,
) -> list[CommandSpec]:
    return [
        CommandSpec(
            name="agents",
            description="列出所有调度器 Agent 状态",
            execute=partial(_execute_agents, scheduler),
        ),
        CommandSpec(
            name="detail",
            description="查看 Agent 详情 — /detail <state_id>",
            execute=partial(_execute_detail, scheduler),
        ),
        CommandSpec(
            name="steer",
            description="向 Agent 发送引导消息 — /steer <state_id> <message>",
            execute=partial(_execute_steer, scheduler),
        ),
        CommandSpec(
            name="cancel",
            description="取消 Agent 执行 — /cancel <state_id>",
            execute=partial(_execute_cancel, scheduler),
        ),
        CommandSpec(
            name="resume",
            description="恢复持久 Agent — /resume <state_id> <message>",
            execute=partial(_execute_resume, scheduler, registry, console_config),
        ),
    ]


async def _execute_agents(
    scheduler: Scheduler,
    ctx: CommandContext,
    args: str,
) -> CommandResult:
    del ctx, args

    states = await scheduler.list_states(limit=20)
    if not states:
        return CommandResult(text="当前没有 Agent 状态记录。")

    content: list[list[dict]] = []

    # Title
    content.append([bold(f"🤖 Agent 列表 (共 {len(states)} 个)")])
    content.append(new_line())

    for i, state in enumerate(states, 1):
        status_label = format_scheduler_status(state.status)
        status_emoji = status_to_emoji(state.status)
        persistent = "📌" if state.is_persistent else ""
        depth_indent = "  " * (state.depth if state.depth > 0 else 0)

        # Header line with depth indicator
        header_parts = [
            bold(f"{i}. "),
            text_element(f"{depth_indent}{status_emoji} "),
            code(state.id),
        ]
        if persistent:
            header_parts.append(text_element(f" {persistent}"))
        content.append(header_parts)

        # Details
        content.append(
            [
                text_element(f"   状态: {status_label}"),
            ]
        )
        content.append(
            [
                text_element(f"   深度: {state.depth} | 唤醒: {state.wake_count} 次"),
            ]
        )
        if state.session_id:
            content.append(
                [
                    text_element("   会话ID: "),
                    code(state.session_id),
                ]
            )
        if state.agent_config_id:
            content.append(
                [
                    text_element("   配置ID: "),
                    code(state.agent_config_id),
                ]
            )
        if state.parent_id:
            content.append(
                [
                    text_element("   父Agent: "),
                    code(state.parent_id),
                ]
            )
        if state.task is not None:
            task_preview = _user_input_to_preview(state.task, max_len=50)
            content.append(
                [
                    text_element("   任务: "),
                    text_element(task_preview, ["italic"]),
                ]
            )

        # Separator between items
        if i < len(states):
            content.append(separator_line())

    # Tips
    content.append(new_line())
    content.append([text_element("💡 提示: 使用 /detail <state_id> 查看详情")])

    post_content = build_post_content("Agent 列表", content)
    return CommandResult(post_content=post_content)


async def _execute_detail(
    scheduler: Scheduler,
    ctx: CommandContext,
    args: str,
) -> CommandResult:
    del ctx

    state_id = args.strip()
    if not state_id:
        return CommandResult(text="用法: /detail <state_id>")

    state = await scheduler.get_state(state_id)
    if state is None:
        return CommandResult(text=f"未找到 Agent: {state_id}")

    status_label = format_scheduler_status(state.status)
    status_emoji = status_to_emoji(state.status)

    content: list[list[dict]] = []

    # Title with ID
    content.append([bold("📊 Agent 详情")])
    content.append(new_line())

    # Basic info section
    content.append([bold("🔹 基本信息")])
    content.append(
        [
            text_element("  ID: "),
            code(state.id),
        ]
    )
    content.append(
        [
            text_element("  状态: "),
            text_element(f"{status_emoji} {status_label}"),
        ]
    )
    content.append(
        [
            text_element("  持久化: "),
            text_element("是 📌" if state.is_persistent else "否"),
        ]
    )
    content.append(
        [
            text_element("  深度: "),
            text_element(str(state.depth)),
        ]
    )
    content.append(
        [
            text_element("  唤醒次数: "),
            text_element(str(state.wake_count)),
        ]
    )

    content.append(new_line())
    content.append([bold("🔹 关联信息")])

    if state.session_id:
        content.append(
            [
                text_element("  会话ID: "),
                code(state.session_id),
            ]
        )
    if state.agent_config_id:
        content.append(
            [
                text_element("  配置ID: "),
                code(state.agent_config_id),
            ]
        )
    if state.parent_id:
        content.append(
            [
                text_element("  父Agent: "),
                code(state.parent_id),
            ]
        )

    content.append(new_line())
    content.append([bold("🔹 任务")])

    if state.task is not None:
        task_text = _user_input_to_string(state.task)
        content.append([text_element(task_text)])
    else:
        content.append([text_element("(无任务)", ["italic"])])

    if state.result_summary:
        content.append(new_line())
        content.append([bold("🔹 结果摘要")])
        content.append([text_element(state.result_summary)])

    # Wake condition
    if state.wake_condition:
        content.append(new_line())
        content.append([bold("🔹 唤醒条件")])
        wake_type = state.wake_condition.wait_for
        content.append([text_element(f"  类型: {wake_type}")])
        if state.wake_condition.completed_ids:
            completed = ", ".join(state.wake_condition.completed_ids)
            content.append(
                [
                    text_element("  已完成: "),
                    code(completed),
                ]
            )

    content.append(new_line())
    content.append([text_element("💡 提示: /steer <state_id> <消息> 发送引导消息")])

    post_content = build_post_content("Agent 详情", content)
    return CommandResult(post_content=post_content)


async def _execute_steer(
    scheduler: Scheduler,
    ctx: CommandContext,
    args: str,
) -> CommandResult:
    del ctx

    parsed = _parse_state_message_args(args, "用法: /steer <state_id> <message>")
    if isinstance(parsed, CommandResult):
        return parsed

    state_id, message = parsed
    ok = await scheduler.steer(state_id, message, urgent=True)
    if not ok:
        return CommandResult(text=f"发送失败: Agent {state_id} 不存在或不可用。")
    return CommandResult(text=f"已向 {state_id} 发送引导消息。")


async def _execute_cancel(
    scheduler: Scheduler,
    ctx: CommandContext,
    args: str,
) -> CommandResult:
    del ctx

    state_id = args.strip()
    if not state_id:
        return CommandResult(text="用法: /cancel <state_id>")

    ok = await scheduler.cancel(state_id, reason="飞书 /cancel 命令")
    if not ok:
        return CommandResult(text=f"取消失败: Agent {state_id} 不存在或不活跃。")
    return CommandResult(text=f"已取消 Agent {state_id} 及其所有子 Agent。")


async def _execute_resume(
    scheduler: Scheduler,
    registry: AgentRegistry,
    console_config: ConsoleConfig,
    ctx: CommandContext,
    args: str,
) -> CommandResult:
    del ctx

    parsed = _parse_state_message_args(args, "用法: /resume <state_id> <message>")
    if isinstance(parsed, CommandResult):
        return parsed

    state_id, message = parsed
    try:
        await resume_persistent_agent(
            scheduler,
            state_id=state_id,
            message=message,
            registry=registry,
            console_config=console_config,
        )
    except RuntimeError as exc:
        return CommandResult(text=f"恢复失败: {exc}")
    return CommandResult(text=f"已向持久 Agent {state_id} 提交新任务。")


def _parse_state_message_args(
    args: str,
    usage_text: str,
) -> tuple[str, str] | CommandResult:
    parts = args.strip().split(maxsplit=1)
    if len(parts) < 2:
        return CommandResult(text=usage_text)
    return (parts[0], parts[1])


def _user_input_to_string(user_input: UserInput) -> str:
    """Convert UserInput to plain string representation.

    Handles str, list[ContentPart], or UserMessage.
    """
    if isinstance(user_input, str):
        return user_input
    if isinstance(user_input, UserMessage):
        return _content_parts_to_string(user_input.content)
    if isinstance(user_input, list):
        return _content_parts_to_string(user_input)
    return str(user_input)


def _user_input_to_preview(user_input: UserInput, max_len: int = 50) -> str:
    """Convert UserInput to a short preview string.

    Args:
        user_input: The task input (str, list[ContentPart], or UserMessage)
        max_len: Maximum length for the preview

    Returns:
        Truncated string with ellipsis if needed
    """
    text = _user_input_to_string(user_input)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _content_parts_to_string(parts: list[ContentPart]) -> str:
    """Convert list of ContentPart to string (extract text from each part)."""
    texts: list[str] = []
    for part in parts:
        if part.text:
            texts.append(part.text)
        elif part.type.value == "image":
            texts.append("[图片]")
        elif part.type.value == "file":
            texts.append("[文件]")
        elif part.type.value == "audio":
            texts.append("[音频]")
        elif part.type.value == "video":
            texts.append("[视频]")
        else:
            texts.append(f"[{part.type.value}]")
    return " ".join(texts) if texts else "(无内容)"
