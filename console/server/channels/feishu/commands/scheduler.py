"""
Scheduler control commands for Feishu channel.

Provides /agents, /detail, /steer, /cancel, /resume commands
for managing scheduler agents directly from Feishu chat.
"""

from functools import partial

from agiwo.scheduler.scheduler import Scheduler

from server.channels.feishu.commands.base import (
    CommandContext,
    CommandResult,
    CommandSpec,
)
from server.channels.feishu.commands.status_text import format_scheduler_status
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

    states = await scheduler.store.list_states(limit=20)
    if not states:
        return CommandResult(text="当前没有 Agent 状态记录。")

    lines = [f"Agent 列表 (共 {len(states)} 个):\n"]
    for i, state in enumerate(states, 1):
        status_label = format_scheduler_status(state.status)
        persistent = " [持久]" if state.is_persistent else ""
        lines.append(
            f"{i}. {state.id}{persistent}\n"
            f"   状态: {status_label} | 唤醒: {state.wake_count} 次"
        )
    return CommandResult(text="\n".join(lines))


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
    lines = [
        f"Agent: {state.id}",
        f"状态: {status_label}",
        f"会话: {state.session_id}",
        f"持久: {'是' if state.is_persistent else '否'}",
        f"深度: {state.depth}",
        f"唤醒次数: {state.wake_count}",
    ]
    if state.agent_config_id:
        lines.append(f"配置ID: {state.agent_config_id}")
    if state.parent_id:
        lines.append(f"父Agent: {state.parent_id}")
    if state.result_summary:
        lines.append(f"结果摘要: {state.result_summary[:200]}")
    return CommandResult(text="\n".join(lines))


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
