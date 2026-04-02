"""
Runtime context inspection commands: /context and /status.
"""

from functools import partial

from agiwo.agent import Agent
from agiwo.llm.limits import (
    resolve_max_context_window,
    resolve_max_input_tokens_per_call,
    resolve_max_output_tokens,
)
from agiwo.scheduler.engine import Scheduler

from server.channels.feishu.commands.base import (
    CommandContext,
    CommandResult,
    CommandSpec,
)
from server.channels.feishu.commands.status_text import format_scheduler_status
from server.models.session import Session
from server.services.metrics import summarize_runs_paginated
from server.services.runtime import AgentRuntimeCache


_PROMPT_PREVIEW_MAX_LEN = 5000


def build_context_command_specs(
    agent_pool: AgentRuntimeCache,
    scheduler: Scheduler,
) -> list[CommandSpec]:
    return [
        CommandSpec(
            name="context",
            description="查看当前会话的 SystemPrompt 和概览",
            execute=partial(_execute_context, agent_pool),
        ),
        CommandSpec(
            name="status",
            description="查看当前对话的统计信息",
            execute=partial(_execute_status, agent_pool, scheduler),
        ),
    ]


async def _execute_context(
    agent_pool: AgentRuntimeCache,
    ctx: CommandContext,
    args: str,
) -> CommandResult:
    del args

    current_session = ctx.current_session
    if current_session is None:
        return CommandResult(text="当前会话未初始化，请先发送一条消息。")

    agent, error_text = await _load_session_agent(agent_pool, current_session)
    if error_text is not None:
        return CommandResult(text=error_text)
    if agent is None:
        return CommandResult(text="加载 Agent 失败: runtime agent unavailable")

    prompt = await agent.get_effective_system_prompt()
    prompt_preview = _truncate(prompt, _PROMPT_PREVIEW_MAX_LEN)
    tool_names = [tool.name for tool in agent.tools]

    lines = [
        f"Agent: {agent.name} (id: {agent.id})",
        f"Model: {_format_model_info(agent)}",
        f"Tools ({len(tool_names)}): {', '.join(tool_names)}",
        f"Session: {current_session.id}",
        "",
        "--- System Prompt ---",
        prompt_preview,
    ]
    return CommandResult(text="\n".join(lines))


async def _execute_status(
    agent_pool: AgentRuntimeCache,
    scheduler: Scheduler,
    ctx: CommandContext,
    args: str,
) -> CommandResult:
    del args

    current_session = ctx.current_session
    if current_session is None:
        return CommandResult(text="当前会话未初始化，请先发送一条消息。")

    agent, error_text = await _load_session_agent(agent_pool, current_session)
    if error_text is not None:
        return CommandResult(text=error_text)
    if agent is None:
        return CommandResult(text="加载 Agent 失败: runtime agent unavailable")

    state = None
    state = await scheduler.get_state(current_session.id)
    scheduler_status = format_scheduler_status(state.status) if state else "未启动"

    metrics_summary = await summarize_runs_paginated(
        agent.run_step_storage,
        session_id=current_session.id,
    )

    opts = agent.options
    max_input_tokens_per_call = resolve_max_input_tokens_per_call(
        opts.max_input_tokens_per_call,
        agent.model,
    )
    max_context_window = resolve_max_context_window(agent.model)
    max_output_tokens = resolve_max_output_tokens(agent.model)
    cache_hit_price = float(getattr(agent.model, "cache_hit_price", 0.0) or 0.0)
    input_price = float(getattr(agent.model, "input_price", 0.0) or 0.0)
    output_price = float(getattr(agent.model, "output_price", 0.0) or 0.0)

    lines = [
        "当前对话统计\n",
        f"调度状态: {scheduler_status}",
        f"模型: {_format_model_info(agent)}",
        "Runs: "
        f"{int(metrics_summary.run_count)} "
        f"(完成 {int(metrics_summary.completed_run_count)})",
        "",
        "Token 用量:",
        f"  输入: {int(metrics_summary.input_tokens):,}",
        f"  输出: {int(metrics_summary.output_tokens):,}",
        f"  合计: {int(metrics_summary.total_tokens):,}",
        f"  Cache Read: {int(metrics_summary.cache_read_tokens):,}",
        f"  Cache Creation: {int(metrics_summary.cache_creation_tokens):,}",
        _format_cost(float(metrics_summary.token_cost)),
        "",
        f"总耗时: {float(metrics_summary.duration_ms) / 1000:.1f}s",
        "",
        "配置:",
        f"  max_steps: {opts.max_steps}",
        f"  run_timeout: {opts.run_timeout}s",
        f"  max_context_window: {max_context_window:,}",
        f"  max_output_tokens: {max_output_tokens:,}",
        f"  max_input_tokens_per_call: {max_input_tokens_per_call:,}",
        f"  cache_hit_price: ${cache_hit_price:.6f} / 1M tokens",
        f"  input_price: ${input_price:.6f} / 1M tokens",
        f"  output_price: ${output_price:.6f} / 1M tokens",
    ]
    if opts.max_run_cost is not None:
        lines.append(f"  max_run_cost: ${opts.max_run_cost:.4f}")

    return CommandResult(text="\n".join(lines))


async def _load_session_agent(
    agent_pool: AgentRuntimeCache,
    current_session: Session,
) -> tuple[Agent | None, str | None]:
    agent = agent_pool.runtime_agents.get(current_session.id)
    if agent is not None:
        return (agent, None)

    try:
        return (await agent_pool.get_or_create_runtime_agent(current_session), None)
    except Exception as exc:  # noqa: BLE001
        return (None, f"加载 Agent 失败: {exc}")


def _format_model_info(agent: Agent) -> str:
    model_info = agent.model.__class__.__name__
    if hasattr(agent.model, "name"):
        model_info += f" ({agent.model.name})"
    return model_info


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 20] + "\n\n...[已截断]"


def _format_cost(cost: float) -> str:
    return f"  费用: ${cost:.4f}"
