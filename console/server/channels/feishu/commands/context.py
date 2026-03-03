"""
Runtime context inspection commands: /context and /status.
"""

from agiwo.agent.agent import Agent
from agiwo.agent.schema import Run, RunStatus
from agiwo.scheduler.models import AgentStateStatus
from agiwo.scheduler.scheduler import Scheduler

from server.channels.feishu.commands.base import CommandContext, CommandHandler, CommandResult


_PROMPT_PREVIEW_MAX_LEN = 5000


class ContextCommand(CommandHandler):
    """Show the current session's system prompt and configuration overview."""

    def __init__(
        self,
        runtime_agents: dict[str, Agent],
    ) -> None:
        self._runtime_agents = runtime_agents

    @property
    def name(self) -> str:
        return "context"

    @property
    def description(self) -> str:
        return "查看当前会话的 SystemPrompt 和概览"

    async def execute(self, ctx: CommandContext, args: str) -> CommandResult:
        if ctx.runtime is None:
            return CommandResult(text="当前会话未初始化，请先发送一条消息。")

        agent = self._runtime_agents.get(ctx.runtime.runtime_agent_id)
        if agent is None:
            return CommandResult(text="Agent 尚未加载，请先发送一条消息。")

        prompt = await agent.get_effective_system_prompt()
        prompt_preview = self._truncate(prompt, _PROMPT_PREVIEW_MAX_LEN)

        tool_names = [t.get_name() for t in agent.tools]
        model_info = f"{agent.model.__class__.__name__}"
        if hasattr(agent.model, "name"):
            model_info += f" ({agent.model.name})"

        lines = [
            f"Agent: {agent.name} (id: {agent.id})",
            f"Model: {model_info}",
            f"Tools ({len(tool_names)}): {', '.join(tool_names)}",
            f"Session: {ctx.runtime.agiwo_session_id[:12]}...",
            "",
            "--- System Prompt ---",
            prompt_preview,
        ]
        return CommandResult(text="\n".join(lines))

    def _truncate(self, text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        return text[: max_len - 20] + "\n\n...[已截断]"


class StatusCommand(CommandHandler):
    """Show aggregate statistics for the current conversation."""

    def __init__(
        self,
        runtime_agents: dict[str, Agent],
        scheduler: Scheduler,
    ) -> None:
        self._runtime_agents = runtime_agents
        self._scheduler = scheduler

    @property
    def name(self) -> str:
        return "status"

    @property
    def description(self) -> str:
        return "查看当前对话的统计信息"

    async def execute(self, ctx: CommandContext, args: str) -> CommandResult:
        if ctx.runtime is None:
            return CommandResult(text="当前会话未初始化，请先发送一条消息。")

        agent = self._runtime_agents.get(ctx.runtime.runtime_agent_id)
        if agent is None:
            return CommandResult(text="Agent 尚未加载，请先发送一条消息。")

        state = await self._scheduler.get_state(ctx.runtime.scheduler_state_id)
        scheduler_status = state.status.value if state else "未启动"

        runs = await agent.run_step_storage.list_runs(
            session_id=ctx.runtime.agiwo_session_id,
            limit=200,
        )

        total_tokens, total_input, total_output = 0, 0, 0
        total_cost = 0.0
        total_duration_ms = 0.0
        completed_runs = 0
        for run in runs:
            m = run.metrics
            total_tokens += m.total_tokens
            total_input += m.input_tokens
            total_output += m.output_tokens
            total_cost += m.token_cost
            total_duration_ms += m.duration_ms
            if run.status == RunStatus.COMPLETED:
                completed_runs += 1

        model_info = f"{agent.model.__class__.__name__}"
        if hasattr(agent.model, "name"):
            model_info += f" ({agent.model.name})"

        opts = agent.options
        lines = [
            "当前对话统计\n",
            f"调度状态: {scheduler_status}",
            f"模型: {model_info}",
            f"Runs: {len(runs)} (完成 {completed_runs})",
            "",
            "Token 用量:",
            f"  输入: {total_input:,}",
            f"  输出: {total_output:,}",
            f"  合计: {total_tokens:,}",
            self._format_cost(total_cost),
            "",
            f"总耗时: {total_duration_ms / 1000:.1f}s",
            "",
            "配置:",
            f"  max_steps: {opts.max_steps}",
            f"  run_timeout: {opts.run_timeout}s",
            f"  max_context_window_tokens: {opts.max_context_window_tokens:,}",
            f"  max_tokens_per_run: {opts.max_tokens_per_run:,}",
        ]
        if opts.max_run_token_cost is not None:
            lines.append(f"  max_run_token_cost: ${opts.max_run_token_cost:.4f}")

        return CommandResult(text="\n".join(lines))

    def _format_cost(self, cost: float) -> str:
        if cost > 0:
            return f"  费用: ${cost:.4f}"
        return ""
