"""
Session management commands: /new and /list.
"""

from datetime import datetime, timezone
from uuid import uuid4

from agiwo.agent.agent import Agent
from agiwo.scheduler.models import AgentStateStatus
from agiwo.scheduler.scheduler import Scheduler

from server.channels.feishu.commands.base import CommandContext, CommandHandler, CommandResult
from server.channels.feishu.store import FeishuChannelStore
from server.channels.models import SessionRuntime
from server.channels.session_manager import SessionManager


class NewSessionCommand(CommandHandler):
    """Create a fresh conversation, resetting the current session context."""

    def __init__(
        self,
        store: FeishuChannelStore,
        scheduler: Scheduler,
        runtime_agents: dict[str, Agent],
        session_manager: SessionManager,
    ) -> None:
        self._store = store
        self._scheduler = scheduler
        self._runtime_agents = runtime_agents
        self._session_manager = session_manager

    @property
    def name(self) -> str:
        return "new"

    @property
    def description(self) -> str:
        return "创建新会话，重置当前对话上下文"

    async def execute(self, ctx: CommandContext, args: str) -> CommandResult:
        runtime = ctx.runtime
        if runtime is None:
            return CommandResult(text="当前没有活跃会话，发送消息即可自动创建。")

        state = await self._scheduler.get_state(runtime.scheduler_state_id)
        if state is not None and state.status in (
            AgentStateStatus.RUNNING,
            AgentStateStatus.SLEEPING,
            AgentStateStatus.PENDING,
        ):
            await self._scheduler.cancel(runtime.scheduler_state_id, "用户执行 /new 重置会话")

        cached_agent = self._runtime_agents.pop(runtime.runtime_agent_id, None)
        if cached_agent is not None:
            await cached_agent.close()

        runtime.agiwo_session_id = str(uuid4())
        runtime.scheduler_state_id = runtime.runtime_agent_id
        runtime.updated_at = datetime.now(timezone.utc)
        await self._store.upsert_session_runtime(runtime)

        self._session_manager.reset_session(ctx.session_key)

        return CommandResult(text="新会话已创建，对话上下文已重置。")


class ListSessionsCommand(CommandHandler):
    """List the user's session runtimes across chats."""

    def __init__(
        self,
        store: FeishuChannelStore,
        scheduler: Scheduler,
    ) -> None:
        self._store = store
        self._scheduler = scheduler

    @property
    def name(self) -> str:
        return "list"

    @property
    def description(self) -> str:
        return "列出历史会话和概览"

    async def execute(self, ctx: CommandContext, args: str) -> CommandResult:
        runtimes = await self._store.list_session_runtimes_by_user(
            ctx.trigger_user_open_id
        )
        if not runtimes:
            return CommandResult(text="暂无会话记录。")

        runtimes.sort(key=lambda r: r.updated_at, reverse=True)

        lines: list[str] = [f"会话列表 (共 {len(runtimes)} 个):\n"]
        for i, rt in enumerate(runtimes, 1):
            is_current = rt.session_key == ctx.session_key
            marker = " [当前]" if is_current else ""
            chat_label = "私聊" if rt.chat_type == "p2p" else f"群聊 {rt.chat_id[:8]}..."

            status_text = await self._resolve_status_text(rt)

            lines.append(
                f"{i}. {chat_label}{marker}\n"
                f"   状态: {status_text}\n"
                f"   会话ID: {rt.agiwo_session_id[:8]}...\n"
                f"   更新于: {self._format_time(rt.updated_at)}"
            )

        return CommandResult(text="\n".join(lines))

    async def _resolve_status_text(self, runtime: SessionRuntime) -> str:
        state = await self._scheduler.get_state(runtime.scheduler_state_id)
        if state is None:
            return "未启动"
        status_map = {
            AgentStateStatus.PENDING: "等待中",
            AgentStateStatus.RUNNING: "运行中",
            AgentStateStatus.SLEEPING: "空闲",
            AgentStateStatus.COMPLETED: "已完成",
            AgentStateStatus.FAILED: "已失败",
        }
        return status_map.get(state.status, state.status.value)

    def _format_time(self, dt: datetime) -> str:
        local = dt.astimezone()
        return local.strftime("%m-%d %H:%M")
