"""Feishu channel command protocol."""

from agiwo.scheduler.scheduler import Scheduler

from server.channels.feishu.commands.base import (
    CommandContext,
    CommandHandler,
    CommandRegistry,
    CommandResult,
    CommandSpec,
    build_command_registry,
)
from server.channels.feishu.commands.context import build_context_command_specs
from server.channels.feishu.commands.scheduler import build_scheduler_command_specs
from server.channels.feishu.commands.session import build_session_command_specs
from server.channels.runtime_agent_pool import RuntimeAgentPool
from server.channels.session import SessionContextService, SessionManager
from server.channels.session.models import ChannelChatSessionStore
from server.config import ConsoleConfig
from server.services.agent_registry import AgentRegistry


def build_feishu_command_registry(
    *,
    session_service: SessionContextService,
    agent_pool: RuntimeAgentPool,
    session_manager: SessionManager,
    scheduler: Scheduler,
    agent_registry: AgentRegistry,
    console_config: ConsoleConfig,
    store: ChannelChatSessionStore | None = None,
) -> CommandRegistry:
    workspace_session_service = session_service.as_remote_workspace_service()
    specs: list[CommandSpec] = [
        *build_session_command_specs(
            workspace_session_service,
            session_manager,
            scheduler,
        ),
        *build_context_command_specs(agent_pool, scheduler),
        *build_scheduler_command_specs(
            scheduler,
            agent_registry,
            console_config,
        ),
    ]
    return build_command_registry(specs)


__all__ = [
    "CommandContext",
    "CommandHandler",
    "CommandRegistry",
    "CommandResult",
    "CommandSpec",
    "build_command_registry",
    "build_feishu_command_registry",
]
