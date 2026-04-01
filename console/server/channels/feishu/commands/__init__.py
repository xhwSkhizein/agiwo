"""Feishu channel command protocol."""

from agiwo.scheduler.engine import Scheduler

from server.channels.feishu.commands.base import (
    CommandContext,
    CommandRegistry,
    CommandSpec,
    build_command_registry,
)
from server.channels.feishu.commands.context import build_context_command_specs
from server.channels.feishu.commands.scheduler import build_scheduler_command_specs
from server.channels.feishu.commands.session import build_session_command_specs
from server.channels.session import SessionManager
from server.config import ConsoleConfig
from server.services.agent_registry import AgentRegistry
from server.services.runtime import AgentRuntimeCache, SessionContextService


def build_feishu_command_registry(
    *,
    session_service: SessionContextService,
    agent_pool: AgentRuntimeCache,
    session_manager: SessionManager,
    scheduler: Scheduler,
    agent_registry: AgentRegistry,
    console_config: ConsoleConfig,
) -> CommandRegistry:
    specs: list[CommandSpec] = [
        *build_session_command_specs(
            session_service,
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
    "CommandRegistry",
    "build_feishu_command_registry",
]
