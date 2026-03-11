"""Feishu channel command protocol."""

from agiwo.scheduler.scheduler import Scheduler

from server.channels.agent_runtime import AgentRuntimeManager
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
from server.channels.session_manager import SessionManager
from server.config import ConsoleConfig
from server.services.agent_registry import AgentRegistry


def build_feishu_command_registry(
    *,
    runtime_mgr: AgentRuntimeManager,
    session_manager: SessionManager,
    scheduler: Scheduler,
    agent_registry: AgentRegistry,
    console_config: ConsoleConfig,
) -> CommandRegistry:
    specs: list[CommandSpec] = [
        *build_session_command_specs(runtime_mgr, session_manager, scheduler),
        *build_context_command_specs(runtime_mgr, scheduler),
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
