"""
Command system base abstractions.

Provides the core types for the channel command protocol:
- CommandContext: request-scoped context
- CommandResult: handler response
- CommandHandler: abstract handler interface
- CommandRegistry: parse + dispatch
- HelpCommand: auto-generated help listing
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from server.channels.models import SessionRuntime


@dataclass
class CommandContext:
    """Request-scoped context passed to every command handler."""

    session_key: str
    chat_id: str
    chat_type: str
    trigger_user_open_id: str
    trigger_message_id: str
    base_agent_id: str
    runtime: SessionRuntime | None


@dataclass
class CommandResult:
    """Response produced by a command handler."""

    text: str


class CommandHandler(ABC):
    """Base class for all slash-command handlers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Command name without the leading slash (e.g. ``"new"``)."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description shown in /help."""
        ...

    @abstractmethod
    async def execute(self, ctx: CommandContext, args: str) -> CommandResult:
        """Execute the command and return a text response."""
        ...


class CommandRegistry:
    """Parses incoming text for ``/command`` patterns and dispatches to handlers."""

    def __init__(self) -> None:
        self._handlers: dict[str, CommandHandler] = {}

    def register(self, handler: CommandHandler) -> None:
        self._handlers[handler.name] = handler

    @property
    def handlers(self) -> dict[str, CommandHandler]:
        return self._handlers

    def try_parse(self, text: str) -> tuple[CommandHandler, str] | None:
        """If *text* is a recognized command, return ``(handler, args)``.

        Returns ``None`` for non-command text or unknown commands.
        """
        stripped = text.strip()
        if not stripped.startswith("/"):
            return None
        parts = stripped.split(maxsplit=1)
        cmd_name = parts[0][1:]
        if not cmd_name:
            return None
        args = parts[1] if len(parts) > 1 else ""
        handler = self._handlers.get(cmd_name)
        if handler is None:
            return None
        return (handler, args)


class HelpCommand(CommandHandler):
    """Lists all registered commands and their descriptions."""

    def __init__(self, registry: CommandRegistry) -> None:
        self._registry = registry

    @property
    def name(self) -> str:
        return "help"

    @property
    def description(self) -> str:
        return "显示可用命令列表"

    async def execute(self, ctx: CommandContext, args: str) -> CommandResult:
        lines = ["可用命令:\n"]
        for cmd_name in sorted(self._registry.handlers):
            handler = self._registry.handlers[cmd_name]
            lines.append(f"  /{cmd_name} — {handler.description}")
        return CommandResult(text="\n".join(lines))
