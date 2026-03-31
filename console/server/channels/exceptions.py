"""Channel-specific exceptions for type-safe error handling."""


class ChannelError(RuntimeError):
    """Base exception for channel-related errors."""


class PreviousTaskRunningError(ChannelError):
    """Raised when a previous task is still running after timeout."""


class BaseAgentNotFoundError(ChannelError):
    """Raised when the base agent does not exist or has been deleted."""

    def __init__(self, base_agent_id: str) -> None:
        self.base_agent_id = base_agent_id
        super().__init__(f"base_agent_not_found: {base_agent_id}")


class DefaultAgentNameNotFoundError(ChannelError):
    """Raised when the default agent name does not exist in registry."""

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        super().__init__(f"default_agent_name_not_found: {agent_name}")


class SessionNotFoundError(ChannelError):
    """Raised when a session is not found."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session not found: {session_id}")


class ChatContextNotFoundError(ChannelError):
    """Raised when a chat context is not found."""

    def __init__(self, scope_id: str) -> None:
        self.scope_id = scope_id
        super().__init__(f"Chat context not found: {scope_id}")


class SessionNotInChatContextError(ChannelError):
    """Raised when a session does not belong to the specified chat context."""

    def __init__(self, session_id: str, scope_id: str) -> None:
        self.session_id = session_id
        self.scope_id = scope_id
        super().__init__(f"Session {session_id} not in chat context {scope_id}")
