"""Channel-specific exceptions for type-safe error handling."""


class ChannelError(RuntimeError):
    """Base exception for channel-related errors."""


class PreviousTaskRunningError(ChannelError):
    """Raised when a previous task is still running after timeout."""


class BaseAgentNotFoundError(ChannelError):
    """Raised when the base agent does not exist or has been deleted."""

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        super().__init__(f"base_agent_not_found: {agent_name}")


class DefaultAgentNameNotFoundError(ChannelError):
    """Raised when the default agent name does not exist in registry."""

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        super().__init__(f"default_agent_name_not_found: {agent_name}")
