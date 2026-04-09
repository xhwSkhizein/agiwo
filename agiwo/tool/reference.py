"""Tool reference abstraction for unified tool identification.

This module provides value objects for tool references, centralizing
the parsing and validation of tool identifiers like "agent:<id>".
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from agiwo.tool.builtin.registry import BUILTIN_TOOLS


class ToolReference(ABC):
    """Abstract base class for tool references.

    A tool reference is a pointer to a tool that may be resolved to
    different implementations in different contexts.
    """

    @property
    @abstractmethod
    def type(self) -> str:
        """Reference type identifier, e.g., 'builtin', 'agent', 'skill'."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_resolved(self) -> bool:
        """Whether this reference can be directly used as a tool instance."""
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        """Serialize to string format for storage/transmission."""
        raise NotImplementedError


@dataclass(frozen=True)
class BuiltinToolReference(ToolReference):
    """Reference to a builtin tool by name."""

    name: str

    @property
    def type(self) -> str:
        return "builtin"

    @property
    def is_resolved(self) -> bool:
        return True

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class AgentToolReference(ToolReference):
    """Reference to an agent-as-tool.

    Format: "agent:<agent_id>"

    Example:
        >>> ref = AgentToolReference.parse("agent:my-agent")
        >>> ref.agent_id
        'my-agent'
        >>> str(ref)
        'agent:my-agent'
    """

    agent_id: str
    PREFIX: ClassVar[str] = "agent:"

    @property
    def type(self) -> str:
        return "agent"

    @property
    def is_resolved(self) -> bool:
        return False

    @classmethod
    def parse(cls, value: str) -> "AgentToolReference | None":
        """Parse a string as an agent tool reference.

        Args:
            value: The string to parse.

        Returns:
            AgentToolReference if value starts with "agent:".
            None if value does not start with "agent:".

        Raises:
            InvalidToolReferenceError: If the reference is malformed
                (e.g., "agent:" with empty agent_id).
        """
        if not value.startswith(cls.PREFIX):
            return None

        agent_id = value[len(cls.PREFIX) :].strip()
        if not agent_id:
            raise InvalidToolReferenceError(f"Empty agent id in reference: {value!r}")

        return cls(agent_id=agent_id)

    def __str__(self) -> str:
        return f"{self.PREFIX}{self.agent_id}"


class InvalidToolReferenceError(ValueError):
    """Raised when a tool reference string is malformed."""

    pass


def parse_tool_reference(value: str) -> ToolReference:
    """Parse any tool reference string.

    Resolution order:
    1. Agent reference ("agent:<id>")
    2. Builtin tool reference

    Args:
        value: The reference string to parse.

    Returns:
        A ToolReference subclass instance.

    Raises:
        InvalidToolReferenceError: If the reference cannot be parsed.
    """
    # 1. Try agent reference
    agent_ref = AgentToolReference.parse(value)
    if agent_ref is not None:
        return agent_ref

    # 2. Try builtin tool
    if value in BUILTIN_TOOLS:
        return BuiltinToolReference(name=value)

    raise InvalidToolReferenceError(f"Unknown tool reference: {value!r}")
