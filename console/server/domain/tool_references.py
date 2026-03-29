"""Canonical console tool-reference validation and parsing.

Tool references are plain strings:
- Builtin tools: just the tool name (e.g. "web_search")
- Agent tools: prefixed with "agent:" (e.g. "agent:child-1")
"""

from agiwo.tool.builtin.registry import BUILTIN_TOOLS, ensure_builtin_tools_loaded

ensure_builtin_tools_loaded()

AGENT_TOOL_PREFIX = "agent:"


class InvalidToolReferenceError(ValueError):
    def __init__(self, value: object) -> None:
        super().__init__(f"Invalid tool reference: {value!r}")
        self.value = value


def parse_tool_reference(value: object) -> str:
    """Validate and normalize a single tool reference to its canonical string form."""
    if not isinstance(value, str):
        raise InvalidToolReferenceError(value)

    normalized = value.strip()
    if not normalized:
        raise InvalidToolReferenceError(value)

    if normalized.startswith(AGENT_TOOL_PREFIX):
        agent_id = normalized[len(AGENT_TOOL_PREFIX) :].strip()
        if not agent_id:
            raise InvalidToolReferenceError(value)
        return f"{AGENT_TOOL_PREFIX}{agent_id}"

    if normalized not in BUILTIN_TOOLS:
        raise InvalidToolReferenceError(normalized)
    return normalized


def parse_tool_references(values: list[object]) -> list[str]:
    return [parse_tool_reference(value) for value in values]


__all__ = [
    "AGENT_TOOL_PREFIX",
    "InvalidToolReferenceError",
    "parse_tool_reference",
    "parse_tool_references",
]
