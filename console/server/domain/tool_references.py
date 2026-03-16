"""Canonical console tool-reference contract and strict parsing."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from agiwo.tool.builtin.registry import BUILTIN_TOOLS, ensure_builtin_tools_loaded

ensure_builtin_tools_loaded()

AGENT_TOOL_PREFIX = "agent:"


class InvalidToolReferenceError(ValueError):
    def __init__(self, value: object) -> None:
        super().__init__(f"Invalid tool reference: {value!r}")
        self.value = value


class BuiltinToolRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    kind: Literal["builtin"] = "builtin"

    def to_storage_name(self) -> str:
        return self.name


class AgentToolRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    agent_id: str
    kind: Literal["agent"] = "agent"

    def to_storage_name(self) -> str:
        return f"{AGENT_TOOL_PREFIX}{self.agent_id}"


ToolReference = BuiltinToolRef | AgentToolRef


def parse_tool_reference(value: object) -> ToolReference:
    if isinstance(value, BuiltinToolRef | AgentToolRef):
        return value
    if isinstance(value, dict):
        kind = value.get("kind")
        if kind == "builtin":
            ref = BuiltinToolRef.model_validate(value)
            _validate_builtin_name(ref.name)
            return ref
        if kind == "agent":
            ref = AgentToolRef.model_validate(value)
            if not ref.agent_id.strip():
                raise InvalidToolReferenceError(value)
            return ref
        raise InvalidToolReferenceError(value)
    if not isinstance(value, str):
        raise InvalidToolReferenceError(value)

    normalized = value.strip()
    if not normalized:
        raise InvalidToolReferenceError(value)
    if normalized.startswith(AGENT_TOOL_PREFIX):
        agent_id = normalized[len(AGENT_TOOL_PREFIX):].strip()
        if not agent_id:
            raise InvalidToolReferenceError(value)
        return AgentToolRef(agent_id=agent_id)

    _validate_builtin_name(normalized)
    return BuiltinToolRef(name=normalized)


def parse_tool_references(values: list[object]) -> list[ToolReference]:
    return [parse_tool_reference(value) for value in values]


def serialize_tool_reference(ref: ToolReference) -> str:
    return ref.to_storage_name()


def serialize_tool_references(refs: list[ToolReference]) -> list[str]:
    return [serialize_tool_reference(ref) for ref in refs]


def _validate_builtin_name(name: str) -> None:
    if name not in BUILTIN_TOOLS:
        raise InvalidToolReferenceError(name)


__all__ = [
    "AGENT_TOOL_PREFIX",
    "AgentToolRef",
    "BuiltinToolRef",
    "InvalidToolReferenceError",
    "ToolReference",
    "parse_tool_reference",
    "parse_tool_references",
    "serialize_tool_reference",
    "serialize_tool_references",
]
