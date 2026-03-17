from dataclasses import dataclass

from agiwo.agent.config import AgentConfig
from agiwo.agent.hooks import AgentHooks
from agiwo.agent.options import AgentOptions
from agiwo.agent.runtime_tools import RuntimeToolLike
from agiwo.llm.base import Model


@dataclass(frozen=True, slots=True)
class ResolvedExecutionDefinition:
    """Immutable execution snapshot consumed by the runner/executor."""

    agent_id: str
    agent_name: str
    description: str
    model: Model
    hooks: AgentHooks
    options: AgentOptions
    tools: tuple[RuntimeToolLike, ...]
    system_prompt: str


@dataclass(frozen=True, slots=True)
class AgentCloneSpec:
    """Pure inputs for cloning a scheduler child Agent template."""

    agent_id: str
    config: AgentConfig
    hooks: AgentHooks
    tools: tuple[RuntimeToolLike, ...]


__all__ = ["AgentCloneSpec", "ResolvedExecutionDefinition"]
