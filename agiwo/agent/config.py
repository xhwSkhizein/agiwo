"""Pure public configuration for Agent construction."""

from dataclasses import dataclass, field

from agiwo.agent.options import AgentOptions


@dataclass
class AgentConfig:
    """Canonical public Agent configuration without live runtime objects."""

    name: str
    description: str = ""
    system_prompt: str = ""
    options: AgentOptions = field(default_factory=AgentOptions)
