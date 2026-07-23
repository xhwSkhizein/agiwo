"""Reusable agent template types (ADR 0049 / CONTEXT AgentSpec)."""

from dataclasses import dataclass

from agiwo.agent.models.config import AgentConfig


@dataclass(frozen=True)
class AgentSpec:
    """Reusable agent template (ADR 0049 / CONTEXT AgentSpec).

    Holds pure configuration only. Live ``Model``, tools, and hooks are supplied
    when binding a session-scoped :class:`~agiwo.agent.main_agent.MainAgent`, not
    on the spec itself. Must not hold message queues, run ids, abort signals, or
    stream subscribers.
    """

    config: AgentConfig
