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

    Fields will grow with spec-reuse scenarios (e.g. multiple Sessions bound to
    the same template); do not delete this type merely because it currently
    wraps ``AgentConfig``.
    """

    config: AgentConfig
