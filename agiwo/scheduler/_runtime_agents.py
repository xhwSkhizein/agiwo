"""Scheduler-managed runtime-agent cloning and reuse helpers."""

from collections.abc import Sequence

from agiwo.agent import Agent
from agiwo.scheduler.runtime_state import RuntimeState
from agiwo.tool.base import BaseTool
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


async def ensure_root_runtime_agent(
    *,
    rt: RuntimeState,
    scheduling_tools: Sequence[BaseTool],
    canonical_agent: Agent,
    state_id: str,
) -> Agent:
    """Ensure a scheduler-managed runtime agent exists for ``state_id``.

    Identity rule:
    - if the cached canonical agent for ``state_id`` is the same Python object
      as the caller-supplied one, reuse the existing runtime agent
    - otherwise clone a new runtime agent, inject scheduler system tools,
      replace the cached runtime agent, and close the old one
    """
    cached_canonical = rt.canonical_agents.get(state_id)
    cached_runtime = rt.agents.get(state_id)
    if cached_canonical is canonical_agent and cached_runtime is not None:
        return cached_runtime

    runtime_agent = Agent(
        canonical_agent.config,
        id=state_id,
        model=canonical_agent.model,
        tools=list(canonical_agent.extra_tools) or None,
        hooks=canonical_agent.hooks,
    )
    runtime_agent._inject_system_tools(list(scheduling_tools))

    rt.agents[state_id] = runtime_agent
    rt.canonical_agents[state_id] = canonical_agent
    if cached_runtime is not None:
        try:
            await cached_runtime.close()
        except Exception:  # noqa: BLE001 - runtime-agent close boundary
            logger.exception(
                "scheduler_root_runtime_agent_close_failed",
                state_id=state_id,
            )
    return runtime_agent


__all__ = ["ensure_root_runtime_agent"]
