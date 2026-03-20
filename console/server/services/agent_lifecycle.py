"""
Agent lifecycle management — build, rehydrate, and resume agents.

Consolidates all Agent construction paths:
- build_agent: from AgentConfigRecord (new agent)
- rehydrate_agent: from AgentState (server restart recovery)
- resume_persistent_agent: wake a persistent agent with new task
"""

from typing import Any

from agiwo.agent import Agent, AgentConfig
from agiwo.agent.runtime_tools import RuntimeToolLike
from agiwo.agent.options import AgentOptions
from agiwo.llm.base import Model
from agiwo.llm import create_model_from_dict
from agiwo.scheduler.models import AgentState
from agiwo.scheduler.scheduler import Scheduler

from server.config import ConsoleConfig
from server.domain.agent_configs import AgentOptionsInput, ModelParamsInput
from server.services.agent_registry import AgentConfigRecord, AgentRegistry
from server.services.storage_wiring import (
    build_run_step_storage_config,
    build_trace_storage_config,
)
from server.tools import AgentToolRef, build_agent_tool, build_tools


# ── Exceptions ────────────────────────────────────────────────────────


class PersistentAgentResumeError(RuntimeError):
    """Base error for persistent-agent resume failures."""


class PersistentAgentNotFoundError(PersistentAgentResumeError):
    """Raised when the target scheduler state does not exist."""


class PersistentAgentValidationError(PersistentAgentResumeError):
    """Raised when the target state cannot be resumed."""


# ── Defaults ──────────────────────────────────────────────────────────


def build_default_agent_options() -> dict[str, Any]:
    """Return the canonical default agent options payload."""
    return AgentOptionsInput.model_validate({}).model_dump(exclude_none=True)


# ── Build ─────────────────────────────────────────────────────────────


def build_model(config: AgentConfigRecord) -> Model:
    """Build a Model instance from agent config."""
    model_params = ModelParamsInput.model_validate(config.model_params or {})
    return create_model_from_dict(
        provider=config.model_provider,
        model_name=config.model_name,
        params=model_params.model_dump(exclude_none=True),
    )


def build_agent_options(
    config: AgentConfigRecord, console_config: ConsoleConfig
) -> AgentOptions:
    """Build AgentOptions with storage config matching the console storage backend."""
    opts = AgentOptionsInput.model_validate(config.options or {})
    return opts.to_agent_options(
        run_step_storage=build_run_step_storage_config(console_config),
        trace_storage=build_trace_storage_config(console_config),
    )


async def build_agent(
    config: AgentConfigRecord,
    console_config: ConsoleConfig,
    registry: AgentRegistry,
    *,
    id: str | None = None,
    _building: set[str] | None = None,
) -> Agent:
    """Build an Agent instance from persisted config.

    The _building set prevents circular agent references.
    """
    if _building is None:
        _building = set()
    if config.id in _building:
        raise ValueError(f"Circular agent reference detected: {config.id}")
    _building.add(config.id)

    model = build_model(config)
    options = build_agent_options(config, console_config)

    async def _build_agent_tool(ref: AgentToolRef) -> RuntimeToolLike | None:
        child_config = await registry.get_agent(ref.agent_id)
        if child_config is None:
            return None
        child_agent = await build_agent(
            child_config,
            console_config,
            registry,
            _building=_building.copy(),
        )
        return build_agent_tool(ref, child_agent)

    tools = await build_tools(
        config.tools or [],
        console_config=console_config,
        build_agent_tool=_build_agent_tool,
    )

    agent_config = AgentConfig(
        name=config.name,
        description=config.description,
        system_prompt=config.system_prompt,
        options=options,
    )
    return Agent(
        agent_config,
        model=model,
        tools=tools or None,
        id=id,
    )


# ── Rehydrate ─────────────────────────────────────────────────────────


async def rehydrate_agent(
    state: AgentState,
    registry: AgentRegistry,
    console_config: ConsoleConfig,
) -> Agent:
    """Rebuild an Agent from its persisted state and config registry.

    Uses ``state.agent_config_id`` (if set) to look up the original
    ``AgentConfigRecord``, then delegates to ``build_agent`` to construct the
    full Agent with model, tools, and options.

    The returned Agent's ``id`` is set to ``state.id`` (the runtime instance
    ID) so that the Scheduler can match it back to the correct AgentState.

    Raises:
        RuntimeError: If the config record cannot be found.
    """
    config_id = state.agent_config_id or state.id
    config = await registry.get_agent(config_id)
    if config is None:
        raise RuntimeError(
            f"Cannot rehydrate agent '{state.id}': "
            f"config '{config_id}' not found in registry"
        )

    return await build_agent(config, console_config, registry, id=state.id)


# ── Resume ────────────────────────────────────────────────────────────


async def resume_persistent_agent(
    scheduler: Scheduler,
    *,
    state_id: str,
    message: str,
    registry: AgentRegistry,
    console_config: ConsoleConfig,
) -> None:
    """Resume a persistent root agent, rehydrating it if needed."""
    state = await scheduler.get_state(state_id)
    if state is None:
        raise PersistentAgentNotFoundError("Agent state not found")

    if not state.is_persistent:
        raise PersistentAgentValidationError(
            "Only persistent root agents can be resumed"
        )
    if state.parent_id is not None:
        raise PersistentAgentValidationError("Cannot resume child agents")

    agent = scheduler.get_registered_agent(state_id)
    if agent is None:
        agent = await rehydrate_agent(state, registry, console_config)

    await scheduler.enqueue_input(state_id, message, agent=agent)
