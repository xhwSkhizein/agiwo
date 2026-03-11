"""
Agent lifecycle management — build, rehydrate, and resume agents.

Consolidates all Agent construction paths:
- build_agent: from AgentConfigRecord (new agent)
- rehydrate_agent: from AgentState (server restart recovery)
- resume_persistent_agent: wake a persistent agent with new task
"""

import json
from typing import Any

from agiwo.agent.agent import Agent
from agiwo.agent.options import AgentOptions, AgentOptionsInput
from agiwo.agent.schema import StreamEvent
from agiwo.agent.serialization import serialize_stream_event_payload
from agiwo.llm.base import Model
from agiwo.llm import create_model_from_dict
from agiwo.llm.factory import ModelParamsInput
from agiwo.scheduler.models import AgentState
from agiwo.scheduler.scheduler import Scheduler

from server.config import ConsoleConfig
from server.services.agent_registry import AgentConfigRecord, AgentRegistry
from server.services.storage_manager import (
    build_run_step_storage_config,
    build_trace_storage_config,
)
from server.tools import AgentToolRef, console_tool_catalog


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


def build_agent_options(config: AgentConfigRecord, console_config: ConsoleConfig) -> AgentOptions:
    """Build AgentOptions with storage config matching the console storage backend."""
    opts = AgentOptionsInput.model_validate(config.options or {})

    return opts.to_agent_options(
        run_step_storage=build_run_step_storage_config(console_config),
        trace_storage=build_trace_storage_config(console_config),
    )


def serialize_event(event: StreamEvent) -> str:
    """Serialize a StreamEvent to JSON string for SSE data field."""
    return json.dumps(serialize_stream_event_payload(event), default=str)


async def build_agent(
    config: AgentConfigRecord,
    console_config: ConsoleConfig,
    registry: AgentRegistry,
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

    async def _build_agent_tool(ref: AgentToolRef):
        child_config = await registry.get_agent(ref.agent_id)
        if child_config is None:
            return None
        child_agent = await build_agent(
            child_config,
            console_config,
            registry,
            _building.copy(),
        )
        return console_tool_catalog.build_agent_tool(ref, child_agent)

    tools = await console_tool_catalog.build_tools(
        config.tools or [],
        console_config=console_config,
        build_agent_tool=_build_agent_tool,
    )

    return Agent(
        name=config.name,
        description=config.description,
        model=model,
        tools=tools or None,
        system_prompt=config.system_prompt,
        options=options,
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

    agent = await build_agent(config, console_config, registry)
    agent.id = state.id
    return agent


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

    await scheduler.submit_task(state_id, message, agent=agent)
