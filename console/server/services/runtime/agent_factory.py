"""Agent construction and recovery for the Console runtime."""

from agiwo.agent import Agent, AgentConfig
from agiwo.agent import (
    AgentOptions,
    AgentStorageOptions,
    RunStepStorageConfig,
    TraceStorageConfig,
)
from agiwo.llm import create_model_from_dict
from agiwo.llm.base import Model
from agiwo.scheduler.engine import Scheduler
from agiwo.scheduler.models import AgentState
from agiwo.skill.manager import get_global_skill_manager
from agiwo.tool.base import BaseTool

from server.config import ConsoleConfig, DefaultAgentConfig
from server.models.agent_config import AgentOptionsInput, ModelParamsInput
from server.services.agent_registry import AgentConfigRecord, AgentRegistry
from server.services.storage_wiring import (
    build_run_step_storage_config,
    build_trace_storage_config,
)
from server.services.tool_catalog.tool_builder import build_tools
from server.services.tool_catalog.tool_references import AGENT_TOOL_PREFIX


class PersistentAgentResumeError(RuntimeError):
    """Base error for persistent-agent resume failures."""


class PersistentAgentNotFoundError(PersistentAgentResumeError):
    """Raised when the target scheduler state does not exist."""


class PersistentAgentValidationError(PersistentAgentResumeError):
    """Raised when the target state cannot be resumed."""


def agent_options_input_to_agent_options(
    options_input: AgentOptionsInput,
    *,
    run_step_storage: RunStepStorageConfig,
    trace_storage: TraceStorageConfig,
) -> AgentOptions:
    data = options_input.model_dump(exclude_none=True)
    return AgentOptions(
        **data,
        storage=AgentStorageOptions(
            run_step_storage=run_step_storage,
            trace_storage=trace_storage,
        ),
    )


def build_default_agent_record(template: DefaultAgentConfig) -> AgentConfigRecord:
    allowed_skills = get_global_skill_manager().expand_allowed_skills(
        template.allowed_skills
    )
    return AgentConfigRecord(
        id=template.id,
        name=template.name,
        description=template.description,
        model_provider=template.model_provider,
        model_name=template.model_name,
        system_prompt=template.system_prompt,
        tools=list(template.tools),
        allowed_skills=allowed_skills or [],
        options=AgentOptionsInput.model_validate({}).model_dump(exclude_none=True),
        model_params=dict(template.model_params),
    )


def build_model(config: AgentConfigRecord) -> Model:
    model_params = ModelParamsInput.model_validate(config.model_params or {})
    return create_model_from_dict(
        provider=config.model_provider,
        model_name=config.model_name,
        params=model_params.model_dump(exclude_none=True),
    )


async def build_agent(
    config: AgentConfigRecord,
    console_config: ConsoleConfig,
    registry: AgentRegistry,
    *,
    id: str | None = None,
    _building: set[str] | None = None,
) -> Agent:
    if _building is None:
        _building = set()
    if config.id in _building:
        raise ValueError(f"Circular agent reference detected: {config.id}")
    _building.add(config.id)

    model = build_model(config)
    opts_input = AgentOptionsInput.model_validate(config.options or {})
    options = agent_options_input_to_agent_options(
        opts_input,
        run_step_storage=build_run_step_storage_config(console_config),
        trace_storage=build_trace_storage_config(console_config),
    )

    async def _build_agent_tool(ref: str) -> BaseTool | None:
        agent_id = ref[len(AGENT_TOOL_PREFIX) :]
        child_config = await registry.get_agent(agent_id)
        if child_config is None:
            return None
        child_agent = await build_agent(
            child_config,
            console_config,
            registry,
            _building=_building.copy(),
        )
        return child_agent.as_tool()

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
        allowed_skills=list(config.allowed_skills),
    )
    return Agent(
        agent_config,
        model=model,
        tools=tools or None,
        id=id or config.id,
    )


async def rehydrate_agent(
    state: AgentState,
    registry: AgentRegistry,
    console_config: ConsoleConfig,
) -> Agent:
    config_id = state.agent_config_id or state.id
    config = await registry.get_agent(config_id)
    if config is None:
        raise RuntimeError(
            f"Cannot rehydrate agent '{state.id}': "
            f"config '{config_id}' not found in registry"
        )

    return await build_agent(
        config,
        console_config,
        registry,
        id=state.id,
    )


async def resume_persistent_agent(
    scheduler: Scheduler,
    *,
    state_id: str,
    message: str,
    registry: AgentRegistry,
    console_config: ConsoleConfig,
) -> None:
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
        agent = await rehydrate_agent(
            state,
            registry,
            console_config,
        )

    await scheduler.enqueue_input(state_id, message, agent=agent)
