"""Agents CRUD API router."""

from fastapi import APIRouter, HTTPException, Query as QueryParam
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from agiwo.config.settings import (
    ALL_MODEL_PROVIDERS,
    COMPATIBLE_MODEL_PROVIDERS,
    load_settings,
)
from agiwo.skill.manager import get_global_skill_manager
from agiwo.utils.serialization import serialize_optional_datetime
from server.dependencies import (
    ConsoleRuntimeDep,
    get_session_context_service,
    get_session_view_service,
)
from server.response_serialization import session_summary_response_from_record
from server.models.view import (
    AgentCapabilitiesResponse,
    AgentConfigPayload,
    AgentConfigResponse,
    AgentProviderCapabilityResponse,
    AgentOptionsInput,
    ModelParamsInput,
    PageResponse,
    SessionSummaryResponse,
)
from server.services.agent_registry import AgentConfigRecord
from server.services.tool_catalog.tool_catalog import (
    list_available_tools as _list_available_tools,
)

router = APIRouter(prefix="/api/agents", tags=["agents"])


@router.get("/tools/available")
async def list_available_tools(
    runtime: ConsoleRuntimeDep,
    exclude: str | None = QueryParam(
        default=None, description="Agent ID to exclude from agent tools"
    ),
) -> list[dict[str, str]]:
    """Return all available tools (builtin + agent-as-tool) that can be assigned to agents."""
    return await _list_available_tools(
        runtime.agent_registry,
        exclude_agent_id=exclude,
    )


def _body_to_record(body: AgentConfigPayload) -> AgentConfigRecord:
    return AgentConfigRecord(
        name=body.name,
        description=body.description,
        model_provider=body.model_provider,
        model_name=body.model_name,
        system_prompt=body.system_prompt,
        tools=body.tools,
        allowed_skills=get_global_skill_manager().expand_allowed_skills(
            body.allowed_skills
        )
        or [],
        options=body.options.model_dump(exclude_none=True),
        model_params=body.model_params.model_dump(exclude_none=True),
    )


def _record_to_response(
    record: AgentConfigRecord,
    *,
    default_agent_id: str,
) -> AgentConfigResponse:
    return AgentConfigResponse(
        id=record.id,
        name=record.name,
        description=record.description,
        is_default=record.id == default_agent_id,
        model_provider=record.model_provider,
        model_name=record.model_name,
        system_prompt=record.system_prompt,
        tools=record.tools,
        allowed_skills=record.allowed_skills,
        options=AgentOptionsInput.model_validate(record.options or {}),
        model_params=ModelParamsInput.model_validate(record.model_params or {}),
        created_at=serialize_optional_datetime(record.created_at) or "",
        updated_at=serialize_optional_datetime(record.updated_at) or "",
    )


def _provider_label(value: str) -> str:
    if value == "openai":
        return "OpenAI"
    if value == "deepseek":
        return "DeepSeek"
    if value == "anthropic":
        return "Anthropic"
    if value == "nvidia":
        return "Nvidia"
    return value.replace("-", " ").title()


def _provider_default_model_name(provider: str) -> str | None:
    settings = load_settings(include_env_file=False)
    attr_map = {
        "openai": "openai_model_name",
        "deepseek": "deepseek_model_name",
        "anthropic": "anthropic_model_name",
        "nvidia": "nvidia_model_name",
    }
    attr_name = attr_map.get(provider)
    if attr_name is None:
        return None
    return getattr(settings, attr_name, None)


@router.get("/skills/available")
async def list_available_skills() -> list[dict[str, str]]:
    """Return all globally discovered skills."""
    skill_manager = get_global_skill_manager()
    await skill_manager.initialize()
    return [
        {"name": meta.name, "description": meta.description}
        for meta in skill_manager.list_available_skills()
    ]


@router.get("", response_model=list[AgentConfigResponse])
async def list_agents(runtime: ConsoleRuntimeDep) -> list[AgentConfigResponse]:
    """List all saved agent configurations."""
    records = await runtime.agent_registry.list_agents()
    default_agent_id = runtime.config.default_agent.id
    return [_record_to_response(r, default_agent_id=default_agent_id) for r in records]


@router.get("/capabilities", response_model=AgentCapabilitiesResponse)
async def get_agent_capabilities() -> AgentCapabilitiesResponse:
    providers = [
        AgentProviderCapabilityResponse(
            value=provider,
            label=_provider_label(provider),
            default_model_name=_provider_default_model_name(provider),
            requires_base_url=provider in COMPATIBLE_MODEL_PROVIDERS,
            requires_api_key_env_name=provider in COMPATIBLE_MODEL_PROVIDERS,
        )
        for provider in ALL_MODEL_PROVIDERS
    ]
    return AgentCapabilitiesResponse(providers=providers)


@router.get("/{agent_id}/sessions", response_model=PageResponse[SessionSummaryResponse])
async def list_agent_sessions(
    agent_id: str,
    runtime: ConsoleRuntimeDep,
) -> PageResponse[SessionSummaryResponse]:
    record = await runtime.agent_registry.get_agent(agent_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    page = await get_session_view_service(runtime).list_sessions(
        limit=50,
        offset=0,
        agent_id=agent_id,
    )
    return PageResponse(
        items=[session_summary_response_from_record(item) for item in page.items],
        limit=page.limit,
        offset=page.offset,
        has_more=page.has_more,
        total=page.total,
    )


@router.post("/{agent_id}/sessions", status_code=201)
async def create_agent_session(
    agent_id: str,
    runtime: ConsoleRuntimeDep,
):
    record = await runtime.agent_registry.get_agent(agent_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    session = await get_session_context_service(runtime).create_standalone_session(
        base_agent_id=agent_id,
        created_by="CONSOLE_CREATE",
    )
    return {
        "session_id": session.id,
        "source_session_id": session.source_session_id,
    }


@router.post("", response_model=AgentConfigResponse, status_code=201)
async def create_agent(
    body: AgentConfigPayload,
    runtime: ConsoleRuntimeDep,
) -> AgentConfigResponse:
    """Create a new agent configuration."""
    try:
        record = _body_to_record(body)
        created = await runtime.agent_registry.create_agent(record)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422, detail=jsonable_encoder(exc.errors())
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _record_to_response(
        created,
        default_agent_id=runtime.config.default_agent.id,
    )


@router.get("/{agent_id}", response_model=AgentConfigResponse)
async def get_agent(agent_id: str, runtime: ConsoleRuntimeDep) -> AgentConfigResponse:
    """Get a single agent configuration."""
    record = await runtime.agent_registry.get_agent(agent_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return _record_to_response(
        record,
        default_agent_id=runtime.config.default_agent.id,
    )


@router.put("/{agent_id}", response_model=AgentConfigResponse)
async def update_agent(
    agent_id: str,
    body: AgentConfigPayload,
    runtime: ConsoleRuntimeDep,
) -> AgentConfigResponse:
    """Replace an existing agent configuration."""
    try:
        record = await runtime.agent_registry.replace_agent(
            agent_id,
            _body_to_record(body),
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=422, detail=jsonable_encoder(exc.errors())
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return _record_to_response(
        record,
        default_agent_id=runtime.config.default_agent.id,
    )


@router.delete("/{agent_id}", status_code=204)
async def delete_agent(agent_id: str, runtime: ConsoleRuntimeDep) -> None:
    """Delete an agent configuration."""
    deleted = await runtime.agent_registry.delete_agent(agent_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Agent not found")
