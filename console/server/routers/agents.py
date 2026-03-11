"""Agents CRUD API router."""

from fastapi import APIRouter, HTTPException, Query as QueryParam
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from agiwo.utils.serialization import serialize_optional_datetime
from server.dependencies import ConsoleRuntimeDep
from server.schemas import (
    AgentConfigCreate,
    AgentConfigReplace,
    AgentConfigResponse,
    AgentOptionsPayload,
    ModelParamsPayload,
)
from server.services.agent_registry import AgentConfigRecord
from server.tools import console_tool_catalog

router = APIRouter(prefix="/api/agents", tags=["agents"])


@router.get("/tools/available")
async def list_available_tools(
    runtime: ConsoleRuntimeDep,
    exclude: str | None = QueryParam(default=None, description="Agent ID to exclude from agent tools"),
) -> list[dict[str, str]]:
    """Return all available tools (builtin + agent-as-tool) that can be assigned to agents."""
    return await console_tool_catalog.list_available_tools(
        runtime.agent_registry,
        exclude_agent_id=exclude,
    )


def _body_to_record(body: AgentConfigCreate | AgentConfigReplace) -> AgentConfigRecord:
    return AgentConfigRecord(
        name=body.name,
        description=body.description,
        model_provider=body.model_provider,
        model_name=body.model_name,
        system_prompt=body.system_prompt,
        tools=body.tools,
        options=body.options.model_dump(exclude_none=True),
        model_params=body.model_params.model_dump(exclude_none=True),
    )


def _record_to_response(record: AgentConfigRecord) -> AgentConfigResponse:
    return AgentConfigResponse(
        id=record.id,
        name=record.name,
        description=record.description,
        model_provider=record.model_provider,
        model_name=record.model_name,
        system_prompt=record.system_prompt,
        tools=record.tools,
        options=AgentOptionsPayload.model_validate(record.options or {}),
        model_params=ModelParamsPayload.model_validate(record.model_params or {}),
        created_at=serialize_optional_datetime(record.created_at) or "",
        updated_at=serialize_optional_datetime(record.updated_at) or "",
    )


@router.get("", response_model=list[AgentConfigResponse])
async def list_agents(runtime: ConsoleRuntimeDep) -> list[AgentConfigResponse]:
    """List all saved agent configurations."""
    records = await runtime.agent_registry.list_agents()
    return [_record_to_response(r) for r in records]


@router.post("", response_model=AgentConfigResponse, status_code=201)
async def create_agent(
    body: AgentConfigCreate,
    runtime: ConsoleRuntimeDep,
) -> AgentConfigResponse:
    """Create a new agent configuration."""
    record = _body_to_record(body)
    try:
        created = await runtime.agent_registry.create_agent(record)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=jsonable_encoder(exc.errors())) from exc
    return _record_to_response(created)


@router.get("/{agent_id}", response_model=AgentConfigResponse)
async def get_agent(agent_id: str, runtime: ConsoleRuntimeDep) -> AgentConfigResponse:
    """Get a single agent configuration."""
    record = await runtime.agent_registry.get_agent(agent_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return _record_to_response(record)


@router.put("/{agent_id}", response_model=AgentConfigResponse)
async def update_agent(
    agent_id: str,
    body: AgentConfigReplace,
    runtime: ConsoleRuntimeDep,
) -> AgentConfigResponse:
    """Replace an existing agent configuration."""
    try:
        record = await runtime.agent_registry.replace_agent(agent_id, _body_to_record(body))
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=jsonable_encoder(exc.errors())) from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return _record_to_response(record)


@router.delete("/{agent_id}", status_code=204)
async def delete_agent(agent_id: str, runtime: ConsoleRuntimeDep) -> None:
    """Delete an agent configuration."""
    deleted = await runtime.agent_registry.delete_agent(agent_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Agent not found")
