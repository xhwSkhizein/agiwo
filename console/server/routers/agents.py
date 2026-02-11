"""
Agents CRUD API router.
"""

from fastapi import APIRouter, HTTPException, Query as QueryParam

from server.dependencies import get_agent_registry
from server.schemas import AgentConfigCreate, AgentConfigUpdate, AgentConfigResponse
from server.services.agent_registry import AgentConfigRecord
from server.tools import get_available_builtin_tools

router = APIRouter(prefix="/api/agents", tags=["agents"])


@router.get("/tools/available")
async def list_available_tools(
    exclude: str | None = QueryParam(default=None, description="Agent ID to exclude from agent tools"),
) -> list[dict[str, str]]:
    """Return all available tools (builtin + agent-as-tool) that can be assigned to agents."""
    tools = get_available_builtin_tools()

    registry = get_agent_registry()
    agents = await registry.list_agents()
    for agent in agents:
        if exclude and agent.id == exclude:
            continue
        tools.append({
            "name": f"agent:{agent.id}",
            "description": agent.description or f"Delegate tasks to {agent.name}",
            "type": "agent",
            "agent_name": agent.name,
        })

    return tools


def _record_to_response(record: AgentConfigRecord) -> AgentConfigResponse:
    return AgentConfigResponse(
        id=record.id,
        name=record.name,
        description=record.description,
        model_provider=record.model_provider,
        model_name=record.model_name,
        system_prompt=record.system_prompt,
        tools=record.tools,
        options=record.options,
        model_params=record.model_params,
        created_at=record.created_at.isoformat() if hasattr(record.created_at, "isoformat") else str(record.created_at),
        updated_at=record.updated_at.isoformat() if hasattr(record.updated_at, "isoformat") else str(record.updated_at),
    )


@router.get("", response_model=list[AgentConfigResponse])
async def list_agents() -> list[AgentConfigResponse]:
    """List all saved agent configurations."""
    registry = get_agent_registry()
    records = await registry.list_agents()
    return [_record_to_response(r) for r in records]


@router.post("", response_model=AgentConfigResponse, status_code=201)
async def create_agent(body: AgentConfigCreate) -> AgentConfigResponse:
    """Create a new agent configuration."""
    registry = get_agent_registry()
    record = AgentConfigRecord(
        name=body.name,
        description=body.description,
        model_provider=body.model_provider,
        model_name=body.model_name,
        system_prompt=body.system_prompt,
        tools=body.tools,
        options=body.options,
        model_params=body.model_params,
    )
    created = await registry.create_agent(record)
    return _record_to_response(created)


@router.get("/{agent_id}", response_model=AgentConfigResponse)
async def get_agent(agent_id: str) -> AgentConfigResponse:
    """Get a single agent configuration."""
    registry = get_agent_registry()
    record = await registry.get_agent(agent_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return _record_to_response(record)


@router.put("/{agent_id}", response_model=AgentConfigResponse)
async def update_agent(agent_id: str, body: AgentConfigUpdate) -> AgentConfigResponse:
    """Update an existing agent configuration."""
    registry = get_agent_registry()
    updates = body.model_dump(exclude_none=True)
    record = await registry.update_agent(agent_id, updates)
    if record is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return _record_to_response(record)


@router.delete("/{agent_id}", status_code=204)
async def delete_agent(agent_id: str) -> None:
    """Delete an agent configuration."""
    registry = get_agent_registry()
    deleted = await registry.delete_agent(agent_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Agent not found")
