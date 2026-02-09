"""
Chat API router — real-time Agent conversation via SSE.
"""

import asyncio
import json
from dataclasses import asdict
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from agiwo.agent.agent import Agent
from agiwo.agent.options import AgentOptions, RunStepStorageConfig, TraceStorageConfig
from agiwo.agent.schema import StreamEvent
from agiwo.llm.openai import OpenAIModel
from agiwo.llm.anthropic import AnthropicModel
from agiwo.llm.deepseek import DeepseekModel

from server.config import ConsoleConfig
from server.dependencies import get_agent_registry, get_console_config, get_storage_manager
from server.schemas import ChatRequest
from server.services.agent_registry import AgentConfigRecord
from server.tools import create_tools

router = APIRouter(prefix="/api/chat", tags=["chat"])

MODEL_PROVIDER_MAP: dict[str, type] = {
    "openai": OpenAIModel,
    "deepseek": DeepseekModel,
    "anthropic": AnthropicModel,
}


def _build_model(config: AgentConfigRecord) -> Any:
    """Build a Model instance from agent config."""
    provider_cls = MODEL_PROVIDER_MAP.get(config.model_provider)
    if provider_cls is None:
        raise ValueError(f"Unknown model provider: {config.model_provider}")

    params: dict[str, Any] = {
        "id": config.model_name,
        "name": config.model_name,
    }
    params.update(config.model_params)

    return provider_cls(**params)


def _build_agent_options(config: AgentConfigRecord, console_config: ConsoleConfig) -> AgentOptions:
    """Build AgentOptions with storage config matching the console storage backend."""
    opts = config.options or {}

    if console_config.storage_type == "sqlite":
        run_step_cfg = RunStepStorageConfig(
            storage_type="sqlite",
            config={"db_path": console_config.sqlite_db_path},
        )
        trace_cfg = TraceStorageConfig(
            storage_type="sqlite",
            config={
                "db_path": console_config.sqlite_db_path,
                "collection_name": console_config.sqlite_trace_collection,
            },
        )
    else:
        run_step_cfg = RunStepStorageConfig(
            storage_type="mongodb",
            config={
                "uri": console_config.mongodb_uri,
                "db_name": console_config.mongodb_db_name,
            },
        )
        trace_cfg = TraceStorageConfig(
            storage_type="mongodb",
            config={
                "mongo_uri": console_config.mongodb_uri,
                "db_name": console_config.mongodb_db_name,
                "collection_name": console_config.mongodb_trace_collection,
            },
        )

    return AgentOptions(
        max_steps=opts.get("max_steps", 10),
        run_timeout=opts.get("run_timeout", 600),
        max_output_tokens=opts.get("max_output_tokens", 8196),
        run_step_storage=run_step_cfg,
        trace_storage=trace_cfg,
    )


def _serialize_event(event: StreamEvent) -> str:
    """Serialize a StreamEvent to JSON string for SSE data field."""
    data: dict[str, Any] = {
        "type": event.type.value,
        "run_id": event.run_id,
    }
    if event.delta is not None:
        data["delta"] = asdict(event.delta)
    if event.step is not None:
        step_dict = asdict(event.step)
        step_dict["role"] = event.step.role.value if hasattr(event.step.role, "value") else str(event.step.role)
        if event.step.metrics:
            step_dict["metrics"] = asdict(event.step.metrics)
        data["step"] = step_dict
    if event.data is not None:
        data["data"] = event.data
    if event.agent_id is not None:
        data["agent_id"] = event.agent_id
    if event.span_id is not None:
        data["span_id"] = event.span_id
    return json.dumps(data, default=str)


def _build_agent(config: AgentConfigRecord, console_config: ConsoleConfig) -> Agent:
    """Build an Agent instance from persisted config."""
    model = _build_model(config)
    options = _build_agent_options(config, console_config)
    tools = create_tools(config.tools or [])

    return Agent(
        id=config.id,
        description=config.description,
        model=model,
        tools=tools or None,
        system_prompt=config.system_prompt,
        options=options,
    )


@router.post("/{agent_id}")
async def chat(agent_id: str, body: ChatRequest) -> EventSourceResponse:
    """
    Send a message to an agent and stream the response via SSE.

    The response streams StreamEvent objects in real-time, including:
    - step_delta: Token-by-token content
    - step_completed: Completed assistant/tool steps
    - run_started / run_completed / run_failed: Lifecycle events
    """
    registry = get_agent_registry()
    agent_config = await registry.get_agent(agent_id)
    if agent_config is None:
        raise HTTPException(status_code=404, detail="Agent not found")

    console_config = get_console_config()

    agent = _build_agent(agent_config, console_config)
    session_id = body.session_id or str(uuid4())

    async def event_generator():
        try:
            async for event in agent.run_stream(
                body.message,
                session_id=session_id,
            ):
                yield {"event": event.type.value, "data": _serialize_event(event)}
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": str(e)})}
        finally:
            await agent.close()

    return EventSourceResponse(event_generator())


@router.get("/{agent_id}/sessions")
async def list_agent_sessions(agent_id: str):
    """Get conversation sessions for a specific agent."""
    storage = get_storage_manager().run_step_storage
    runs = await storage.list_runs(limit=500)

    session_map: dict[str, dict[str, Any]] = {}
    for run in runs:
        if run.agent_id != agent_id:
            continue
        sid = run.session_id
        if sid not in session_map:
            session_map[sid] = {
                "session_id": sid,
                "runs": [],
                "last_input": None,
                "last_response": None,
                "updated_at": None,
            }
        session_map[sid]["runs"].append(run)
        if run.user_input:
            session_map[sid]["last_input"] = (
                run.user_input if isinstance(run.user_input, str) else str(run.user_input)
            )
        if run.response_content:
            session_map[sid]["last_response"] = run.response_content
        if run.updated_at:
            if session_map[sid]["updated_at"] is None or run.updated_at > session_map[sid]["updated_at"]:
                session_map[sid]["updated_at"] = run.updated_at

    sessions = sorted(session_map.values(), key=lambda s: s["updated_at"] or "", reverse=True)
    return [
        {
            "session_id": s["session_id"],
            "run_count": len(s["runs"]),
            "last_input": s["last_input"][:200] if s["last_input"] else None,
            "last_response": s["last_response"][:200] if s["last_response"] else None,
            "updated_at": s["updated_at"].isoformat() if hasattr(s["updated_at"], "isoformat") else str(s["updated_at"]) if s["updated_at"] else None,
        }
        for s in sessions
    ]
