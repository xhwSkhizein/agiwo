"""
Chat API router — real-time Agent conversation via SSE.
"""

import json
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from server.dependencies import get_agent_registry, get_console_config, get_storage_manager
from server.schemas import ChatRequest
from server.services.agent_builder import build_agent, serialize_event

router = APIRouter(prefix="/api/chat", tags=["chat"])


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
    agent = await build_agent(agent_config, console_config, registry)
    session_id = body.session_id or str(uuid4())

    async def event_generator():
        try:
            async for event in agent.run_stream(
                body.message,
                session_id=session_id,
            ):
                yield {"event": event.type.value, "data": serialize_event(event)}
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
