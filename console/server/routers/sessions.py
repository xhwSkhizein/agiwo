"""Sessions and Runs API router."""

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timezone
import json
import uuid

from fastapi import APIRouter, Header, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from agiwo.agent.models.input import UserMessage

from server.dependencies import (
    ConsoleRuntimeDep,
    get_run_query_service,
    get_session_context_service,
    get_session_gateway,
    get_session_view_service,
)
from server.response_serialization import (
    run_response_from_sdk,
    session_detail_response_from_record,
    session_summary_response_from_record,
    step_response_from_sdk,
)
from server.channels.exceptions import BaseAgentNotFoundError
from server.services.runtime.session_runtime_service import SessionRuntimeService
from server.models.view import (
    CancelRequest,
    ChatRequest,
    ForkSessionRequest,
    PageResponse,
    RunResponse,
    SessionDetailResponse,
    SessionSummaryResponse,
    StepResponse,
)

router = APIRouter(prefix="/api", tags=["sessions"])
_STEPS_MAX_LIMIT = 5000
_ARCHIVE_DRAIN_TIMEOUT_SECONDS = 30.0


async def _session_input_event_stream(
    *,
    session_id: str,
    body: ChatRequest,
    idempotency_key: str,
    gateway,
) -> AsyncIterator[dict[str, str]]:
    message = UserMessage.from_value(body.message)
    try:
        result = await gateway.handle_user_message(
            session_id,
            message,
            idempotency_key=idempotency_key,
        )
    except (ValueError, BaseAgentNotFoundError) as exc:
        yield {
            "event": "session_error",
            "data": json.dumps({"message": str(exc)}, default=str),
        }
        return

    payload = {
        "kind": result.kind,
        "session_id": result.session_id,
        "run_id": result.run_id,
        "status": result.status,
        "response": result.response,
    }
    yield {"event": "session_turn", "data": json.dumps(payload, default=str)}


@router.get("/runs", response_model=PageResponse[RunResponse])
async def list_runs(
    runtime: ConsoleRuntimeDep,
    user_id: str | None = None,
    session_id: str | None = None,
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> PageResponse[RunResponse]:
    """List all runs with optional filtering."""
    page = await get_run_query_service(runtime).list_runs(
        user_id=user_id,
        session_id=session_id,
        limit=limit,
        offset=offset,
    )
    return PageResponse(
        items=[run_response_from_sdk(r) for r in page.items],
        limit=page.limit,
        offset=page.offset,
        has_more=page.has_more,
        total=page.total,
    )


@router.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(run_id: str, runtime: ConsoleRuntimeDep) -> RunResponse:
    """Get a single run by ID."""
    run = await get_run_query_service(runtime).get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run_response_from_sdk(run)


@router.get("/sessions", response_model=PageResponse[SessionSummaryResponse])
async def list_sessions(
    runtime: ConsoleRuntimeDep,
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    include_archived: bool = Query(default=False),
) -> PageResponse[SessionSummaryResponse]:
    """List sessions from the session store with lightweight enrichment."""
    page = await get_session_view_service(runtime).list_sessions(
        limit=limit, offset=offset, include_archived=include_archived
    )
    return PageResponse(
        items=[session_summary_response_from_record(item) for item in page.items],
        limit=page.limit,
        offset=page.offset,
        has_more=page.has_more,
        total=page.total,
    )


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session_detail(
    session_id: str,
    runtime: ConsoleRuntimeDep,
) -> SessionDetailResponse:
    detail = await get_session_view_service(runtime).get_session_detail(session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session_detail_response_from_record(detail)


@router.post("/sessions/{session_id}/input")
async def send_session_input(
    session_id: str,
    body: ChatRequest,
    runtime: ConsoleRuntimeDep,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> EventSourceResponse:
    if runtime.session_store is None:
        raise RuntimeError("Session store not available")
    session = await runtime.session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    gateway = get_session_gateway(runtime)
    key = idempotency_key or uuid.uuid4().hex

    return EventSourceResponse(
        _session_input_event_stream(
            gateway=gateway,
            session_id=session_id,
            body=body,
            idempotency_key=key,
        )
    )


@router.get("/sessions/{session_id}/summary", response_model=SessionSummaryResponse)
async def get_session_summary(
    session_id: str,
    runtime: ConsoleRuntimeDep,
) -> SessionSummaryResponse:
    """Get full aggregated metrics for one session."""
    detail = await get_session_view_service(runtime).get_session_detail(session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session_summary_response_from_record(detail.summary)


@router.post("/sessions/{session_id}/cancel")
async def cancel_session(
    session_id: str,
    body: CancelRequest,
    runtime: ConsoleRuntimeDep,
):
    if runtime.scheduler is None:
        raise RuntimeError("Scheduler not initialized")
    success = await runtime.scheduler.cancel(session_id, body.reason)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"No active orchestration found for session_id={session_id}",
        )
    return {"ok": True, "session_id": session_id, "state_id": session_id}


@router.post("/sessions/{session_id}/fork")
async def fork_session(
    session_id: str,
    body: ForkSessionRequest,
    runtime: ConsoleRuntimeDep,
):
    result = await get_session_context_service(runtime).fork_session_by_id(
        session_id=session_id,
        context_summary=body.context_summary,
        created_by="CONSOLE_FORK",
        update_chat_context=False,
    )
    return {
        "session_id": result.session.id,
        "source_session_id": result.session.source_session_id,
    }


@router.post("/sessions/{session_id}/archive")
async def archive_session_endpoint(
    session_id: str,
    runtime: ConsoleRuntimeDep,
) -> dict[str, object]:
    """Archive a session: drain/cancel any active root run, then hide from listing."""
    if runtime.session_store is None:
        raise RuntimeError("Session store not available")
    if runtime.scheduler is None:
        raise RuntimeError("Scheduler not initialized")
    session = await runtime.session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    session_runtime = SessionRuntimeService(
        scheduler=runtime.scheduler,
        session_store=runtime.session_store,
    )
    await session_runtime.cancel_if_active(session, reason="user_archive")

    deadline = asyncio.get_running_loop().time() + _ARCHIVE_DRAIN_TIMEOUT_SECONDS
    while asyncio.get_running_loop().time() < deadline:
        state = await runtime.scheduler.get_state(session_id)
        if state is None or not state.is_active():
            break
        await asyncio.sleep(0.2)
    else:
        raise HTTPException(
            status_code=409,
            detail="archive drain incomplete; root run still active",
        )

    session.archived_at = datetime.now(timezone.utc)
    await runtime.session_store.upsert_session(session)
    return {"ok": True, "session_id": session_id, "archived_at": session.archived_at}


@router.post("/sessions/{session_id}/restore")
async def restore_session_endpoint(
    session_id: str,
    runtime: ConsoleRuntimeDep,
) -> dict[str, bool]:
    """Clear a session's archived_at flag."""
    if runtime.session_store is None:
        raise RuntimeError("Session store not available")
    session = await runtime.session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    session.archived_at = None
    await runtime.session_store.upsert_session(session)
    return {"ok": True}


@router.get("/sessions/{session_id}/steps", response_model=PageResponse[StepResponse])
async def get_session_steps(
    session_id: str,
    runtime: ConsoleRuntimeDep,
    start_seq: int | None = Query(default=None, ge=1),
    end_seq: int | None = Query(default=None, ge=1),
    run_id: str | None = None,
    agent_id: str | None = None,
    limit: int = Query(default=1000, ge=1, le=_STEPS_MAX_LIMIT),
    order: str = Query(default="asc", pattern="^(asc|desc)$"),
) -> PageResponse[StepResponse]:
    """Get all steps for a session."""
    page = await get_run_query_service(runtime).list_session_steps(
        session_id,
        start_seq=start_seq,
        end_seq=end_seq,
        run_id=run_id,
        agent_id=agent_id,
        limit=limit,
        order=order,
    )
    return PageResponse(
        items=[step_response_from_sdk(s) for s in page.items],
        limit=page.limit,
        offset=page.offset,
        has_more=page.has_more,
        total=page.total,
    )
