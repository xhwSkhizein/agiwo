"""Sessions and Runs API router."""

from collections.abc import AsyncIterator
import json

from fastapi import APIRouter, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from server.dependencies import (
    ConsoleRuntimeDep,
    get_session_context_service,
    get_session_view_service,
)
from server.response_serialization import (
    run_response_from_sdk,
    session_detail_response_from_record,
    session_summary_response_from_record,
    step_response_from_sdk,
    stream_event_to_sse_message,
)
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
from server.channels.exceptions import BaseAgentNotFoundError
from server.services.runtime import SessionRuntimeService

router = APIRouter(prefix="/api", tags=["sessions"])
_STEPS_MAX_LIMIT = 5000


@router.get("/runs", response_model=PageResponse[RunResponse])
async def list_runs(
    runtime: ConsoleRuntimeDep,
    user_id: str | None = None,
    session_id: str | None = None,
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> PageResponse[RunResponse]:
    """List all runs with optional filtering."""
    storage = runtime.run_step_storage
    runs = await storage.list_runs(
        user_id=user_id,
        session_id=session_id,
        limit=limit + 1,
        offset=offset,
    )
    has_more = len(runs) > limit
    page = runs[:limit]
    return PageResponse(
        items=[run_response_from_sdk(r) for r in page],
        limit=limit,
        offset=offset,
        has_more=has_more,
        total=None,
    )


@router.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(run_id: str, runtime: ConsoleRuntimeDep) -> RunResponse:
    """Get a single run by ID."""
    storage = runtime.run_step_storage
    run = await storage.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run_response_from_sdk(run)


@router.get("/sessions", response_model=PageResponse[SessionSummaryResponse])
async def list_sessions(
    runtime: ConsoleRuntimeDep,
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> PageResponse[SessionSummaryResponse]:
    """List sessions from the session store with lightweight enrichment."""
    page = await get_session_view_service(runtime).list_sessions(
        limit=limit, offset=offset
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
) -> EventSourceResponse:
    if runtime.session_store is None:
        raise RuntimeError("Session store not available")
    if runtime.agent_runtime_cache is None:
        raise RuntimeError("Agent runtime cache not available")

    session = await runtime.session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        agent = await runtime.agent_runtime_cache.get_or_create_runtime_agent(session)
    except BaseAgentNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Agent not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if runtime.scheduler is None:
        raise RuntimeError("Scheduler not initialized")

    runtime_service = SessionRuntimeService(
        scheduler=runtime.scheduler,
        session_store=runtime.session_store,
        timeout=600,
    )
    dispatch = await runtime_service.execute(
        agent,
        session,
        body.message,
        stream_mode="until_settled",
    )

    async def event_generator() -> AsyncIterator[dict[str, str]]:
        if dispatch.stream is not None:
            async for item in dispatch.stream:
                yield stream_event_to_sse_message(item)
            return

        state = await runtime_service.get_state(session.id)
        if state is not None and state.result_summary:
            yield {
                "event": "scheduler_ack",
                "data": json.dumps(
                    {
                        "type": "scheduler_ack",
                        "session_id": session.id,
                        "state_id": session.id,
                        "result_summary": state.result_summary,
                    },
                    default=str,
                ),
            }
            return
        yield {
            "event": "scheduler_ack",
            "data": json.dumps(
                {
                    "type": "scheduler_ack",
                    "session_id": session.id,
                    "state_id": session.id,
                    "message": "消息已收到，正在继续处理。",
                },
                default=str,
            ),
        }

    return EventSourceResponse(event_generator())


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


@router.delete("/sessions/{session_id}")
async def delete_session_endpoint(
    session_id: str,
    runtime: ConsoleRuntimeDep,
) -> dict[str, bool]:
    """Delete a session from the session store."""
    if runtime.session_store is None:
        raise RuntimeError("Session store not available")
    deleted = await runtime.session_store.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
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
    storage = runtime.run_step_storage
    raw_steps = await storage.get_steps(
        session_id=session_id,
        start_seq=start_seq,
        end_seq=end_seq,
        run_id=run_id,
        agent_id=agent_id,
        limit=_STEPS_MAX_LIMIT + 1 if order == "desc" else limit + 1,
    )
    if order == "desc":
        raw_steps = list(reversed(raw_steps))
    has_more = len(raw_steps) > limit
    page = raw_steps[:limit]
    total = None
    if start_seq is None and end_seq is None and run_id is None and agent_id is None:
        total = await storage.get_step_count(session_id)
    return PageResponse(
        items=[step_response_from_sdk(s) for s in page],
        limit=limit,
        offset=0,
        has_more=has_more,
        total=total,
    )
