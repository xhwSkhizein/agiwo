"""Sessions and Runs API router."""

from collections.abc import AsyncIterator
import json
from dataclasses import dataclass

from fastapi import APIRouter, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from agiwo.scheduler.commands import RouteStreamMode

from server.dependencies import (
    ConsoleRuntimeDep,
    get_run_query_service,
    get_session_context_service,
    get_session_view_service,
)
from server.response_serialization import (
    run_response_from_sdk,
    scheduler_run_result_response_from_sdk,
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
from server.models.session import Session
from server.channels.exceptions import BaseAgentNotFoundError
from server.services.runtime import SessionRuntimeService

router = APIRouter(prefix="/api", tags=["sessions"])
_STEPS_MAX_LIMIT = 5000


@dataclass(frozen=True)
class _SessionExecutionContext:
    session: Session
    runtime_service: SessionRuntimeService


def _require_session_runtime(runtime: ConsoleRuntimeDep) -> None:
    if runtime.session_store is None:
        raise RuntimeError("Session store not available")
    if runtime.agent_runtime_cache is None:
        raise RuntimeError("Agent runtime cache not available")
    if runtime.scheduler is None:
        raise RuntimeError("Scheduler not initialized")


async def _build_session_execution_context(
    runtime: ConsoleRuntimeDep,
    session_id: str,
) -> _SessionExecutionContext:
    _require_session_runtime(runtime)
    assert runtime.session_store is not None
    assert runtime.scheduler is not None
    session = await runtime.session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    runtime_service = SessionRuntimeService(
        scheduler=runtime.scheduler,
        session_store=runtime.session_store,
        timeout=600,
    )
    return _SessionExecutionContext(session=session, runtime_service=runtime_service)


def _scheduler_ack_payload(
    *,
    session_id: str,
    last_run_result=None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": "scheduler_ack",
        "session_id": session_id,
        "state_id": session_id,
    }
    if last_run_result is not None:
        payload["last_run_result"] = scheduler_run_result_response_from_sdk(
            last_run_result
        ).model_dump()
    else:
        payload["message"] = "消息已收到，正在继续处理。"
    return payload


async def _session_input_event_stream(
    *,
    dispatch,
    runtime_service: SessionRuntimeService,
    session_id: str,
) -> AsyncIterator[dict[str, str]]:
    if dispatch.stream is not None:
        async for item in dispatch.stream:
            yield stream_event_to_sse_message(item)
        return

    state = await runtime_service.get_state(session_id)
    yield {
        "event": "scheduler_ack",
        "data": json.dumps(
            _scheduler_ack_payload(
                session_id=session_id,
                last_run_result=state.last_run_result if state is not None else None,
            ),
            default=str,
        ),
    }


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
    context = await _build_session_execution_context(runtime, session_id)
    assert runtime.agent_runtime_cache is not None

    try:
        agent = await runtime.agent_runtime_cache.get_or_create_runtime_agent(
            context.session
        )
    except BaseAgentNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Agent not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    dispatch = await context.runtime_service.execute(
        agent,
        context.session,
        body.message,
        stream_mode=RouteStreamMode.UNTIL_SETTLED,
    )
    return EventSourceResponse(
        _session_input_event_stream(
            dispatch=dispatch,
            runtime_service=context.runtime_service,
            session_id=context.session.id,
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
