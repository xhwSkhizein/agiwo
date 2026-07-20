"""Objective gateway API router (P5-01, P5-02)."""

from collections.abc import AsyncIterator
from typing import NoReturn

from fastapi import APIRouter, Header, HTTPException, Query, Request
from sse_starlette.sse import EventSourceResponse

from agiwo.objective import (
    AdjustBudgetRequest,
    CommandUnavailable,
    CreateObjectiveRequest,
    ExternalizeUserInputRequest,
    IdempotencyConflict,
    InvariantViolation,
    ObjectiveError,
    PauseObjectiveRequest,
    ResumeObjectiveRequest,
    SubmitUserInputRequest,
    ValidationError,
)
from agiwo.objective.errors import BudgetBoundaryHit
from agiwo.objective.projection import project_timeline_page

from server.dependencies import (
    ConsoleRuntimeDep,
    get_objective_service,
    get_objective_store,
)
from server.models.objective import (
    AdjustBudgetBody,
    CommandResultResponse,
    CreateObjectiveBody,
    ExternalizeBody,
    ObjectiveEventResponse,
    ObjectiveViewResponse,
    PauseBody,
    ResumeBody,
    SubmitInputBody,
)
from server.services.objective_event_stream import stream_objective_events
from server.services.objective_gateway import DEFAULT_BUDGET_LIMITS
from server.services.objective_serialization import (
    budget_limits_from_body,
    command_result_response,
    objective_event_response,
    objective_view_response,
    user_message_from_parts,
)

router = APIRouter(prefix="/api", tags=["objectives"])

_DEFAULT_TIMELINE_LIMIT = 100
_MAX_TIMELINE_LIMIT = 1000


def _require_idempotency_key(idempotency_key: str | None) -> str:
    if not idempotency_key:
        raise HTTPException(
            status_code=400,
            detail="Idempotency-Key header is required",
        )
    return idempotency_key


def _raise_for_objective_error(exc: ObjectiveError) -> NoReturn:
    detail = {"code": exc.code, "message": exc.message, **exc.details}
    if isinstance(exc, ValidationError):
        status_code = 404 if "not found" in exc.message else 400
        raise HTTPException(status_code=status_code, detail=detail) from exc
    if isinstance(exc, (InvariantViolation, IdempotencyConflict, BudgetBoundaryHit)):
        raise HTTPException(status_code=409, detail=detail) from exc
    if isinstance(exc, CommandUnavailable):
        raise HTTPException(status_code=503, detail=detail) from exc
    raise HTTPException(status_code=500, detail=detail) from exc


@router.post("/objectives", response_model=CommandResultResponse, status_code=201)
async def create_objective(
    body: CreateObjectiveBody,
    runtime: ConsoleRuntimeDep,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> CommandResultResponse:
    key = _require_idempotency_key(idempotency_key)
    service = get_objective_service(runtime)
    message = user_message_from_parts(body.message, body.parts)
    budget = budget_limits_from_body(body.budget) or DEFAULT_BUDGET_LIMITS
    try:
        result = await service.create_objective(
            CreateObjectiveRequest(
                session_id=body.session_id,
                user_message=message,
                budget=budget,
                idempotency_key=key,
                related_outcome_id=body.related_outcome_id,
            )
        )
    except ObjectiveError as exc:
        _raise_for_objective_error(exc)
    return command_result_response(result)


@router.get("/objectives/{objective_id}", response_model=ObjectiveViewResponse)
async def get_objective(
    objective_id: str,
    runtime: ConsoleRuntimeDep,
) -> ObjectiveViewResponse:
    service = get_objective_service(runtime)
    view = await service.get_view(objective_id)
    if view is None:
        raise HTTPException(status_code=404, detail="Objective not found")
    return objective_view_response(view)


@router.get("/objectives/{objective_id}/metrics")
async def get_objective_metrics(
    objective_id: str,
    runtime: ConsoleRuntimeDep,
) -> dict[str, object]:
    """Objective-level aggregates projected from committed ObjectiveLog facts."""
    service = get_objective_service(runtime)
    metrics = await service.get_metrics(objective_id)
    if metrics is None:
        raise HTTPException(status_code=404, detail="Objective not found")
    return metrics.to_dict()


@router.get(
    "/sessions/{session_id}/objectives",
    response_model=list[ObjectiveViewResponse],
)
async def list_session_objectives(
    session_id: str,
    runtime: ConsoleRuntimeDep,
) -> list[ObjectiveViewResponse]:
    service = get_objective_service(runtime)
    views = await service.list_by_session(session_id)
    return [objective_view_response(view) for view in views]


@router.get(
    "/objectives/{objective_id}/timeline",
    response_model=list[ObjectiveEventResponse],
)
async def get_objective_timeline(
    objective_id: str,
    runtime: ConsoleRuntimeDep,
    after_sequence: int = Query(default=0, ge=0),
    limit: int = Query(default=_DEFAULT_TIMELINE_LIMIT, ge=1, le=_MAX_TIMELINE_LIMIT),
) -> list[ObjectiveEventResponse]:
    store = get_objective_store(runtime)
    facts = await store.list_facts(objective_id=objective_id)
    if not facts:
        raise HTTPException(status_code=404, detail="Objective not found")
    nodes = project_timeline_page(facts, after_sequence=after_sequence, limit=limit)
    return [objective_event_response(node) for node in nodes]


@router.post(
    "/objectives/{objective_id}/inputs",
    response_model=CommandResultResponse,
)
async def submit_objective_input(
    objective_id: str,
    body: SubmitInputBody,
    runtime: ConsoleRuntimeDep,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> CommandResultResponse:
    key = _require_idempotency_key(idempotency_key)
    service = get_objective_service(runtime)
    message = user_message_from_parts(body.message, body.parts)
    try:
        result = await service.submit_user_input(
            SubmitUserInputRequest(
                objective_id=objective_id,
                user_message=message,
                idempotency_key=key,
                in_reply_to_message_id=body.in_reply_to_message_id,
                related_outcome_id=body.related_outcome_id,
            )
        )
    except ObjectiveError as exc:
        _raise_for_objective_error(exc)
    return command_result_response(result)


@router.post(
    "/objectives/{objective_id}/inputs/{input_id}/externalize",
    response_model=CommandResultResponse,
)
async def externalize_objective_input(
    objective_id: str,
    input_id: str,
    body: ExternalizeBody,
    runtime: ConsoleRuntimeDep,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> CommandResultResponse:
    key = _require_idempotency_key(idempotency_key)
    service = get_objective_service(runtime)
    try:
        result = await service.externalize_user_input(
            ExternalizeUserInputRequest(
                objective_id=objective_id,
                input_id=input_id,
                summary=body.summary,
                idempotency_key=key,
                content_hash=body.content_hash,
            )
        )
    except ObjectiveError as exc:
        _raise_for_objective_error(exc)
    return command_result_response(result)


@router.post("/objectives/{objective_id}/pause", response_model=CommandResultResponse)
async def pause_objective(
    objective_id: str,
    body: PauseBody,
    runtime: ConsoleRuntimeDep,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> CommandResultResponse:
    key = _require_idempotency_key(idempotency_key)
    service = get_objective_service(runtime)
    try:
        result = await service.pause(
            PauseObjectiveRequest(
                objective_id=objective_id,
                idempotency_key=key,
                reason=body.reason,
            )
        )
    except ObjectiveError as exc:
        _raise_for_objective_error(exc)
    return command_result_response(result)


@router.post("/objectives/{objective_id}/resume", response_model=CommandResultResponse)
async def resume_objective(
    objective_id: str,
    body: ResumeBody,
    runtime: ConsoleRuntimeDep,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> CommandResultResponse:
    key = _require_idempotency_key(idempotency_key)
    service = get_objective_service(runtime)
    try:
        result = await service.resume(
            ResumeObjectiveRequest(
                objective_id=objective_id,
                idempotency_key=key,
                reason=body.reason,
            )
        )
    except ObjectiveError as exc:
        _raise_for_objective_error(exc)
    return command_result_response(result)


@router.post("/objectives/{objective_id}/budget", response_model=CommandResultResponse)
async def adjust_objective_budget(
    objective_id: str,
    body: AdjustBudgetBody,
    runtime: ConsoleRuntimeDep,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> CommandResultResponse:
    key = _require_idempotency_key(idempotency_key)
    service = get_objective_service(runtime)
    try:
        result = await service.adjust_budget(
            AdjustBudgetRequest(
                objective_id=objective_id,
                idempotency_key=key,
                handoffs=body.handoffs,
                verification_attempts=body.verification_attempts,
                llm_cost_usd=body.llm_cost_usd,
                active_seconds=body.active_seconds,
            )
        )
    except ObjectiveError as exc:
        _raise_for_objective_error(exc)
    return command_result_response(result)


def _resolve_cursor(request: Request, after_sequence: int) -> int:
    last_event_id = request.headers.get("last-event-id")
    if last_event_id:
        try:
            return int(last_event_id)
        except ValueError:
            pass
    return after_sequence


@router.get("/objectives/{objective_id}/events")
async def stream_objective_events_endpoint(
    objective_id: str,
    request: Request,
    runtime: ConsoleRuntimeDep,
    after_sequence: int = Query(default=0, ge=0),
) -> EventSourceResponse:
    service = get_objective_service(runtime)
    view = await service.get_view(objective_id)
    if view is None:
        raise HTTPException(status_code=404, detail="Objective not found")

    store = get_objective_store(runtime)
    cursor = _resolve_cursor(request, after_sequence)

    async def _events() -> AsyncIterator[dict[str, str]]:
        async for message in stream_objective_events(store, objective_id, cursor):
            yield message

    return EventSourceResponse(_events())


__all__ = ["router"]
