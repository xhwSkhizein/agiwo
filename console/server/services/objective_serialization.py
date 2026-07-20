"""Serialize Objective SDK types into Console API view models and SSE payloads.

Timeline/event payloads are intentionally limited to ``summary`` + ``refs``:
sensitive or bulky debug payloads (raw fact payloads, full user message
content, etc.) never leave the ObjectiveLog through these helpers.
"""

import json
from typing import Any

from agiwo.agent.models.input import UserMessage
from agiwo.objective import BudgetLimits, CommandResult
from agiwo.objective.log import ObjectiveFactKind
from agiwo.objective.models import ObjectiveBudget
from agiwo.objective.projection import ObjectiveView, RootRunView, TimelineNode

from server.models.objective import (
    ArtifactSummaryResponse,
    BudgetDimensionResponse,
    BudgetLimitsBody,
    CommandResultResponse,
    ObjectiveBudgetResponse,
    ObjectiveEventResponse,
    ObjectiveViewResponse,
    RootRunViewResponse,
    TimelineNodeResponse,
)


def user_message_from_parts(
    message: str | None,
    parts: list[dict[str, Any]] | None,
) -> UserMessage:
    """Build a user-provided ``UserMessage`` from a Console request body."""
    if parts:
        restored = UserMessage.from_storage_value(
            {
                "__type": "user_message",
                "content": parts,
                "is_user_provided": True,
            }
        )
        if not isinstance(restored, UserMessage):
            raise TypeError("expected UserMessage from request parts")
        message_obj = restored
    else:
        message_obj = UserMessage.from_value(message or "")
    UserMessage.require_user_provided(message_obj)
    return message_obj


def budget_limits_from_body(body: BudgetLimitsBody | None) -> BudgetLimits | None:
    if body is None:
        return None
    return body.to_budget_limits()


def _budget_dimension_response(dimension) -> BudgetDimensionResponse:
    return BudgetDimensionResponse(
        limit=dimension.limit,
        used=dimension.used,
        remaining=dimension.remaining,
    )


def objective_budget_response(budget: ObjectiveBudget) -> ObjectiveBudgetResponse:
    return ObjectiveBudgetResponse(
        handoffs=_budget_dimension_response(budget.handoffs),
        verification_attempts=_budget_dimension_response(budget.verification_attempts),
        llm_cost_usd=_budget_dimension_response(budget.llm_cost_usd),
        active_seconds=_budget_dimension_response(budget.active_seconds),
    )


def timeline_node_response(node: TimelineNode) -> TimelineNodeResponse:
    return TimelineNodeResponse(
        sequence=node.sequence,
        fact_id=node.fact_id,
        kind=node.kind.value
        if isinstance(node.kind, ObjectiveFactKind)
        else str(node.kind),
        occurred_at=node.occurred_at.isoformat(),
        summary=node.summary,
        refs=_safe_refs(node.refs),
    )


def objective_event_response(node: TimelineNode) -> ObjectiveEventResponse:
    return ObjectiveEventResponse(
        sequence=node.sequence,
        fact_id=node.fact_id,
        kind=node.kind.value
        if isinstance(node.kind, ObjectiveFactKind)
        else str(node.kind),
        occurred_at=node.occurred_at.isoformat(),
        summary=node.summary,
        refs=_safe_refs(node.refs),
    )


def _safe_refs(refs: dict[str, Any]) -> dict[str, Any]:
    """Keep only small, JSON-safe reference values (ids, reasons, counters)."""
    safe: dict[str, Any] = {}
    for key, value in refs.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            safe[key] = value
    return safe


def root_run_view_response(root_run: RootRunView) -> RootRunViewResponse:
    status_value = root_run.status.value if root_run.status is not None else "requested"
    return RootRunViewResponse(
        run_id=root_run.run_id,
        role=root_run.role.value,
        status=status_value,
        run_ids=[root_run.run_id],
        outcome_report=root_run.outcome.report if root_run.outcome else None,
        decision_target=root_run.decision.target.value if root_run.decision else None,
        created_at=root_run.created_at.isoformat() if root_run.created_at else None,
        updated_at=root_run.updated_at.isoformat() if root_run.updated_at else None,
    )


def objective_view_response(view: ObjectiveView) -> ObjectiveViewResponse:
    return ObjectiveViewResponse(
        objective_id=view.objective_id,
        session_id=view.session_id,
        status=view.status.value,
        is_terminal=view.is_terminal,
        budget=objective_budget_response(view.budget),
        timeline=[timeline_node_response(node) for node in view.timeline],
        root_runs=[root_run_view_response(run) for run in view.root_runs],
        delivery_report=view.delivery_report,
        delivery_outcome_id=view.delivery_outcome_id,
        artifacts=[
            ArtifactSummaryResponse(
                artifact_id=artifact.artifact_id,
                path=artifact.path,
                summary=artifact.summary,
            )
            for artifact in view.artifacts
        ],
        context_capacity=view.context_capacity,
        last_sequence=view.last_sequence,
        created_at=view.created_at.isoformat() if view.created_at else None,
        updated_at=view.updated_at.isoformat() if view.updated_at else None,
    )


def command_result_response(result: CommandResult) -> CommandResultResponse:
    return CommandResultResponse(
        objective_id=result.objective_id,
        status=result.status,
        replayed=result.replayed,
        payload=result.payload,
    )


def sse_message_from_node(node: TimelineNode) -> dict[str, str]:
    event = objective_event_response(node)
    payload = {"type": "objective_event", **event.model_dump()}
    return {
        "event": "objective_event",
        "id": str(node.sequence),
        "data": json.dumps(payload, default=str),
    }


def objective_ack_sse_message(
    objective_id: str, *, status: str | None = None
) -> dict[str, str]:
    payload: dict[str, Any] = {"type": "objective_ack", "objective_id": objective_id}
    if status is not None:
        payload["status"] = status
    return {
        "event": "objective_ack",
        "id": "0",
        "data": json.dumps(payload, default=str),
    }


__all__ = [
    "root_run_view_response",
    "budget_limits_from_body",
    "command_result_response",
    "objective_ack_sse_message",
    "objective_budget_response",
    "objective_event_response",
    "objective_view_response",
    "sse_message_from_node",
    "timeline_node_response",
    "user_message_from_parts",
]
