"""Shared scheduler event payloads for console SSE boundaries."""

from typing import Any

from pydantic import BaseModel


class SchedulerCompletedEventPayloadData(BaseModel):
    type: str = "scheduler_completed"
    state_id: str
    response: str | None = None
    termination_reason: str | None = None


class SchedulerFailedEventPayloadData(BaseModel):
    type: str = "scheduler_failed"
    error: str


def scheduler_completed_payload(
    *,
    state_id: str,
    response: str | None,
    termination_reason: str | None,
) -> dict[str, Any]:
    return SchedulerCompletedEventPayloadData(
        state_id=state_id,
        response=response,
        termination_reason=termination_reason,
    ).model_dump(exclude_none=True)


def scheduler_failed_payload(error: str) -> dict[str, Any]:
    return SchedulerFailedEventPayloadData(error=error).model_dump()


__all__ = [
    "SchedulerCompletedEventPayloadData",
    "SchedulerFailedEventPayloadData",
    "scheduler_completed_payload",
    "scheduler_failed_payload",
]
