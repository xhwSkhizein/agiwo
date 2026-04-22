"""Helpers for building scheduler parent-event payloads."""

from datetime import datetime, timezone
from uuid import uuid4

from agiwo.scheduler.models import PendingEvent, SchedulerEventType


def build_parent_pending_event(
    *,
    parent_agent_id: str,
    session_id: str,
    source_agent_id: str,
    event_type: SchedulerEventType,
    payload: dict[str, object],
) -> PendingEvent:
    created_at = datetime.now(timezone.utc)
    child_agent_id = source_agent_id

    if event_type == SchedulerEventType.CHILD_FAILED:
        return PendingEvent.create_child_failed(
            id=str(uuid4()),
            target_agent_id=parent_agent_id,
            session_id=session_id,
            child_agent_id=child_agent_id,
            reason=str(payload.get("reason", "")),
            created_at=created_at,
            source_agent_id=source_agent_id,
        )

    if event_type == SchedulerEventType.CHILD_COMPLETED:
        return PendingEvent.create_child_completed(
            id=str(uuid4()),
            target_agent_id=parent_agent_id,
            session_id=session_id,
            child_agent_id=child_agent_id,
            result=str(payload.get("result", "")),
            created_at=created_at,
            source_agent_id=source_agent_id,
        )

    if event_type == SchedulerEventType.CHILD_SLEEP_RESULT:
        return PendingEvent.create_child_sleep_result(
            id=str(uuid4()),
            target_agent_id=parent_agent_id,
            session_id=session_id,
            child_agent_id=child_agent_id,
            result=str(payload.get("result", "")),
            explain=payload.get("explain")
            if isinstance(payload.get("explain"), str)
            else None,
            periodic=bool(payload.get("periodic", False)),
            created_at=created_at,
            source_agent_id=source_agent_id,
        )

    return PendingEvent(
        id=str(uuid4()),
        target_agent_id=parent_agent_id,
        session_id=session_id,
        event_type=event_type,
        payload={
            **payload,
            "child_agent_id": child_agent_id,
        },
        created_at=created_at,
        source_agent_id=source_agent_id,
    )


__all__ = ["build_parent_pending_event"]
