from agiwo.scheduler.models import SchedulerEventType
from agiwo.scheduler.runner_events import build_parent_pending_event


def test_build_parent_pending_event_creates_child_failed_payload() -> None:
    event = build_parent_pending_event(
        parent_agent_id="parent-1",
        session_id="sess-1",
        source_agent_id="child-1",
        event_type=SchedulerEventType.CHILD_FAILED,
        payload={"reason": "boom"},
    )

    assert event.target_agent_id == "parent-1"
    assert event.session_id == "sess-1"
    assert event.source_agent_id == "child-1"
    assert event.event_type is SchedulerEventType.CHILD_FAILED
    assert event.get_payload_child_failed() is not None
    assert event.get_payload_child_failed().reason == "boom"


def test_build_parent_pending_event_creates_sleep_result_payload() -> None:
    event = build_parent_pending_event(
        parent_agent_id="parent-1",
        session_id="sess-1",
        source_agent_id="child-1",
        event_type=SchedulerEventType.CHILD_SLEEP_RESULT,
        payload={"result": "waiting", "explain": "pause", "periodic": True},
    )

    payload = event.get_payload_child_sleep_result()

    assert payload is not None
    assert payload.result == "waiting"
    assert payload.explain == "pause"
    assert payload.periodic is True


def test_build_parent_pending_event_falls_back_for_unknown_event_type() -> None:
    event = build_parent_pending_event(
        parent_agent_id="parent-1",
        session_id="sess-1",
        source_agent_id="child-1",
        event_type=SchedulerEventType.USER_HINT,
        payload={"reason": "custom"},
    )

    assert event.event_type is SchedulerEventType.USER_HINT
    assert event.payload["child_agent_id"] == "child-1"
    assert event.payload["reason"] == "custom"
