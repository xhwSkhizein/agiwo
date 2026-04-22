from datetime import datetime

from agiwo.agent import ChannelContext, ContentPart, ContentType, UserMessage
from agiwo.agent import StepCompletedEvent
from agiwo.agent.models.log import RunFinished, RunStarted, UserStepCommitted
from agiwo.agent.models.log import AssistantStepCommitted, RunRolledBack
from agiwo.agent.models.run import RunIdentity
from agiwo.agent.models.step import StepView as InternalStepView
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.storage.base import InMemoryRunLogStorage
from agiwo.agent.storage.serialization import (
    build_run_view_from_entries,
    build_step_views_from_entries,
    deserialize_run_log_entry_from_storage,
    serialize_run_log_entry_for_storage,
)
from agiwo.agent.models.stream import (
    StepCompletedEvent as InternalStepCompletedEvent,
)
from agiwo.agent import (
    MessageRole,
    RunMetrics,
    RunOutput,
    RunStatus,
    StepView,
    StepMetrics,
    TerminationReason,
)


_RUN_COMPLETED_METRIC_DEFAULTS: dict[str, int | float] = {
    "duration_ms": 0.0,
    "total_tokens": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_read_tokens": 0,
    "cache_creation_tokens": 0,
    "token_cost": 0.0,
    "steps_count": 0,
    "tool_calls_count": 0,
}


def test_public_agent_exports_remain_stable_after_internal_type_split() -> None:
    assert StepView is InternalStepView
    assert StepCompletedEvent is InternalStepCompletedEvent


def test_step_view_to_message_is_public_conversion_surface() -> None:
    step = StepView.user(_make_context(), sequence=1, user_input="hello")

    msg = step.to_message()
    assert msg["role"] == MessageRole.USER.value
    assert msg["content"] == "hello"
    assert msg["_sequence"] == 1


def apply_run_metrics_payload(metrics: RunMetrics, payload: dict | None) -> None:
    if not payload:
        return

    for field_name, default in _RUN_COMPLETED_METRIC_DEFAULTS.items():
        setattr(metrics, field_name, payload.get(field_name, default))


def build_run_completed_event_data(result: RunOutput) -> dict:
    metrics_payload = dict(_RUN_COMPLETED_METRIC_DEFAULTS)
    if result.metrics is not None:
        for field_name in metrics_payload:
            metrics_payload[field_name] = getattr(
                result.metrics, field_name, metrics_payload[field_name]
            )

    data = {
        "response": result.response or "",
        "metrics": metrics_payload,
    }
    if result.termination_reason is not None:
        data["termination_reason"] = result.termination_reason.value
    return data


def _make_context():
    return RunContext(
        identity=RunIdentity(
            run_id="run-1",
            agent_id="agent-1",
            agent_name="test-agent",
        ),
        session_runtime=SessionRuntime(
            session_id="session-1",
            run_log_storage=InMemoryRunLogStorage(),
        ),
    )


def test_step_view_direct_user_construction_derives_content() -> None:
    step = StepView.user(
        _make_context(),
        sequence=1,
        content=[
            ContentPart(type=ContentType.TEXT, text="hello"),
            ContentPart(type=ContentType.IMAGE, url="https://example.com/a.png"),
        ],
    )
    step.id = "step-direct-user"
    step.created_at = datetime(2026, 3, 8, 12, 0, 0)

    assert isinstance(step.content, list)
    assert step.content[0].text == "hello"
    assert step.content[1].type is ContentType.IMAGE
    assert step.to_message()["content"] == step.content


def test_step_view_user_factory_preserves_name_override() -> None:
    step = StepView.user(
        _make_context(),
        sequence=1,
        content="summary prompt",
        name="summary_request",
    )

    assert step.name == "summary_request"


def test_run_completed_payload_round_trip_uses_shared_metrics_mapping() -> None:
    result = RunOutput(
        response="done",
        metrics=RunMetrics(
            duration_ms=123.0,
            total_tokens=45,
            input_tokens=12,
            output_tokens=33,
            cache_read_tokens=4,
            cache_creation_tokens=5,
            token_cost=0.12,
            steps_count=6,
            tool_calls_count=2,
        ),
        termination_reason=TerminationReason.MAX_STEPS,
    )

    payload = build_run_completed_event_data(result)
    restored = RunMetrics()
    apply_run_metrics_payload(restored, payload["metrics"])

    assert payload["termination_reason"] == "max_steps"
    assert restored.duration_ms == 123.0
    assert restored.total_tokens == 45
    assert restored.input_tokens == 12
    assert restored.output_tokens == 33
    assert restored.cache_read_tokens == 4
    assert restored.cache_creation_tokens == 5
    assert restored.token_cost == 0.12
    assert restored.steps_count == 6
    assert restored.tool_calls_count == 2


def test_step_metrics_to_dict_tolerates_string_timestamps() -> None:
    metrics = StepMetrics(usage_source="estimated")
    metrics.start_at = "2026-03-17T10:11:12+00:00"
    metrics.end_at = "2026-03-17T10:11:13+00:00"

    payload = metrics.to_dict()

    assert payload["start_at"] == "2026-03-17T10:11:12+00:00"
    assert payload["end_at"] == "2026-03-17T10:11:13+00:00"


def test_run_log_entry_storage_round_trip_restores_structured_user_input() -> None:
    entry = RunStarted(
        sequence=1,
        session_id="session-1",
        run_id="run-1",
        agent_id="agent-1",
        user_id="user-1",
        parent_run_id="parent-run-1",
        user_input=UserMessage(
            content=[ContentPart(type=ContentType.TEXT, text="hello")],
            context=ChannelContext(source="api"),
        ),
    )

    payload = serialize_run_log_entry_for_storage(entry)
    restored = deserialize_run_log_entry_from_storage(payload)

    assert isinstance(restored, RunStarted)
    assert isinstance(restored.user_input, UserMessage)
    assert restored.user_input.content[0].text == "hello"
    assert restored.user_id == "user-1"
    assert restored.parent_run_id == "parent-run-1"


def test_build_run_and_step_views_from_run_log_entries() -> None:
    entries = [
        RunStarted(
            sequence=1,
            session_id="session-1",
            run_id="run-1",
            agent_id="agent-1",
            user_id="user-1",
            parent_run_id="parent-run-1",
            user_input="hello",
        ),
        UserStepCommitted(
            sequence=2,
            session_id="session-1",
            run_id="run-1",
            agent_id="agent-1",
            step_id="step-2",
            role=MessageRole.USER,
            content="hello",
            user_input="hello",
        ),
        RunFinished(
            sequence=3,
            session_id="session-1",
            run_id="run-1",
            agent_id="agent-1",
            response="world",
            termination_reason=TerminationReason.COMPLETED,
            metrics=RunMetrics(
                duration_ms=1.0,
                total_tokens=3,
                steps_count=1,
            ).to_dict(),
        ),
    ]

    run_view = build_run_view_from_entries(entries)
    step_views = build_step_views_from_entries(entries)

    assert run_view is not None
    assert run_view.run_id == "run-1"
    assert run_view.response == "world"
    assert run_view.last_user_input == "hello"
    assert run_view.status == RunStatus.COMPLETED
    assert run_view.user_id == "user-1"
    assert run_view.parent_run_id == "parent-run-1"
    assert len(step_views) == 1
    assert step_views[0].id == "step-2"
    assert step_views[0].content == "hello"


def test_build_step_views_hides_steps_covered_by_rollback_fact() -> None:
    entries = [
        UserStepCommitted(
            sequence=10,
            session_id="sess-1",
            run_id="run-1",
            agent_id="agent-1",
            step_id="step-10",
            role=MessageRole.USER,
            content="u1",
            user_input="u1",
        ),
        AssistantStepCommitted(
            sequence=11,
            session_id="sess-1",
            run_id="run-1",
            agent_id="agent-1",
            step_id="step-11",
            role=MessageRole.ASSISTANT,
            content="a1",
        ),
        RunRolledBack(
            sequence=12,
            session_id="sess-1",
            run_id="run-1",
            agent_id="agent-1",
            start_sequence=10,
            end_sequence=11,
            reason="no_progress",
        ),
    ]

    assert build_step_views_from_entries(entries) == []


def test_build_step_views_can_include_rolled_back_steps() -> None:
    entries = [
        UserStepCommitted(
            sequence=10,
            session_id="sess-1",
            run_id="run-1",
            agent_id="agent-1",
            step_id="step-10",
            role=MessageRole.USER,
            content="u1",
            user_input="u1",
        ),
        RunRolledBack(
            sequence=11,
            session_id="sess-1",
            run_id="run-1",
            agent_id="agent-1",
            start_sequence=10,
            end_sequence=10,
            reason="no_progress",
        ),
    ]

    visible = build_step_views_from_entries(entries)
    all_steps = build_step_views_from_entries(entries, include_rolled_back=True)

    assert visible == []
    assert [step.id for step in all_steps] == ["step-10"]
