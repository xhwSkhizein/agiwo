from datetime import datetime

from agiwo.agent import ChannelContext, ContentPart, ContentType, UserMessage
from agiwo.agent import StepCompletedEvent
from agiwo.agent.models.log import RunFinished, RunStarted, UserStepCommitted
from agiwo.agent.models.step import StepRecord as InternalStepRecord
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.storage.base import InMemoryRunStepStorage
from agiwo.agent.storage.serialization import (
    build_run_view_from_entries,
    build_step_views_from_entries,
    deserialize_run_log_entry_from_storage,
    deserialize_run_from_storage,
    deserialize_step_from_storage,
    serialize_run_log_entry_for_storage,
    serialize_run_for_storage,
    serialize_step_for_storage,
)
from agiwo.agent.models.stream import (
    StepCompletedEvent as InternalStepCompletedEvent,
)
from agiwo.agent import (
    MessageRole,
    Run,
    RunMetrics,
    RunOutput,
    RunStatus,
    StepRecord,
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
    assert StepRecord is InternalStepRecord
    assert StepCompletedEvent is InternalStepCompletedEvent


def test_step_record_to_message_is_public_conversion_surface() -> None:
    step = StepRecord.user(_make_context(), sequence=1, user_input="hello")

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


def test_run_storage_round_trip_restores_structured_user_input_and_status() -> None:
    run = Run(
        id="run-1",
        agent_id="agent-1",
        session_id="session-1",
        user_input=UserMessage(
            content=[ContentPart(type=ContentType.TEXT, text="hello")],
            context=ChannelContext(source="api", metadata={"channel": "test"}),
        ),
        status=RunStatus.COMPLETED,
    )

    payload = serialize_run_for_storage(run)
    restored = deserialize_run_from_storage(payload)

    assert restored.status == RunStatus.COMPLETED
    assert isinstance(restored.user_input, UserMessage)
    assert restored.user_input.context is not None
    assert restored.user_input.context.source == "api"


def _make_context():
    return RunContext(
        session_runtime=SessionRuntime(
            session_id="session-1",
            run_step_storage=InMemoryRunStepStorage(),
        ),
        run_id="run-1",
        agent_id="agent-1",
        agent_name="test-agent",
    )


def test_step_storage_round_trip_derives_user_content_from_user_input() -> None:
    step = StepRecord(
        id="step-1",
        session_id="session-1",
        run_id="run-1",
        sequence=1,
        role=MessageRole.USER,
        user_input=[
            ContentPart(type=ContentType.TEXT, text="hello"),
            ContentPart(type=ContentType.IMAGE, url="https://example.com/a.png"),
        ],
        created_at=datetime(2026, 3, 8, 12, 0, 0),
    )

    payload = serialize_step_for_storage(step)

    assert "content" not in payload

    restored = deserialize_step_from_storage(payload)

    assert restored.role == MessageRole.USER
    assert restored.user_input is not None
    assert isinstance(restored.content, list)
    assert restored.content[0]["text"] == "hello"
    assert restored.content[1]["type"] == "image_url"


def test_step_record_direct_user_construction_derives_content() -> None:
    step = StepRecord(
        id="step-direct-user",
        session_id="session-1",
        run_id="run-1",
        sequence=1,
        role=MessageRole.USER,
        user_input=[
            ContentPart(type=ContentType.TEXT, text="hello"),
            ContentPart(type=ContentType.IMAGE, url="https://example.com/a.png"),
        ],
        created_at=datetime(2026, 3, 8, 12, 0, 0),
    )

    assert isinstance(step.content, list)
    assert step.content[0]["text"] == "hello"
    assert step.content[1]["type"] == "image_url"
    assert step.to_message()["content"] == step.content


def test_step_record_user_factory_preserves_name_override() -> None:
    step = StepRecord.user(
        _make_context(),
        sequence=1,
        content="summary prompt",
        name="summary_request",
    )

    assert step.name == "summary_request"


def test_step_storage_deserializer_ignores_backend_only_fields() -> None:
    payload = {
        "id": "step-2",
        "session_id": "session-1",
        "run_id": "run-1",
        "sequence": 2,
        "role": "assistant",
        "content": "done",
        "trace_id": "trace-1",
        "span_id": "span-1",
        "llm_messages": [],
    }

    restored = deserialize_step_from_storage(payload)

    assert restored.id == "step-2"
    assert restored.role == MessageRole.ASSISTANT
    assert restored.content == "done"


def test_run_storage_deserializer_restores_datetime_fields_from_iso_strings() -> None:
    restored = deserialize_run_from_storage(
        {
            "id": "run-iso",
            "agent_id": "agent-1",
            "session_id": "session-1",
            "user_input": "hello",
            "status": "completed",
            "created_at": "2026-03-17T10:11:12+00:00",
            "updated_at": "2026-03-17T10:11:13+00:00",
        }
    )

    assert restored.status == RunStatus.COMPLETED
    assert isinstance(restored.created_at, datetime)
    assert isinstance(restored.updated_at, datetime)


def test_step_storage_deserializer_restores_datetime_fields_from_iso_strings() -> None:
    restored = deserialize_step_from_storage(
        {
            "id": "step-iso",
            "session_id": "session-1",
            "run_id": "run-1",
            "sequence": 3,
            "role": "assistant",
            "content": "done",
            "metrics": {
                "start_at": "2026-03-17T10:11:12+00:00",
                "end_at": "2026-03-17T10:11:13+00:00",
                "usage_source": "estimated",
            },
            "created_at": "2026-03-17T10:11:12+00:00",
        }
    )

    assert isinstance(restored.created_at, datetime)
    assert restored.metrics is not None
    assert isinstance(restored.metrics.start_at, datetime)
    assert isinstance(restored.metrics.end_at, datetime)
    assert restored.metrics.usage_source == "estimated"


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


def test_step_storage_round_trip_preserves_usage_source() -> None:
    step = StepRecord.assistant(
        _make_context(),
        sequence=1,
        content="done",
        metrics=StepMetrics(
            input_tokens=1,
            output_tokens=2,
            total_tokens=3,
            usage_source="estimated",
        ),
    )

    payload = serialize_step_for_storage(step)
    restored = deserialize_step_from_storage(payload)

    assert restored.metrics is not None
    assert restored.metrics.usage_source == "estimated"


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
    assert run_view.status == "completed"
    assert run_view.user_id == "user-1"
    assert run_view.parent_run_id == "parent-run-1"
    assert len(step_views) == 1
    assert step_views[0].id == "run-1:2"
    assert step_views[0].content == "hello"
