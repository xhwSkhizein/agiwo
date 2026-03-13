from datetime import datetime

from agiwo.agent import (
    ChannelContext,
    ContentPart,
    ContentType,
    MessageRole,
    Run,
    RunMetrics,
    RunOutput,
    RunStatus,
    StepRecord,
    StepMetrics,
    TerminationReason,
    UserMessage,
)
from agiwo.agent.execution_context import ExecutionContext, SessionSequenceCounter
from agiwo.agent.inner.run_payloads import (
    apply_run_metrics_payload,
    build_run_completed_event_data,
)
from agiwo.agent.stream_channel import StreamChannel
from agiwo.agent.storage.serialization import (
    deserialize_run_from_storage,
    deserialize_step_from_storage,
    serialize_run_for_storage,
    serialize_step_for_storage,
)


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


def _make_context() -> ExecutionContext:
    return ExecutionContext(
        session_id="session-1",
        run_id="run-1",
        channel=StreamChannel(),
        agent_id="agent-1",
        agent_name="test-agent",
        sequence_counter=SessionSequenceCounter(0),
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
