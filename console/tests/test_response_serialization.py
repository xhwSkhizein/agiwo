from datetime import datetime, timezone

from agiwo.agent import (
    ContentPart,
    ContentType,
    MessageRole,
    StepCompletedEvent,
    StepMetrics,
    StepView,
    TerminationReason,
    UserMessage,
)
from agiwo.scheduler.models import AgentState, AgentStateStatus, SchedulerRunResult
from agiwo.llm.config_policy import sanitize_model_params_data

from server.models.agent_config import (
    AgentOptionsInput,
    ModelParamsInput,
    sanitize_agent_options_data,
)
from server.models.session import (
    ConversationEventRecord,
    MilestoneRecord,
    ReviewCheckpointRecord,
    ReviewCycleRecord,
    ReviewOutcomeRecord,
    SessionDetailRecord,
    SessionMilestoneBoardRecord,
    SessionSummaryRecord,
    TraceLlmCallRecord,
)
from server.response_serialization import (
    agent_state_response_from_sdk,
    session_detail_response_from_record,
    step_response_from_sdk,
    stream_event_to_payload,
    trace_llm_call_response_from_record,
)


def test_stream_event_step_payload_matches_rest_step_response() -> None:
    user_input = UserMessage(
        content=[ContentPart(type=ContentType.TEXT, text="hello")],
    )
    step = StepView(
        id="step-1",
        session_id="sess-1",
        run_id="run-1",
        sequence=1,
        role=MessageRole.USER,
        user_input=user_input,
        content=[{"type": "text", "text": "hello"}],
        created_at=datetime(2026, 3, 9, tzinfo=timezone.utc),
    )
    event = StepCompletedEvent(
        session_id="sess-1",
        run_id="run-1",
        agent_id="agent-1",
        parent_run_id=None,
        depth=0,
        step=step,
        timestamp=datetime(2026, 3, 9, 1, tzinfo=timezone.utc),
    )

    rest_payload = step_response_from_sdk(step).model_dump()
    stream_payload = stream_event_to_payload(event)

    assert stream_payload["step"] == rest_payload
    assert stream_payload["step"]["user_input"]["content"][0]["type"] == "text"
    assert stream_payload["step"]["user_input"]["content"][0]["text"] == "hello"


def test_step_response_includes_usage_source_in_metrics() -> None:
    step = StepView(
        id="step-usage",
        session_id="sess-1",
        run_id="run-1",
        sequence=2,
        role=MessageRole.ASSISTANT,
        content="done",
        metrics=StepMetrics(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            usage_source="estimated",
            model_name="gpt-test",
        ),
        created_at=datetime(2026, 3, 9, tzinfo=timezone.utc),
    )

    payload = step_response_from_sdk(step).model_dump()

    assert payload["metrics"]["usage_source"] == "estimated"
    assert payload["metrics"]["model_name"] == "gpt-test"


def test_scheduler_state_response_normalizes_serialized_user_input() -> None:
    state = AgentState(
        id="agent-1",
        session_id="sess-1",
        status=AgentStateStatus.QUEUED,
        task=UserMessage.serialize([ContentPart(type=ContentType.TEXT, text="queued")]),
        pending_input=UserMessage.serialize(
            [ContentPart(type=ContentType.TEXT, text="wake me")]
        ),
    )

    payload = agent_state_response_from_sdk(state).model_dump()

    assert payload["task"][0]["type"] == "text"
    assert payload["task"][0]["text"] == "queued"
    assert payload["pending_input"][0]["type"] == "text"
    assert payload["pending_input"][0]["text"] == "wake me"


def test_scheduler_state_response_includes_last_run_result() -> None:
    state = AgentState(
        id="agent-1",
        session_id="sess-1",
        status=AgentStateStatus.IDLE,
        task="done",
        last_run_result=SchedulerRunResult(
            run_id="run-1",
            termination_reason=TerminationReason.COMPLETED,
            summary="finished",
        ),
    )

    payload = agent_state_response_from_sdk(state).model_dump()

    assert payload["last_run_result"]["run_id"] == "run-1"
    assert payload["last_run_result"]["termination_reason"] == "completed"
    assert payload["last_run_result"]["summary"] == "finished"


def test_agent_config_view_model_and_registry_share_normalization_policy() -> None:
    option_input = {"max_steps": 42}
    option_schema = AgentOptionsInput.model_validate(option_input)
    model_param_input = {
        "base_url": " https://api.example.com/v1 ",
        "api_key_env_name": " TEST_API_KEY ",
    }
    model_param_schema = ModelParamsInput.model_validate(model_param_input)

    assert sanitize_agent_options_data(option_input) == option_schema.model_dump(
        exclude_none=True,
        exclude_defaults=True,
    )
    assert sanitize_model_params_data(
        model_param_input,
        reject_plain_api_key=False,
    ) == model_param_schema.model_dump(
        exclude_none=True,
        exclude_defaults=True,
    )


def test_session_detail_serializes_new_mainline_fields() -> None:
    detail = SessionDetailRecord(
        summary=SessionSummaryRecord(session_id="sess-1"),
        milestone_board=SessionMilestoneBoardRecord(
            session_id="sess-1",
            run_id="run-1",
            milestones=[
                MilestoneRecord(
                    id="inspect",
                    description="Inspect the auth flow",
                    status="active",
                    declared_at_seq=3,
                )
            ],
            active_milestone_id="inspect",
            latest_checkpoint=ReviewCheckpointRecord(
                seq=8,
                milestone_id="inspect",
                confirmed_at=datetime(2026, 4, 25, tzinfo=timezone.utc),
            ),
            latest_review_outcome=ReviewOutcomeRecord(
                aligned=True,
                step_back_applied=False,
                trigger_reason="step_interval",
                active_milestone="Inspect the auth flow",
            ),
            pending_review_reason=None,
        ),
        review_cycles=[
            ReviewCycleRecord(
                cycle_id="run-1:8",
                run_id="run-1",
                agent_id="agent-1",
                trigger_reason="step_interval",
                steps_since_last_review=8,
                active_milestone="Inspect the auth flow",
                active_milestone_id="inspect",
                aligned=True,
                experience=None,
                step_back_applied=False,
            )
        ],
        conversation_events=[
            ConversationEventRecord(
                id="evt-1",
                session_id="sess-1",
                run_id="run-1",
                sequence=10,
                kind="assistant_message",
                priority="primary",
                title="Assistant",
                summary="Auth check is in auth.py",
                details={},
            )
        ],
    )

    payload = session_detail_response_from_record(detail)

    assert payload.milestone_board is not None
    assert payload.milestone_board.active_milestone_id == "inspect"
    assert payload.review_cycles[0].active_milestone_id == "inspect"
    assert payload.review_cycles[0].trigger_reason == "step_interval"
    assert payload.conversation_events[0].kind == "assistant_message"


def test_trace_llm_call_serialization_preserves_summary_fields() -> None:
    record = TraceLlmCallRecord(
        span_id="span-1",
        run_id="run-1",
        agent_id="agent-1",
        model="gpt-5.4",
        provider="openai-response",
        finish_reason="stop",
        duration_ms=1234.0,
        first_token_latency_ms=345.0,
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
        message_count=6,
        tool_schema_count=2,
        response_tool_call_count=1,
        output_preview="Looks aligned.",
    )

    payload = trace_llm_call_response_from_record(record)

    assert payload.model == "gpt-5.4"
    assert payload.response_tool_call_count == 1
    assert payload.output_preview == "Looks aligned."
