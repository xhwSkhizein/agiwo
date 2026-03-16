from datetime import datetime, timezone

from agiwo.agent import (
    ContentPart,
    ContentType,
    EventType,
    MessageRole,
    StepMetrics,
    StepRecord,
    StreamEvent,
    UserMessage,
    serialize_user_input,
)
from agiwo.scheduler.models import AgentState, AgentStateStatus, WakeCondition, WakeType
from server.domain.agent_configs import (
    AgentOptionsInput,
    ModelParamsInput,
    sanitize_agent_options_data,
)

from server.response_serialization import (
    state_to_response,
    step_to_response,
    stream_event_to_payload,
)
from agiwo.llm.config_policy import sanitize_model_params_data


def test_stream_event_step_payload_matches_rest_step_response() -> None:
    user_input = UserMessage(
        content=[ContentPart(type=ContentType.TEXT, text="hello")],
    )
    step = StepRecord(
        id="step-1",
        session_id="sess-1",
        run_id="run-1",
        sequence=1,
        role=MessageRole.USER,
        user_input=user_input,
        content=[{"type": "text", "text": "hello"}],
        created_at=datetime(2026, 3, 9, tzinfo=timezone.utc),
    )
    event = StreamEvent(
        type=EventType.STEP_COMPLETED,
        run_id="run-1",
        step=step,
        agent_id="agent-1",
        timestamp=datetime(2026, 3, 9, 1, tzinfo=timezone.utc),
    )

    rest_payload = step_to_response(step).model_dump()
    stream_payload = stream_event_to_payload(event)

    assert stream_payload["step"] == rest_payload
    assert stream_payload["step"]["user_input"]["content"][0]["type"] == "text"
    assert stream_payload["step"]["user_input"]["content"][0]["text"] == "hello"


def test_step_response_includes_usage_source_in_metrics() -> None:
    step = StepRecord(
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

    payload = step_to_response(step).model_dump()

    assert payload["metrics"]["usage_source"] == "estimated"
    assert payload["metrics"]["model_name"] == "gpt-test"


def test_scheduler_state_response_normalizes_serialized_user_input() -> None:
    state = AgentState(
        id="agent-1",
        session_id="sess-1",
        status=AgentStateStatus.SLEEPING,
        task=serialize_user_input([ContentPart(type=ContentType.TEXT, text="queued")]),
        wake_condition=WakeCondition(
            type=WakeType.TASK_SUBMITTED,
            submitted_task=serialize_user_input(
                [ContentPart(type=ContentType.TEXT, text="wake me")]
            ),
        ),
    )

    payload = state_to_response(state).model_dump()

    assert payload["task"][0]["type"] == "text"
    assert payload["task"][0]["text"] == "queued"
    assert payload["wake_condition"]["submitted_task"][0]["type"] == "text"
    assert payload["wake_condition"]["submitted_task"][0]["text"] == "wake me"


def test_agent_config_schema_and_registry_share_normalization_policy() -> None:
    option_input = {"skills_dirs": " ./skills "}
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
