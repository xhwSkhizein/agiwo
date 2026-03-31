from datetime import datetime, timezone

from agiwo.agent import (
    ContentPart,
    ContentType,
    MessageRole,
    StepCompletedEvent,
    StepMetrics,
    StepRecord,
    UserMessage,
)
from agiwo.scheduler.models import AgentState, AgentStateStatus
from server.schemas import (
    AgentOptionsInput,
    AgentStateResponse,
    ModelParamsInput,
    StepResponse,
    sanitize_agent_options_data,
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
    event = StepCompletedEvent(
        session_id="sess-1",
        run_id="run-1",
        agent_id="agent-1",
        parent_run_id=None,
        depth=0,
        step=step,
        timestamp=datetime(2026, 3, 9, 1, tzinfo=timezone.utc),
    )

    rest_payload = StepResponse.from_sdk(step).model_dump()
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

    payload = StepResponse.from_sdk(step).model_dump()

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

    payload = AgentStateResponse.from_sdk(state).model_dump()

    assert payload["task"][0]["type"] == "text"
    assert payload["task"][0]["text"] == "queued"
    assert payload["pending_input"][0]["type"] == "text"
    assert payload["pending_input"][0]["text"] == "wake me"


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
