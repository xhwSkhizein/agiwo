"""Tests for console model conversions and utilities."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from agiwo.agent import UserInput
from agiwo.agent.models.run import Run, RunMetrics, RunStatus
from agiwo.agent.models.step import MessageRole, StepMetrics, StepRecord
from agiwo.scheduler.models import TimeUnit

from server.models.view import (
    AgentStateBase,
    AgentStateResponse,
    RunMetricsResponse,
    WakeConditionResponse,
    extract_content_parts,
)
from server.response_serialization import (
    run_response_from_sdk,
    step_metrics_response_from_sdk,
    step_response_from_sdk,
)
from server.services.tool_catalog.tool_references import (
    InvalidToolReferenceError,
    parse_tool_reference,
    parse_tool_references,
)


class TestStepMetricsResponseFromSdk:
    """Test StepMetricsResponse.from_sdk maps all fields correctly."""

    def test_from_sdk_maps_all_fields(self):
        """Verify all StepMetrics fields are mapped to StepMetricsResponse."""
        metrics = StepMetrics(
            duration_ms=1234.5,
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            cache_read_tokens=50,
            cache_creation_tokens=25,
            token_cost=0.015,
            usage_source="test-source",
            model_name="test-model",
            provider="test-provider",
            first_token_latency_ms=100.5,
        )

        response = step_metrics_response_from_sdk(metrics)

        assert response.duration_ms == 1234.5
        assert response.input_tokens == 100
        assert response.output_tokens == 200
        assert response.total_tokens == 300
        assert response.cache_read_tokens == 50
        assert response.cache_creation_tokens == 25
        assert response.token_cost == 0.015
        assert response.usage_source == "test-source"
        assert response.model_name == "test-model"
        assert response.provider == "test-provider"
        assert response.first_token_latency_ms == 100.5

    def test_from_sdk_handles_none_values(self):
        """Verify from_sdk handles None metric values."""
        metrics = StepMetrics()

        response = step_metrics_response_from_sdk(metrics)

        assert response.duration_ms is None
        assert response.input_tokens is None
        assert response.output_tokens is None
        assert response.total_tokens is None
        assert response.cache_read_tokens is None
        assert response.cache_creation_tokens is None
        assert response.token_cost is None


class TestWakeConditionTimeUnitSerialization:
    """Test time_unit is serialized as string in WakeConditionResponse."""

    def test_time_unit_enum_converted_to_string(self):
        """Verify TimeUnit enum is converted to its string value."""
        # Create a mock wake condition with TimeUnit enum
        wc = MagicMock()
        wc.type = MagicMock()
        wc.type.value = "timer"
        wc.wait_for = []
        wc.wait_mode = MagicMock()
        wc.wait_mode.value = "all"
        wc.completed_ids = []
        wc.time_value = 5.0
        wc.time_unit = TimeUnit.MINUTES
        wc.wakeup_at = None
        wc.timeout_at = None

        response = WakeConditionResponse(
            type=wc.type.value if hasattr(wc.type, "value") else str(wc.type),
            wait_for=list(getattr(wc, "wait_for", []) or []),
            wait_mode=wc.wait_mode.value
            if hasattr(wc.wait_mode, "value")
            else str(getattr(wc, "wait_mode", "all")),
            completed_ids=list(getattr(wc, "completed_ids", []) or []),
            time_value=getattr(wc, "time_value", None),
            time_unit=wc.time_unit.value
            if hasattr(getattr(wc, "time_unit", None), "value")
            else getattr(wc, "time_unit", None),
            wakeup_at=None,
            timeout_at=None,
        )

        assert response.time_unit == "minutes"  # Enum converted to string
        assert isinstance(response.time_unit, str)

    def test_time_unit_none_handled(self):
        """Verify None time_unit is handled correctly."""
        wc = MagicMock()
        wc.time_unit = None

        time_unit = (
            wc.time_unit.value
            if hasattr(getattr(wc, "time_unit", None), "value")
            else getattr(wc, "time_unit", None)
        )

        assert time_unit is None


class TestExtractContentParts:
    """Test the shared extract_content_parts helper function."""

    def test_extracts_content_parts_from_json_string(self):
        """Verify extraction of content_parts from JSON string."""
        json_str = json.dumps(
            {"__type": "content_parts", "parts": ["text", "more text"]}
        )

        result = extract_content_parts(json_str)

        assert result == ["text", "more text"]

    def test_extracts_plain_dict_from_json_string(self):
        """Verify plain dict is returned when no content_parts wrapper."""
        json_str = json.dumps({"key": "value"})

        result = extract_content_parts(json_str)

        assert result == {"key": "value"}

    def test_returns_string_on_invalid_json(self):
        """Verify original string is returned for invalid JSON."""
        invalid_json = "not valid json"

        result = extract_content_parts(invalid_json)

        assert result == "not valid json"

    def test_returns_non_string_value_unchanged(self):
        """Verify non-string values are returned unchanged."""
        list_value = ["item1", "item2"]

        result = extract_content_parts(list_value)

        assert result == ["item1", "item2"]

    def test_empty_parts_list_handled(self):
        """Verify empty parts list is returned correctly."""
        json_str = json.dumps({"__type": "content_parts", "parts": []})

        result = extract_content_parts(json_str)

        assert result == []


class TestAgentStateValidators:
    """Test AgentState validators use shared helper."""

    def test_extract_content_parts_used_by_agent_state_base(self):
        """Verify AgentStateBase task field validator uses extract_content_parts."""
        json_str = json.dumps({"__type": "content_parts", "parts": ["hello", "world"]})

        # The validator should convert the JSON string to the parts list
        result = AgentStateBase._extract_content_parts(json_str)
        assert result == ["hello", "world"]

    def test_extract_content_parts_used_by_agent_state_response(self):
        """Verify AgentStateResponse pending_input validator uses extract_content_parts."""
        json_str = json.dumps(
            {"__type": "content_parts", "parts": [{"type": "text", "text": "hello"}]}
        )

        # The validator should convert the JSON string
        result = AgentStateResponse._extract_pending_input(json_str)
        assert result == [{"type": "text", "text": "hello"}]

    def test_extract_content_parts_returns_non_string_unchanged(self):
        """Verify validators pass through non-string values unchanged."""
        list_value = ["item1", "item2"]
        result = AgentStateBase._extract_content_parts(list_value)
        assert result == ["item1", "item2"]


class TestStepResponseToolCalls:
    """Test StepResponse handles tool_calls correctly."""

    def test_from_sdk_accepts_dict_list_for_tool_calls(self):
        """Verify StepResponse.from_sdk accepts plain dict list for tool_calls."""
        # StepRecord.tool_calls is list[dict], not pydantic models
        tool_calls = [{"id": "1", "type": "function", "function": {"name": "test"}}]

        # Create a minimal StepRecord-like object
        step = MagicMock(spec=StepRecord)
        step.id = "step-1"
        step.session_id = "session-1"
        step.run_id = "run-1"
        step.sequence = 1
        step.role = MessageRole.ASSISTANT
        step.agent_id = "agent-1"
        step.content = "content"
        step.content_for_user = None
        step.reasoning_content = None
        step.user_input = None
        step.tool_calls = tool_calls  # This is already a list of dicts
        step.tool_call_id = None
        step.name = None
        step.metrics = None
        step.created_at = datetime.now(timezone.utc)
        step.parent_run_id = None
        step.depth = 0

        # This should NOT call model_dump() on dict elements
        response = step_response_from_sdk(step)

        assert response.tool_calls == tool_calls
        assert response.tool_calls[0]["id"] == "1"

    def test_from_sdk_handles_none_tool_calls(self):
        """Verify StepResponse.from_sdk handles None tool_calls."""
        step = MagicMock(spec=StepRecord)
        step.id = "step-1"
        step.session_id = "session-1"
        step.run_id = "run-1"
        step.sequence = 1
        step.role = MessageRole.ASSISTANT
        step.agent_id = "agent-1"
        step.content = "content"
        step.content_for_user = None
        step.reasoning_content = None
        step.user_input = None
        step.tool_calls = None
        step.tool_call_id = None
        step.name = None
        step.metrics = None
        step.created_at = datetime.now(timezone.utc)
        step.parent_run_id = None
        step.depth = 0

        response = step_response_from_sdk(step)

        assert response.tool_calls is None


class TestRunResponseFromSdk:
    """Test RunResponse.from_sdk converter."""

    def test_from_sdk_converts_run_correctly(self):
        """Verify RunResponse.from_sdk maps all Run fields."""
        now = datetime.now(timezone.utc)
        user_input: UserInput = "test input"

        run = Run(
            id="run-1",
            agent_id="agent-1",
            session_id="session-1",
            user_id="user-1",
            user_input=user_input,
            status=RunStatus.COMPLETED,
            response_content="test response",
            metrics=RunMetrics(
                duration_ms=1000.0,
                input_tokens=50,
                output_tokens=100,
                total_tokens=150,
                cache_read_tokens=25,
                cache_creation_tokens=10,
                token_cost=0.005,
                steps_count=3,
                tool_calls_count=1,
            ),
            created_at=now,
            updated_at=now,
            parent_run_id="parent-run-1",
        )

        response = run_response_from_sdk(run)

        assert response.id == "run-1"
        assert response.agent_id == "agent-1"
        assert response.session_id == "session-1"
        assert response.user_id == "user-1"
        assert response.user_input == user_input
        assert response.status == "completed"
        assert response.response_content == "test response"
        assert response.parent_run_id == "parent-run-1"

        # Check metrics
        assert response.metrics is not None
        assert response.metrics.duration_ms == 1000.0
        assert response.metrics.input_tokens == 50
        assert response.metrics.output_tokens == 100
        assert response.metrics.total_tokens == 150
        assert response.metrics.cache_read_tokens == 25
        assert response.metrics.cache_creation_tokens == 10
        assert response.metrics.token_cost == 0.005
        assert response.metrics.steps_count == 3
        assert response.metrics.tool_calls_count == 1

        # Check timestamps are ISO formatted
        assert response.created_at == now.isoformat()
        assert response.updated_at == now.isoformat()

    def test_from_sdk_handles_enum_status(self):
        """Verify status enum is converted to string value."""
        user_input: UserInput = "test"

        run = Run(
            id="run-1",
            agent_id="agent-1",
            session_id="session-1",
            user_input=user_input,
            status=RunStatus.RUNNING,
        )

        response = run_response_from_sdk(run)

        assert response.status == "running"

    def test_from_sdk_handles_none_metrics(self):
        """Verify RunResponse.from_sdk handles None metrics."""
        user_input: UserInput = "test"

        run = Run(
            id="run-1",
            agent_id="agent-1",
            session_id="session-1",
            user_input=user_input,
            status=RunStatus.COMPLETED,
            metrics=RunMetrics(),  # Empty metrics
        )

        response = run_response_from_sdk(run)

        assert response.metrics is not None
        assert response.metrics.duration_ms == 0.0


class TestToolReferenceLazyLoading:
    """Test tool_reference lazy loads builtin tools."""

    def test_ensure_builtin_tools_loaded_called_on_parse(self):
        """Verify ensure_builtin_tools_loaded is called when parsing tool references."""
        with patch(
            "server.services.tool_catalog.tool_references.ensure_builtin_tools_loaded"
        ) as mock_load:
            with patch(
                "server.services.tool_catalog.tool_references.BUILTIN_TOOLS",
                {"bash": MagicMock()},
            ):
                # First call should trigger lazy loading
                parse_tool_reference("bash")
                mock_load.assert_called_once()

    def test_invalid_tool_raises_error(self):
        """Verify invalid tool reference raises InvalidToolReferenceError."""
        with patch(
            "server.services.tool_catalog.tool_references.BUILTIN_TOOLS",
            {"bash": MagicMock()},
        ):
            with pytest.raises(InvalidToolReferenceError):
                parse_tool_reference("nonexistent_tool")

    def test_agent_tool_reference_parsed_correctly(self):
        """Verify agent: prefix tool references are parsed correctly."""
        with patch("server.services.tool_catalog.tool_references.BUILTIN_TOOLS", {}):
            result = parse_tool_reference("agent:test-agent")
            assert result == "agent:test-agent"

    def test_empty_agent_id_raises_error(self):
        """Verify empty agent ID after prefix raises error."""
        with pytest.raises(InvalidToolReferenceError):
            parse_tool_reference("agent:  ")

    def test_parse_tool_references_bulk(self):
        """Verify parse_tool_references works for list of tools."""
        with patch(
            "server.services.tool_catalog.tool_references.BUILTIN_TOOLS",
            {"bash": MagicMock(), "web_search": MagicMock()},
        ):
            with patch(
                "server.services.tool_catalog.tool_references.ensure_builtin_tools_loaded"
            ):
                results = parse_tool_references(["bash", "web_search"])
                assert results == ["bash", "web_search"]


class TestRunMetricsResponseFields:
    """Test RunMetricsResponse fields are correctly typed."""

    def test_all_optional_fields(self):
        """Verify all RunMetricsResponse fields are optional."""
        response = RunMetricsResponse()

        assert response.duration_ms is None
        assert response.input_tokens is None
        assert response.output_tokens is None
        assert response.total_tokens is None
        assert response.cache_read_tokens is None
        assert response.cache_creation_tokens is None
        assert response.token_cost is None
        assert response.steps_count is None
        assert response.tool_calls_count is None

    def test_fields_accept_values(self):
        """Verify fields accept correct value types."""
        response = RunMetricsResponse(
            duration_ms=1234.5,
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            cache_read_tokens=50,
            cache_creation_tokens=25,
            token_cost=0.015,
            steps_count=5,
            tool_calls_count=3,
        )

        assert response.duration_ms == 1234.5
        assert response.input_tokens == 100
