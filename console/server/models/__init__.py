"""API-layer Pydantic models for request/response serialization.

This module consolidates all Console API schemas organized by domain.
"""

from server.models.agent_config import AgentConfigPayload, AgentConfigResponse
from server.models.agent_options import (
    AgentOptionsInput,
    ModelParamsInput,
    sanitize_agent_options_data,
)
from server.models.chat import (
    ChatRequest,
    CreateSessionRequest,
    ForkSessionRequest,
    SwitchSessionRequest,
)
from server.models.metrics import (
    RunMetricsResponse,
    RunMetricsSummary,
    StepMetricsResponse,
)
from server.models.scheduler_control import (
    CancelRequest,
    CreateAgentRequest,
    PendingEventResponse,
    ResumeRequest,
    SchedulerChatCancelRequest,
    SteerRequest,
)
from server.models.scheduler_state import (
    AgentStateListItem,
    AgentStateResponse,
    SchedulerStatsResponse,
    WakeConditionResponse,
)
from server.models.step_run import RunResponse, StepResponse
from server.models.stream import stream_event_to_payload
from server.models.tool_reference import (
    AGENT_TOOL_PREFIX,
    InvalidToolReferenceError,
    parse_tool_reference,
    parse_tool_references,
)
from server.models.trace import SpanResponse, TraceListItem, TraceResponse

__all__ = [
    # tool_reference
    "AGENT_TOOL_PREFIX",
    "InvalidToolReferenceError",
    "parse_tool_reference",
    "parse_tool_references",
    # metrics
    "RunMetricsSummary",
    "RunMetricsResponse",
    "StepMetricsResponse",
    # agent_options
    "AgentOptionsInput",
    "ModelParamsInput",
    "sanitize_agent_options_data",
    # step_run
    "StepResponse",
    "RunResponse",
    # trace
    "SpanResponse",
    "TraceResponse",
    "TraceListItem",
    # agent_config
    "AgentConfigPayload",
    "AgentConfigResponse",
    # scheduler_state
    "WakeConditionResponse",
    "AgentStateResponse",
    "AgentStateListItem",
    "SchedulerStatsResponse",
    # scheduler_control
    "SteerRequest",
    "CancelRequest",
    "ResumeRequest",
    "CreateAgentRequest",
    "SchedulerChatCancelRequest",
    "PendingEventResponse",
    # chat
    "ChatRequest",
    "CreateSessionRequest",
    "SwitchSessionRequest",
    "ForkSessionRequest",
    # stream
    "stream_event_to_payload",
]
