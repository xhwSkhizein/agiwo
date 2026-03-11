"""Console domain models shared by services and API adapters."""

from server.domain.agent_configs import AgentConfigInput
from server.domain.run_metrics import RunMetricsSummary
from server.domain.scheduler_events import (
    SchedulerCompletedEventPayloadData,
    SchedulerFailedEventPayloadData,
    scheduler_completed_payload,
    scheduler_failed_payload,
)
from server.domain.sessions import (
    SessionAggregate,
    SessionSummaryData,
    session_aggregate_to_chat_summary,
    session_aggregate_to_summary_data,
)
from server.domain.tool_references import (
    AGENT_TOOL_PREFIX,
    AgentToolRef,
    BuiltinToolRef,
    InvalidToolReferenceError,
    ToolReference,
    parse_tool_reference,
    parse_tool_references,
    serialize_tool_references,
)

__all__ = [
    "AgentConfigInput",
    "RunMetricsSummary",
    "SchedulerCompletedEventPayloadData",
    "SchedulerFailedEventPayloadData",
    "SessionAggregate",
    "SessionSummaryData",
    "AGENT_TOOL_PREFIX",
    "AgentToolRef",
    "BuiltinToolRef",
    "InvalidToolReferenceError",
    "ToolReference",
    "parse_tool_reference",
    "parse_tool_references",
    "serialize_tool_references",
    "scheduler_completed_payload",
    "scheduler_failed_payload",
    "session_aggregate_to_chat_summary",
    "session_aggregate_to_summary_data",
]
