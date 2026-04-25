"""Console session and channel runtime models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

from agiwo.agent import UserInput, UserMessage

from server.models.metrics import RunMetricsSummary

if TYPE_CHECKING:
    from agiwo.observability.trace import Trace
    from agiwo.scheduler.models import AgentState

T = TypeVar("T")


@dataclass(slots=True)
class Attachment:
    type: str
    key: str
    name: str = ""


@dataclass(slots=True)
class InboundMessage:
    channel_instance_id: str
    event_id: str
    message_id: str
    chat_id: str
    chat_type: str
    sender_id: str
    sender_name: str
    text: str
    event_time_ms: int
    raw_payload: dict[str, Any]
    message_type: str = "text"
    thread_id: str | None = None
    mentions: list[str] = field(default_factory=list)
    is_at_bot: bool = False
    attachments: list[Attachment] = field(default_factory=list)


@dataclass(slots=True)
class ChannelChatContext:
    scope_id: str
    channel_instance_id: str
    chat_id: str
    chat_type: str
    user_open_id: str
    base_agent_id: str
    current_session_id: str
    created_at: datetime
    updated_at: datetime


@dataclass(slots=True)
class Session:
    id: str
    chat_context_scope_id: str | None
    base_agent_id: str
    created_by: str
    created_at: datetime
    updated_at: datetime
    source_session_id: str | None = None
    fork_context_summary: str | None = None


@dataclass(slots=True)
class SessionWithContext:
    session: Session
    chat_context: ChannelChatContext


@dataclass(slots=True)
class SessionSwitchResult:
    previous_session: Session | None
    current_session: Session
    chat_context: ChannelChatContext


@dataclass(slots=True)
class UserSessionItem:
    session: Session
    chat_context: ChannelChatContext
    is_current: bool
    in_current_context: bool


@dataclass(slots=True)
class SessionCreateResult:
    chat_context: ChannelChatContext | None
    session: Session


@dataclass(slots=True)
class SessionSummaryRecord:
    session_id: str
    base_agent_id: str | None = None
    last_user_input: UserInput | None = None
    last_response: str | None = None
    run_count: int = 0
    step_count: int = 0
    metrics: RunMetricsSummary = field(default_factory=RunMetricsSummary)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    chat_context_scope_id: str | None = None
    created_by: str | None = None
    root_state_status: str | None = None
    source_session_id: str | None = None
    fork_context_summary: str | None = None


@dataclass(slots=True)
class SessionDetailRecord:
    summary: SessionSummaryRecord
    session: Session | None = None
    chat_context: ChannelChatContext | None = None
    scheduler_state: "AgentState | None" = None
    observability: "SessionObservabilityRecord | None" = None
    milestone_board: "SessionMilestoneBoardRecord | None" = None
    review_cycles: list["ReviewCycleRecord"] = field(default_factory=list)
    conversation_events: list["ConversationEventRecord"] = field(default_factory=list)


@dataclass(slots=True)
class MilestoneRecord:
    id: str
    description: str
    status: str
    declared_at_seq: int | None = None
    completed_at_seq: int | None = None


@dataclass(slots=True)
class ReviewCheckpointRecord:
    seq: int
    milestone_id: str
    confirmed_at: datetime


@dataclass(slots=True)
class ReviewOutcomeRecord:
    aligned: bool | None = None
    experience: str | None = None
    step_back_applied: bool = False
    affected_count: int | None = None
    trigger_reason: str | None = None
    active_milestone: str | None = None
    resolved_at: datetime | None = None


@dataclass(slots=True)
class SessionMilestoneBoardRecord:
    session_id: str
    run_id: str | None
    milestones: list[MilestoneRecord] = field(default_factory=list)
    active_milestone_id: str | None = None
    latest_checkpoint: ReviewCheckpointRecord | None = None
    latest_review_outcome: ReviewOutcomeRecord | None = None
    pending_review_reason: str | None = None


@dataclass(slots=True)
class ReviewCycleRecord:
    cycle_id: str
    run_id: str
    agent_id: str
    trigger_reason: str
    steps_since_last_review: int | None = None
    active_milestone: str | None = None
    active_milestone_id: str | None = None
    hook_advice: str | None = None
    aligned: bool | None = None
    experience: str | None = None
    step_back_applied: bool = False
    rollback_range: tuple[int, int] | None = None
    affected_count: int | None = None
    started_at: datetime | None = None
    resolved_at: datetime | None = None
    raw_notice: str | None = None


@dataclass(slots=True)
class ConversationEventRecord:
    id: str
    session_id: str
    run_id: str | None
    sequence: int | None
    kind: str
    priority: str
    title: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeDecisionRecord:
    kind: str
    sequence: int
    run_id: str
    agent_id: str
    created_at: datetime
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SessionObservabilityRecord:
    recent_traces: list["Trace"] = field(default_factory=list)
    decision_events: list[RuntimeDecisionRecord] = field(default_factory=list)


@dataclass(slots=True)
class TraceTimelineEventRecord:
    kind: str
    timestamp: datetime | None = None
    sequence: int | None = None
    run_id: str | None = None
    agent_id: str | None = None
    span_id: str | None = None
    step_id: str | None = None
    title: str = ""
    summary: str = ""
    status: str = "ok"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TraceMainlineEventRecord:
    id: str
    kind: str
    title: str
    summary: str
    status: str = "ok"
    sequence: int | None = None
    timestamp: datetime | None = None
    run_id: str | None = None
    agent_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TraceLlmCallRecord:
    span_id: str
    run_id: str
    agent_id: str
    model: str | None
    provider: str | None
    finish_reason: str | None
    duration_ms: float | None
    first_token_latency_ms: float | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    message_count: int
    tool_schema_count: int
    response_tool_call_count: int
    output_preview: str | None


@dataclass(slots=True)
class PageSlice(Generic[T]):
    items: list[T]
    limit: int
    offset: int
    has_more: bool
    total: int | None = None


@dataclass(slots=True)
class BatchContext:
    chat_context_scope_id: str
    channel_instance_id: str
    chat_id: str
    chat_type: str
    trigger_user_id: str
    trigger_message_id: str
    base_agent_id: str


@dataclass(slots=True)
class BatchPayload:
    context: BatchContext
    messages: list[InboundMessage]
    user_message: UserMessage


class ChannelChatSessionStore(Protocol):
    async def get_chat_context(self, scope_id: str) -> ChannelChatContext | None: ...
    async def upsert_chat_context(self, chat_context: ChannelChatContext) -> None: ...
    async def get_session(self, session_id: str) -> Session | None: ...
    async def get_session_with_context(
        self,
        session_id: str,
    ) -> SessionWithContext | None: ...
    async def upsert_session(self, session: Session) -> None: ...
    async def delete_session(self, session_id: str) -> bool: ...
    async def list_sessions_by_user(
        self, user_open_id: str
    ) -> list[SessionWithContext]: ...
    async def list_sessions_by_chat_context(
        self, chat_context_scope_id: str
    ) -> list[Session]: ...
    async def list_sessions_by_base_agent(
        self, base_agent_id: str
    ) -> list[Session]: ...
    async def list_sessions(self) -> list[Session]: ...
