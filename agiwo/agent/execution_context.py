from dataclasses import dataclass, field

from agiwo.agent.stream_channel import StreamChannel


@dataclass
class ExecutionContext:
    # Execution identity
    session_id: str
    run_id: str
    # Session-level resources
    channel: StreamChannel
    # User Context
    user_id: str | None = None

    # Hierarchy information
    depth: int = 0
    parent_run_id: str | None = None
    agent_id: str | None = None

    # Observability
    trace_id: str | None = None
    parent_span_id: str | None = None
    span_id: str | None = None

    # Timeout control
    timeout_at: float | None = None

    # Metadata
    metadata: dict = field(default_factory=dict)

    def new_child(self, run_id: str, agent_id: str | None = None) -> "ExecutionContext":
        return ExecutionContext(
            session_id=self.session_id,
            run_id=run_id,
            channel=self.channel,
            user_id=self.user_id,
            depth=self.depth + 1,
            parent_run_id=self.run_id,
            agent_id=agent_id or self.agent_id,
            trace_id=self.trace_id,
            parent_span_id=self.span_id,
            span_id=None,
            timeout_at=self.timeout_at,
            metadata=dict(self.metadata),
        )
