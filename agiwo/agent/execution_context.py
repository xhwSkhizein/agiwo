from dataclasses import dataclass, field

from agiwo.agent.wire import Wire


@dataclass(frozen=True)
class ExecutionContext:
    # Execution identity
    session_id: str
    run_id: str
    # Session-level resources
    wire: Wire
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
