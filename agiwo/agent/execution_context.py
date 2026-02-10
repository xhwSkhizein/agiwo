import asyncio
from dataclasses import dataclass, field

from agiwo.agent.stream_channel import StreamChannel


class SessionSequenceCounter:
    """Session-level thread-safe sequence counter shared across all agents in a session."""

    def __init__(self, initial: int = 0) -> None:
        self._counter = initial
        self._lock = asyncio.Lock()

    async def next(self) -> int:
        async with self._lock:
            seq = self._counter
            self._counter += 1
            return seq

    @property
    def current(self) -> int:
        return self._counter


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

    # Session-level sequence counter (shared across nested agents)
    sequence_counter: SessionSequenceCounter | None = None

    # Observability
    trace_id: str | None = None

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
            sequence_counter=self.sequence_counter,
            trace_id=self.trace_id,
            timeout_at=self.timeout_at,
            metadata=dict(self.metadata),
        )
