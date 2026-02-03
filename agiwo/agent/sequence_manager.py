"""
Sequence Manager for Agent execution.

Manages sequence allocation for all Steps within a Session.
"""

from agiwo.agent.session.base import SessionStore
from agiwo.agent.execution_context import ExecutionContext


class SequenceManager:
    """Session-level sequence allocation service.

    Manages sequence allocation for all Steps within a Session:
    1. Normal allocation: atomic allocation via SessionStore.allocate_sequence
    2. Pre-allocation handling: seq_start mechanism for parallel execution

    This is a Session-level resource that should be shared across all
    nested Agent executions within the same Session.
    """

    def __init__(self, session_store: SessionStore) -> None:
        """Initialize SequenceManager.

        Args:
            session_store: SessionStore instance for atomic sequence allocation
        """
        self.session_store = session_store

    async def allocate(
        self, session_id: str, context: ExecutionContext | None = None
    ) -> int:
        """Allocate next sequence number.

        Args:
            session_id: Session ID
            context: ExecutionContext (optional). If provided and contains
                    seq_start in metadata, uses the pre-allocated sequence
                    for parallel execution branches.

        Returns:
            Next sequence number
        """
        # Handle parallel execution pre-allocated sequences
        # Pre-allocates sequence ranges before execution,
        # passing them via context.metadata
        if context and "seq_start" in context.metadata:
            seq_start = context.metadata.pop("seq_start")
            return seq_start

        # Use SessionStore's atomic allocation
        return await self.session_store.allocate_sequence(session_id)
