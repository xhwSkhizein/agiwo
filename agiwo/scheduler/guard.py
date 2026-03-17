"""
TaskGuard — Centralized limit enforcement for the Scheduler.

All scheduling limit checks converge here. Tools and Scheduler call TaskGuard
instead of implementing checks themselves.
"""

from agiwo.scheduler.models import ACTIVE_AGENT_STATUSES, AgentState, TaskLimits
from agiwo.scheduler.store import AgentStateStorage
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class TaskGuard:
    """Centralized limit checker — the single entry point for all scheduling limits."""

    def __init__(self, limits: TaskLimits, store: AgentStateStorage) -> None:
        self._limits = limits
        self._store = store

    @property
    def limits(self) -> TaskLimits:
        return self._limits

    async def check_spawn(self, parent_state: AgentState) -> str | None:
        """Check if spawning a child is allowed. Returns None=allowed, str=rejection reason."""
        if parent_state.depth >= self._limits.max_depth:
            reason = (
                f"Max spawn depth ({self._limits.max_depth}) reached. "
                f"Current depth: {parent_state.depth}"
            )
            logger.warning(
                "spawn_rejected_max_depth",
                state_id=parent_state.id,
                depth=parent_state.depth,
                max_depth=self._limits.max_depth,
            )
            return reason

        children = await self._store.list_states(
            parent_id=parent_state.id,
            session_id=parent_state.session_id,
            limit=1000,
        )
        active_children = [
            c
            for c in children
            if c.status in ACTIVE_AGENT_STATUSES
        ]
        if len(active_children) >= self._limits.max_children_per_agent:
            reason = (
                f"Max children per agent ({self._limits.max_children_per_agent}) reached. "
                f"Active children: {len(active_children)}"
            )
            logger.warning(
                "spawn_rejected_max_children",
                state_id=parent_state.id,
                active_children=len(active_children),
                max_children=self._limits.max_children_per_agent,
            )
            return reason

        return None

    async def check_wake(self, state: AgentState) -> str | None:
        """Check if waking an agent is allowed. Returns None=allowed, str=rejection reason."""
        if state.is_root:
            # Root agent can always be woken
            return None
        if state.wake_count >= self._limits.max_wake_count:
            reason = (
                f"Max wake count ({self._limits.max_wake_count}) reached. "
                f"Agent has been woken {state.wake_count} times (possible livelock)."
            )
            logger.warning(
                "wake_rejected_max_count",
                state_id=state.id,
                wake_count=state.wake_count,
                max_wake_count=self._limits.max_wake_count,
            )
            return reason

        return None
