"""
Event-driven agent execution bridge for channel sessions.

Handles the full lifecycle of submitting user input to the scheduler
and streaming typed agent events back to the channel with timeout protection.

State resolution logic:
    None / COMPLETED / FAILED  →  submit new persistent agent
    IDLE / FAILED (persistent) →  enqueue input to existing agent
    RUNNING / WAITING / QUEUED →  steer (inject message, no output)
    PENDING                    →  wait then submit/enqueue
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from agiwo.agent import Agent, UserInput
from agiwo.agent.runtime import AgentStreamItem, RunOutput
from agiwo.scheduler.models import AgentState, AgentStateStatus
from agiwo.scheduler.scheduler import Scheduler
from agiwo.utils.logging import get_logger

from server.channels.session.binding import assign_scheduler_state
from server.channels.session.models import ChannelChatSessionStore, Session

logger = get_logger(__name__)


@dataclass(frozen=True)
class DispatchResult:
    """Result of routing a channel message into the scheduler."""

    action: Literal["submitted", "enqueued", "steered"]
    stream: AsyncIterator[AgentStreamItem] | None = None


class AgentExecutor:
    """Submit user input to the scheduler and return dispatch metadata."""

    def __init__(
        self,
        *,
        scheduler: Scheduler,
        store: ChannelChatSessionStore,
        timeout: int | None = None,
    ) -> None:
        self._scheduler = scheduler
        self._store = store
        self._timeout = timeout

    async def execute(
        self,
        agent: Agent,
        session: Session,
        user_input: UserInput,
    ) -> DispatchResult:
        """Resolve scheduler state, submit/enqueue/steer, and return dispatch info."""
        state_id = session.scheduler_state_id
        current_state = None
        if state_id:
            current_state = await self._scheduler.get_state(state_id)

        status = current_state.status if current_state is not None else None

        if status in (
            AgentStateStatus.RUNNING,
            AgentStateStatus.WAITING,
            AgentStateStatus.QUEUED,
        ):
            steered = await self._steer(
                session, user_input, urgent=status == AgentStateStatus.WAITING
            )
            if steered:
                return DispatchResult(action="steered")
            refreshed = await self._scheduler.get_state(session.scheduler_state_id)
            refreshed_status = refreshed.status if refreshed is not None else None
            if refreshed_status in (
                AgentStateStatus.RUNNING,
                AgentStateStatus.WAITING,
                AgentStateStatus.QUEUED,
            ):
                raise RuntimeError(
                    f"Failed to steer active scheduler state '{session.scheduler_state_id}'"
                )
            return await self.execute(agent, session, user_input)

        if status == AgentStateStatus.PENDING:
            return await self._handle_pending(agent, session, user_input)

        use_enqueue = (
            current_state is not None
            and current_state.is_root
            and current_state.is_persistent
            and status in (AgentStateStatus.IDLE, AgentStateStatus.FAILED)
        )

        if use_enqueue:
            return await self._enqueue_and_stream(agent, session, user_input)
        return await self._submit_and_stream(agent, session, user_input)

    async def cancel_if_active(self, session: Session, reason: str) -> None:
        """Cancel the scheduler state for a session if it is still active."""
        if not session.scheduler_state_id:
            return
        state = await self._scheduler.get_state(session.scheduler_state_id)
        if state is None or not state.is_active():
            return
        await self._scheduler.cancel(session.scheduler_state_id, reason)

    async def get_state(self, state_id: str | None) -> AgentState | None:
        """Fetch the latest scheduler state for a session-owned root."""
        if not state_id:
            return None
        return await self._scheduler.get_state(state_id)

    async def wait_for(self, state_id: str) -> RunOutput:
        """Wait until a root state settles to IDLE/COMPLETED/FAILED."""
        return await self._scheduler.wait_for(state_id, timeout=None)

    # -- Private: action handlers ----------------------------------------------

    async def _steer(
        self,
        session: Session,
        user_input: UserInput,
        *,
        urgent: bool,
    ) -> bool:
        steered = await self._scheduler.steer(
            session.scheduler_state_id,
            user_input,
            urgent=urgent,
        )
        if steered:
            logger.info("user_input_steered", state_id=session.scheduler_state_id)
        else:
            logger.warning("steer_failed", state_id=session.scheduler_state_id)
        await self._touch_session(session)
        return steered

    async def _submit_and_stream(
        self,
        agent: Agent,
        session: Session,
        user_input: UserInput,
    ) -> DispatchResult:
        assign_scheduler_state(session, agent.id)
        await self._touch_session(session)

        stream = self._scheduler.stream(
            user_input,
            agent=agent,
            session_id=session.id,
            persistent=True,
            timeout=self._timeout,
        )
        return DispatchResult(action="submitted", stream=stream)

    async def _enqueue_and_stream(
        self,
        agent: Agent,
        session: Session,
        user_input: UserInput,
    ) -> DispatchResult:
        await self._touch_session(session)

        stream = self._scheduler.stream(
            user_input,
            agent=agent,
            state_id=session.scheduler_state_id,
            timeout=self._timeout,
        )
        return DispatchResult(action="enqueued", stream=stream)

    async def _handle_pending(
        self,
        agent: Agent,
        session: Session,
        user_input: UserInput,
    ) -> DispatchResult:
        logger.info(
            "waiting_for_pending_before_submit",
            state_id=session.scheduler_state_id,
        )
        await self._scheduler.wait_for(
            session.scheduler_state_id,
            timeout=self._timeout,
        )
        refreshed = await self._scheduler.get_state(session.scheduler_state_id)

        can_enqueue = (
            refreshed is not None
            and refreshed.is_root
            and refreshed.is_persistent
            and refreshed.status in (AgentStateStatus.IDLE, AgentStateStatus.FAILED)
        )
        if can_enqueue:
            return await self._enqueue_and_stream(agent, session, user_input)
        return await self._submit_and_stream(agent, session, user_input)

    async def _touch_session(self, session: Session) -> None:
        session.updated_at = datetime.now(timezone.utc)
        await self._store.upsert_session(session)
