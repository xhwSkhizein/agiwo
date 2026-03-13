"""
Scheduler submission and state-transition bridge for channel sessions.
"""

from collections.abc import AsyncIterator
from datetime import datetime, timezone

from agiwo.agent import Agent, UserInput
from agiwo.scheduler.models import AgentStateStatus, SchedulerOutput
from agiwo.scheduler.scheduler import Scheduler
from agiwo.utils.logging import get_logger

from server.channels.session_binding import assign_scheduler_state
from server.channels.models import ChannelChatSessionStore, Session

logger = get_logger(__name__)


class _EmptySchedulerOutputStream:
    def __aiter__(self) -> "_EmptySchedulerOutputStream":
        return self

    async def __anext__(self) -> SchedulerOutput:
        raise StopAsyncIteration


def _empty_scheduler_output_stream() -> AsyncIterator[SchedulerOutput]:
    return _EmptySchedulerOutputStream()


class SchedulerSessionBridge:
    def __init__(
        self,
        *,
        scheduler: Scheduler,
        store: ChannelChatSessionStore,
        scheduler_wait_timeout: int,
    ) -> None:
        self._scheduler = scheduler
        self._store = store
        self._scheduler_wait_timeout = scheduler_wait_timeout

    async def cancel_session_state_if_active(
        self,
        session: Session,
        reason: str,
    ) -> None:
        if not session.scheduler_state_id:
            return
        state = await self._scheduler.get_state(session.scheduler_state_id)
        if state is None:
            return
        if state.status not in (
            AgentStateStatus.RUNNING,
            AgentStateStatus.SLEEPING,
            AgentStateStatus.PENDING,
        ):
            return
        await self._scheduler.cancel(session.scheduler_state_id, reason)

    async def submit_to_scheduler(
        self,
        agent: Agent,
        session: Session,
        user_input: UserInput,
    ) -> AsyncIterator[SchedulerOutput]:
        current_state = None
        if session.scheduler_state_id:
            current_state = await self._scheduler.get_state(session.scheduler_state_id)

        if current_state is None or current_state.status in (
            AgentStateStatus.COMPLETED,
            AgentStateStatus.FAILED,
        ):
            output_stream = self._scheduler.submit_and_subscribe(
                agent,
                user_input,
                session_id=session.id,
                persistent=True,
                timeout=self._scheduler_wait_timeout,
            )
            assign_scheduler_state(session, agent.id)
        elif current_state.status == AgentStateStatus.SLEEPING:
            output_stream = self._scheduler.submit_task_and_subscribe(
                session.scheduler_state_id,
                user_input,
                agent=agent,
                timeout=self._scheduler_wait_timeout,
            )
        elif current_state.status == AgentStateStatus.RUNNING:
            output_stream = await self._handle_running_state(session, user_input)
        elif current_state.status == AgentStateStatus.PENDING:
            output_stream = await self._handle_pending_state(agent, session, user_input)
        else:
            output_stream = self._scheduler.submit_and_subscribe(
                agent,
                user_input,
                session_id=session.id,
                persistent=True,
                timeout=self._scheduler_wait_timeout,
            )
            assign_scheduler_state(session, agent.id)

        await self._touch_session(session)
        return output_stream

    async def _handle_running_state(
        self,
        session: Session,
        user_input: UserInput,
    ) -> AsyncIterator[SchedulerOutput]:
        steered = await self._scheduler.steer(session.scheduler_state_id, user_input)
        if steered:
            logger.info(
                "user_input_steered_to_running_agent",
                state_id=session.scheduler_state_id,
            )
        else:
            logger.warning(
                "steer_failed_agent_not_in_process",
                state_id=session.scheduler_state_id,
            )
        return _empty_scheduler_output_stream()

    async def _handle_pending_state(
        self,
        agent: Agent,
        session: Session,
        user_input: UserInput,
    ) -> AsyncIterator[SchedulerOutput]:
        logger.info(
            "waiting_for_pending_agent_before_submit",
            state_id=session.scheduler_state_id,
        )
        await self._scheduler.wait_for(
            session.scheduler_state_id,
            timeout=self._scheduler_wait_timeout,
        )
        refreshed = await self._scheduler.get_state(session.scheduler_state_id)

        if refreshed is not None and refreshed.status == AgentStateStatus.SLEEPING:
            return self._scheduler.submit_task_and_subscribe(
                session.scheduler_state_id,
                user_input,
                agent=agent,
                timeout=self._scheduler_wait_timeout,
            )

        assign_scheduler_state(session, agent.id)
        return self._scheduler.submit_and_subscribe(
            agent,
            user_input,
            session_id=session.id,
            persistent=True,
            timeout=self._scheduler_wait_timeout,
        )

    async def _touch_session(self, session: Session) -> None:
        session.updated_at = datetime.now(timezone.utc)
        await self._store.upsert_session(session)
