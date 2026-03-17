"""
Scheduler submission and state-transition bridge for channel sessions.
"""

from collections.abc import AsyncIterator
from datetime import datetime, timezone

from agiwo.agent import Agent, UserInput
from agiwo.agent.runtime import AgentStreamItem, RunCompletedEvent, RunFailedEvent
from agiwo.scheduler.models import AgentStateStatus
from agiwo.scheduler.scheduler import Scheduler
from agiwo.utils.logging import get_logger

from server.channels.models import ChannelChatSessionStore, Session
from server.channels.session_binding import assign_scheduler_state

logger = get_logger(__name__)


class _EmptySchedulerTextStream:
    def __aiter__(self) -> "_EmptySchedulerTextStream":
        return self

    async def __anext__(self) -> str:
        raise StopAsyncIteration


def _empty_scheduler_output_stream() -> AsyncIterator[str]:
    return _EmptySchedulerTextStream()


def _can_accept_enqueue_input(state: object | None) -> bool:
    if state is None:
        return False
    helper = getattr(state, "can_accept_enqueue_input", None)
    if callable(helper):
        return bool(helper())
    status = getattr(state, "status", None)
    is_persistent = bool(getattr(state, "is_persistent", False))
    parent_id = getattr(state, "parent_id", None)
    return (
        parent_id is None
        and is_persistent
        and status in (AgentStateStatus.IDLE, AgentStateStatus.FAILED)
    )


async def _text_stream_from_scheduler_events(
    event_stream: AsyncIterator[AgentStreamItem],
) -> AsyncIterator[str]:
    async for item in event_stream:
        if isinstance(item, RunCompletedEvent):
            if not item.response:
                continue
            if item.depth == 0:
                yield item.response
                continue
            yield (
                f"<notice>agent_id={item.agent_id}, status=completed</notice>\n"
                f"{item.response}"
            )
        elif isinstance(item, RunFailedEvent):
            if item.depth == 0:
                yield item.error
                continue
            yield (
                f"<notice>agent_id={item.agent_id}, status=failed</notice>\n"
                f"{item.error}"
            )


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
        if not state.is_active():
            return
        await self._scheduler.cancel(session.scheduler_state_id, reason)

    async def submit_to_scheduler(
        self,
        agent: Agent,
        session: Session,
        user_input: UserInput,
    ) -> AsyncIterator[str]:
        current_state = None
        if session.scheduler_state_id:
            current_state = await self._scheduler.get_state(session.scheduler_state_id)

        if current_state is None or current_state.status in (
            AgentStateStatus.COMPLETED,
            AgentStateStatus.FAILED,
        ):
            output_stream = _text_stream_from_scheduler_events(
                self._scheduler.stream(
                    user_input,
                    agent=agent,
                    session_id=session.id,
                    persistent=True,
                    timeout=self._scheduler_wait_timeout,
                )
            )
            assign_scheduler_state(session, agent.id)
        elif _can_accept_enqueue_input(current_state):
            output_stream = _text_stream_from_scheduler_events(
                self._scheduler.stream(
                    user_input,
                    agent=agent,
                    state_id=session.scheduler_state_id,
                    timeout=self._scheduler_wait_timeout,
                )
            )
        elif current_state.status in (
            AgentStateStatus.RUNNING,
            AgentStateStatus.WAITING,
            AgentStateStatus.QUEUED,
        ):
            output_stream = await self._handle_running_state(
                session,
                user_input,
                urgent=current_state.status == AgentStateStatus.WAITING,
            )
        elif current_state.status == AgentStateStatus.PENDING:
            output_stream = await self._handle_pending_state(agent, session, user_input)
        else:
            output_stream = _text_stream_from_scheduler_events(
                self._scheduler.stream(
                    user_input,
                    agent=agent,
                    session_id=session.id,
                    persistent=True,
                    timeout=self._scheduler_wait_timeout,
                )
            )
            assign_scheduler_state(session, agent.id)

        await self._touch_session(session)
        return output_stream

    async def _handle_running_state(
        self,
        session: Session,
        user_input: UserInput,
        *,
        urgent: bool = False,
    ) -> AsyncIterator[str]:
        steered = await self._scheduler.steer(
            session.scheduler_state_id,
            user_input,
            urgent=urgent,
        )
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
    ) -> AsyncIterator[str]:
        logger.info(
            "waiting_for_pending_agent_before_submit",
            state_id=session.scheduler_state_id,
        )
        await self._scheduler.wait_for(
            session.scheduler_state_id,
            timeout=self._scheduler_wait_timeout,
        )
        refreshed = await self._scheduler.get_state(session.scheduler_state_id)

        if _can_accept_enqueue_input(refreshed):
            return _text_stream_from_scheduler_events(
                self._scheduler.stream(
                    user_input,
                    agent=agent,
                    state_id=session.scheduler_state_id,
                    timeout=self._scheduler_wait_timeout,
                )
            )

        assign_scheduler_state(session, agent.id)
        return _text_stream_from_scheduler_events(
            self._scheduler.stream(
                user_input,
                agent=agent,
                session_id=session.id,
                persistent=True,
                timeout=self._scheduler_wait_timeout,
            )
        )

    async def _touch_session(self, session: Session) -> None:
        session.updated_at = datetime.now(timezone.utc)
        await self._store.upsert_session(session)
