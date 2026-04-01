"""Unified runtime orchestration for console chat and channels."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import uuid4

from agiwo.agent import Agent, RunOutput, UserInput
from agiwo.scheduler.commands import RouteResult
from agiwo.scheduler.engine import Scheduler
from agiwo.utils.logging import get_logger

from server.models.session import (
    ChannelChatSessionStore,
    Session,
    append_task_message,
    bind_scheduler_state,
    start_task,
)

logger = get_logger(__name__)

if TYPE_CHECKING:
    from agiwo.scheduler.models import AgentState


class SessionRuntimeService:
    """Route session input into the SDK scheduler while preserving session state."""

    def __init__(
        self,
        *,
        scheduler: Scheduler,
        session_store: ChannelChatSessionStore,
        timeout: int | None = None,
    ) -> None:
        self._scheduler = scheduler
        self._session_store = session_store
        self._timeout = timeout

    async def execute(
        self,
        agent: Agent,
        session: Session,
        user_input: UserInput,
    ) -> RouteResult:
        if session.current_task_id is None:
            start_task(session, task_id=str(uuid4()))
        append_task_message(session)

        result = await self._scheduler.route_root_input(
            user_input,
            agent=agent,
            state_id=session.scheduler_state_id or None,
            session_id=session.id,
            persistent=True,
            timeout=self._timeout,
        )
        if result.state_id != session.scheduler_state_id:
            bind_scheduler_state(session, result.state_id)
        await self._touch_session(session)
        return result

    async def cancel_if_active(self, session: Session, reason: str) -> None:
        if not session.scheduler_state_id:
            return
        state = await self._scheduler.get_state(session.scheduler_state_id)
        if state is None or not state.is_active():
            return
        await self._scheduler.cancel(session.scheduler_state_id, reason)

    async def get_state(self, state_id: str | None) -> "AgentState | None":
        if not state_id:
            return None
        return await self._scheduler.get_state(state_id)

    async def wait_for(self, state_id: str) -> RunOutput:
        return await self._scheduler.wait_for(state_id, timeout=None)

    async def _touch_session(self, session: Session) -> None:
        session.updated_at = datetime.now(timezone.utc)
        await self._session_store.upsert_session(session)
