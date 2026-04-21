"""Unified runtime orchestration for console chat and channels."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from agiwo.agent import Agent, RunOutput, UserInput
from agiwo.scheduler.commands import RouteResult, RouteStreamMode
from agiwo.scheduler.engine import Scheduler
from agiwo.utils.logging import get_logger

from server.models.session import ChannelChatSessionStore, Session

logger = get_logger(__name__)

if TYPE_CHECKING:
    from agiwo.scheduler.models import AgentState


class SessionRuntimeService:
    """Route session input into the SDK scheduler using session-scoped root states."""

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
        *,
        stream_mode: RouteStreamMode = RouteStreamMode.UNTIL_SETTLED,
    ) -> RouteResult:
        result = await self._scheduler.route_root_input(
            user_input,
            agent=agent,
            state_id=session.id,
            session_id=session.id,
            persistent=True,
            timeout=self._timeout,
            stream_mode=stream_mode,
        )
        await self._touch_session(session)
        return result

    async def cancel_if_active(self, session: Session, reason: str) -> None:
        state = await self._scheduler.get_state(session.id)
        if state is None or not state.is_active():
            return
        await self._scheduler.cancel(session.id, reason)

    async def get_state(self, state_id: str | None) -> "AgentState | None":
        if not state_id:
            return None
        return await self._scheduler.get_state(state_id)

    async def wait_for(self, state_id: str) -> RunOutput:
        return await self._scheduler.wait_for(state_id, timeout=None)

    async def _touch_session(self, session: Session) -> None:
        session.updated_at = datetime.now(timezone.utc)
        await self._session_store.upsert_session(session)
