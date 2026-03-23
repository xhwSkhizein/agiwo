"""Event-driven agent execution bridge for channel sessions."""

from datetime import datetime, timezone

from agiwo.agent import Agent, UserInput
from agiwo.agent.runtime import RunOutput
from agiwo.scheduler.commands import RouteResult
from agiwo.scheduler.models import AgentState
from agiwo.scheduler.scheduler import Scheduler
from agiwo.utils.logging import get_logger

from server.channels.session.binding import assign_scheduler_state
from server.channels.session.models import ChannelChatSessionStore, Session

logger = get_logger(__name__)


class AgentExecutor:
    """Route channel input into the scheduler and return typed dispatch metadata."""

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
    ) -> RouteResult:
        result = await self._scheduler.route_root_input(
            user_input,
            agent=agent,
            state_id=session.scheduler_state_id or None,
            session_id=session.id,
            persistent=True,
            timeout=self._timeout,
        )
        if result.state_id != session.scheduler_state_id:
            assign_scheduler_state(session, result.state_id)
        await self._touch_session(session)
        return result

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

    async def _touch_session(self, session: Session) -> None:
        session.updated_at = datetime.now(timezone.utc)
        await self._store.upsert_session(session)
