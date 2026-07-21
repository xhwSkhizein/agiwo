"""SessionRuntimeService: Session → root Run via Scheduler (ADR 0048)."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import uuid4

from agiwo.agent import Agent, RunOutput, UserInput, UserMessage
from agiwo.agent.models.execution import RunTreeRole, RunExecutionRequest
from agiwo.agent.session_history import append_session_user_message_to_history
from agiwo.scheduler.commands import RouteResult, RouteStreamMode
from agiwo.scheduler.engine import Scheduler
from agiwo.scheduler.execution import SchedulerExecutionRequest
from agiwo.scheduler.models import AgentStateStatus
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

    async def submit_user_message(
        self,
        agent: Agent,
        session: Session,
        user_message: UserMessage,
    ) -> tuple[str, RunOutput]:
        """Session submit: history once, then inject RUNNING root or start a new one."""
        if not user_message.is_user_provided:
            raise ValueError("submit_user_message requires is_user_provided=True")

        await append_session_user_message_to_history(
            agent,
            session_id=session.id,
            user_message=user_message,
            run_id=f"session-history-{session.id}",
        )

        state = await self._scheduler.get_state(session.id)
        if state is not None and state.status == AgentStateStatus.RUNNING:
            handle = self._scheduler._rt.execution_handles.get(session.id)
            if handle is None or not handle.run_id:
                raise RuntimeError(
                    f"Session '{session.id}' is RUNNING but has no live run handle"
                )
            run_id = handle.run_id
            await self._scheduler.inject_user_message(run_id, user_message)
            output = await self._scheduler.wait_for(session.id, timeout=self._timeout)
            await self._touch_session(session)
            return run_id, output

        run_id = f"run_{uuid4().hex}"
        await self._scheduler.dispatch_execution(
            agent,
            SchedulerExecutionRequest(
                state_id=session.id,
                session_id=session.id,
                user_input=None,
                execution=RunExecutionRequest(
                    run_id=run_id,
                    run_tree_role=RunTreeRole.NONE,
                ),
                persistent=True,
            ),
        )
        output = await self._scheduler.wait_for(session.id, timeout=self._timeout)
        await self._touch_session(session)
        return run_id, output

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
