"""SessionTurnService: Session turns via MainAgent.accept (ADR 0049)."""

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timezone

from agiwo.agent import (
    AgentStreamItem,
    MainAgent,
    MainAgentState,
    RunOutput,
    UserMessage,
)
from agiwo.utils.logging import get_logger

from server.models.session import ChannelChatSessionStore, Session

logger = get_logger(__name__)


class SessionTurnService:
    """Accept session user input through MainAgent — not Scheduler root dispatch."""

    def __init__(
        self,
        *,
        session_store: ChannelChatSessionStore,
        timeout: int | None = None,
    ) -> None:
        self._session_store = session_store
        self._timeout = timeout

    async def accept_user_message(
        self,
        main_agent: MainAgent,
        session: Session,
        user_message: UserMessage,
    ) -> str:
        """Accept user input and return the active ``run_id`` without waiting.

        Callers that need live tokens should ``main_agent.subscribe()`` immediately
        after this returns (asyncio will not run the run task until the next await).
        """
        if not user_message.is_user_provided:
            raise ValueError("accept_user_message requires is_user_provided=True")

        handle = await main_agent.accept(user_message)
        if handle is None:
            raise RuntimeError(
                f"MainAgent.accept returned no handle for session {session.id!r}"
            )
        await self._touch_session(session)
        return handle.run_id

    async def submit_user_message(
        self,
        main_agent: MainAgent,
        session: Session,
        user_message: UserMessage,
    ) -> tuple[str, RunOutput]:
        """Accept user input and wait for the active run (channels that need a final reply)."""
        run_id = await self.accept_user_message(main_agent, session, user_message)
        output = await self._wait_current_run(main_agent)
        return run_id, output

    async def iter_run_stream(
        self,
        main_agent: MainAgent,
    ) -> AsyncIterator[AgentStreamItem]:
        """Yield live stream items for the current run until the stream closes."""
        async for item in main_agent.subscribe():
            yield item

    async def cancel_if_active(
        self,
        main_agent: MainAgent,
        session: Session,
        reason: str,
    ) -> None:
        if main_agent.state is not MainAgentState.RUNNING:
            return
        await main_agent.cancel(reason)
        await self._touch_session(session)

    async def wait_until_idle(
        self,
        main_agent: MainAgent,
        *,
        timeout_seconds: float,
    ) -> bool:
        """Poll until MainAgent is IDLE or timeout. Returns True if idle."""
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while asyncio.get_running_loop().time() < deadline:
            if main_agent.state is MainAgentState.IDLE:
                return True
            await asyncio.sleep(0.2)
        return main_agent.state is MainAgentState.IDLE

    async def wait_for(self, main_agent: MainAgent) -> RunOutput:
        return await self._wait_current_run(main_agent)

    async def _wait_current_run(self, main_agent: MainAgent) -> RunOutput:
        coro = main_agent.wait_current_run()
        if self._timeout is not None:
            return await asyncio.wait_for(coro, timeout=self._timeout)
        return await coro

    async def _touch_session(self, session: Session) -> None:
        session.updated_at = datetime.now(timezone.utc)
        await self._session_store.upsert_session(session)
