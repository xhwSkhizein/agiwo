"""SessionGateway: unique Console/channel entry for Session → MainAgent.accept (ADR 0049)."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from agiwo.agent import AgentStreamItem, MainAgent
from agiwo.agent.models.input import UserMessage
from agiwo.agent.models.run import RunOutput
from agiwo.utils.logging import get_logger

from server.models.session import ChannelChatSessionStore, Session
from server.services.runtime.session_turn_service import SessionTurnService

if TYPE_CHECKING:
    from server.services.runtime.agent_runtime_cache import AgentRuntimeCache

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SessionTurnResult:
    """Result of submitting a user message on a Session (wait-for-completion path)."""

    kind: Literal["session"]
    session_id: str
    run_id: str | None = None
    status: str = "completed"
    response: str | None = None
    run_output: RunOutput | None = None


@dataclass(frozen=True, slots=True)
class SessionStreamStart:
    """Result of accepting user input without waiting — for live SSE."""

    session_id: str
    run_id: str
    main_agent: MainAgent


class SessionGateway:
    """Write Session history and accept user input on the session MainAgent."""

    def __init__(
        self,
        *,
        session_store: ChannelChatSessionStore,
        agent_runtime_cache: "AgentRuntimeCache",
        session_turn: SessionTurnService,
    ) -> None:
        self._session_store = session_store
        self._agent_runtime_cache = agent_runtime_cache
        self._session_turn = session_turn

    async def start_user_message(
        self,
        session_id: str,
        user_message: UserMessage,
        *,
        idempotency_key: str,
    ) -> SessionStreamStart:
        """Accept input and return a live MainAgent for ``subscribe()`` streaming."""
        del idempotency_key
        session, main_agent = await self._prepare(session_id, user_message)
        run_id = await self._session_turn.accept_user_message(
            main_agent,
            session,
            user_message,
        )
        return SessionStreamStart(
            session_id=session_id,
            run_id=run_id,
            main_agent=main_agent,
        )

    async def stream_user_message(
        self,
        session_id: str,
        user_message: UserMessage,
        *,
        idempotency_key: str,
    ) -> AsyncIterator[AgentStreamItem]:
        """Accept input then yield live ``AgentStreamItem`` events until the run ends."""
        started = await self.start_user_message(
            session_id,
            user_message,
            idempotency_key=idempotency_key,
        )
        async for item in self._session_turn.iter_run_stream(started.main_agent):
            yield item

    async def handle_user_message(
        self,
        session_id: str,
        user_message: UserMessage,
        *,
        idempotency_key: str,
    ) -> SessionTurnResult:
        """Accept input and wait for the run to finish (Feishu / non-streaming clients)."""
        del idempotency_key  # reserved for future command receipts
        session, main_agent = await self._prepare(session_id, user_message)
        run_id, output = await self._session_turn.submit_user_message(
            main_agent,
            session,
            user_message,
        )
        response = output.response if output is not None else None
        return SessionTurnResult(
            kind="session",
            session_id=session_id,
            run_id=run_id,
            status="completed",
            response=response,
            run_output=output,
        )

    async def _prepare(
        self,
        session_id: str,
        user_message: UserMessage,
    ) -> tuple[Session, MainAgent]:
        if not user_message.is_user_provided:
            raise ValueError(
                "SessionGateway requires an is_user_provided=True UserMessage"
            )
        session = await self._session_store.get_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")
        await self._consume_fork_summary_if_needed(session_id, user_message)
        main_agent = await self._agent_runtime_cache.get_or_create_main_agent(session)
        return session, main_agent

    async def submit_user_message(
        self,
        session_id: str,
        user_message: UserMessage,
        *,
        idempotency_key: str,
    ) -> SessionTurnResult:
        """Alias matching the ADR 0048 interface sketch."""
        return await self.handle_user_message(
            session_id,
            user_message,
            idempotency_key=idempotency_key,
        )

    async def _consume_fork_summary_if_needed(
        self,
        session_id: str,
        user_message: UserMessage,
    ) -> None:
        session = await self._session_store.get_session(session_id)
        if session is None or not session.fork_context_summary:
            return
        summary = session.fork_context_summary
        session.fork_context_summary = None
        await self._session_store.upsert_session(session)
        # Fork summary is injected as a system notice on the next root via
        # history; keep the user message unchanged. Summary is stored on
        # session until consumed — clear it so it is one-shot.
        logger.info(
            "session_fork_summary_consumed",
            session_id=session_id,
            summary_chars=len(summary),
            user_chars=len(user_message.extract_text()),
        )


__all__ = [
    "SessionGateway",
    "SessionStreamStart",
    "SessionTurnResult",
]
