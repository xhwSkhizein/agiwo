"""SessionGateway: unique Console/channel entry for Session → root Run (ADR 0048)."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from agiwo.agent.models.input import UserMessage
from agiwo.agent.models.run import RunOutput
from agiwo.utils.logging import get_logger

from server.models.session import ChannelChatSessionStore
from server.services.runtime.session_runtime_service import SessionRuntimeService

if TYPE_CHECKING:
    from server.services.runtime.agent_runtime_cache import AgentRuntimeCache

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SessionTurnResult:
    """Result of submitting a user message on a Session."""

    kind: Literal["session"]
    session_id: str
    run_id: str | None = None
    status: str = "completed"
    response: str | None = None
    run_output: RunOutput | None = None


class SessionGateway:
    """Write Session history and start or inject a root Run."""

    def __init__(
        self,
        *,
        session_store: ChannelChatSessionStore,
        agent_runtime_cache: "AgentRuntimeCache",
        session_runtime: SessionRuntimeService,
    ) -> None:
        self._session_store = session_store
        self._agent_runtime_cache = agent_runtime_cache
        self._session_runtime = session_runtime

    async def handle_user_message(
        self,
        session_id: str,
        user_message: UserMessage,
        *,
        idempotency_key: str,
    ) -> SessionTurnResult:
        del idempotency_key  # reserved for future command receipts
        if not user_message.is_user_provided:
            raise ValueError(
                "SessionGateway.handle_user_message requires an "
                "is_user_provided=True UserMessage"
            )

        session = await self._session_store.get_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")

        await self._consume_fork_summary_if_needed(session_id, user_message)

        agent = await self._agent_runtime_cache.get_or_create_runtime_agent(session)
        run_id, output = await self._session_runtime.execute_plain_turn(
            agent,
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


# Back-compat alias during rename window (callers should use SessionGateway).
SessionObjectiveGateway = SessionGateway

__all__ = [
    "SessionGateway",
    "SessionObjectiveGateway",
    "SessionTurnResult",
]
