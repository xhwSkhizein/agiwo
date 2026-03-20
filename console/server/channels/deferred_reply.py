"""Deferred channel reply delivery for active scheduler roots."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from agiwo.utils.logging import get_logger

from server.channels.agent_executor import AgentExecutor
from server.channels.session import SessionContextService
from server.channels.session.models import BatchContext, Session

logger = get_logger(__name__)

DeliverChunked = Callable[[BatchContext, str], Awaitable[None]]


@dataclass
class _PendingDeferredReply:
    task: asyncio.Task[None]
    session_id: str
    state_id: str
    context: BatchContext


class DeferredReplyManager:
    """Wait for an active root state to settle, then emit one follow-up message."""

    def __init__(
        self,
        *,
        executor: AgentExecutor,
        session_service: SessionContextService,
        deliver_chunked: DeliverChunked,
    ) -> None:
        self._executor = executor
        self._session_service = session_service
        self._deliver_chunked = deliver_chunked
        self._pending_by_scope: dict[str, _PendingDeferredReply] = {}

    async def close(self) -> None:
        tasks = [pending.task for pending in self._pending_by_scope.values()]
        self._pending_by_scope.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def arm(
        self,
        *,
        session: Session,
        context: BatchContext,
    ) -> None:
        state_id = session.scheduler_state_id
        if not state_id:
            return

        scope_id = context.chat_context_scope_id
        existing = self._pending_by_scope.get(scope_id)
        if (
            existing is not None
            and existing.session_id == session.id
            and existing.state_id == state_id
            and not existing.task.done()
        ):
            existing.context = context
            logger.info(
                "deferred_reply_updated",
                scope_id=scope_id,
                session_id=session.id,
                state_id=state_id,
            )
            return

        if existing is not None:
            existing.task.cancel()

        task = asyncio.create_task(
            self._wait_and_deliver(
                scope_id=scope_id,
                session_id=session.id,
                state_id=state_id,
            )
        )
        self._pending_by_scope[scope_id] = _PendingDeferredReply(
            task=task,
            session_id=session.id,
            state_id=state_id,
            context=context,
        )
        logger.info(
            "deferred_reply_armed",
            scope_id=scope_id,
            session_id=session.id,
            state_id=state_id,
        )

    async def _wait_and_deliver(
        self,
        *,
        scope_id: str,
        session_id: str,
        state_id: str,
    ) -> None:
        try:
            result = await self._executor.wait_for(state_id)
            pending = self._pending_by_scope.get(scope_id)
            if pending is None or pending.state_id != state_id:
                return
            if pending.session_id != session_id:
                return
            if not await self._is_current_session(scope_id, session_id, state_id):
                logger.info(
                    "deferred_reply_skipped_stale_session",
                    scope_id=scope_id,
                    session_id=session_id,
                    state_id=state_id,
                )
                return

            text = result.response or result.error
            if not text:
                state = await self._executor.get_state(state_id)
                if state is not None and state.result_summary:
                    text = state.result_summary
                else:
                    text = "执行完成，但未产出可展示内容。"

            await self._deliver_chunked(pending.context, text)
            logger.info(
                "deferred_reply_delivered",
                scope_id=scope_id,
                session_id=session_id,
                state_id=state_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001
            logger.exception(
                "deferred_reply_failed",
                scope_id=scope_id,
                session_id=session_id,
                state_id=state_id,
                error=str(error),
            )
        finally:
            pending = self._pending_by_scope.get(scope_id)
            current_task = asyncio.current_task()
            if pending is not None and pending.task is current_task:
                self._pending_by_scope.pop(scope_id, None)

    async def _is_current_session(
        self,
        scope_id: str,
        session_id: str,
        state_id: str,
    ) -> bool:
        (
            _chat_context,
            current_session,
        ) = await self._session_service.get_chat_context_and_current_session(scope_id)
        if current_session is None:
            return False
        return (
            current_session.id == session_id
            and current_session.scheduler_state_id == state_id
        )


__all__ = ["DeferredReplyManager"]
