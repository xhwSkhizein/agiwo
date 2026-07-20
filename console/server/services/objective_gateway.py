"""SessionObjectiveGateway: session-scoped entry point into ObjectiveService.

Console channels (web chat, Feishu) no longer call ``Scheduler.route_root_input``
directly for normal user turns. Instead they route through this gateway, which
enforces the "at most one active Objective per Session" rule described in
P5-06 and applies the one-shot fork-context-summary notice described in P2-04.
"""

import asyncio

from agiwo.agent.models.input import UserMessage
from agiwo.objective import (
    BudgetLimits,
    CommandResult,
    CreateObjectiveRequest,
    ObjectiveService,
    SubmitUserInputRequest,
)
from agiwo.objective.projection import ObjectiveView
from agiwo.scheduler.engine import Scheduler
from agiwo.utils.logging import get_logger

from server.models.session import ChannelChatSessionStore

logger = get_logger(__name__)

# Console defaults when no explicit budget is supplied at Objective creation.
DEFAULT_BUDGET_LIMITS = BudgetLimits(
    handoffs=10,
    verification_attempts=5,
    llm_cost_usd=5.0,
    active_seconds=3600,
)

_FORK_INJECT_ATTEMPTS = 3
_FORK_INJECT_RETRY_SECONDS = 0.3


class SessionObjectiveGateway:
    """Route a Session's user turns to Objective create-or-continue commands."""

    def __init__(
        self,
        *,
        objective_service: ObjectiveService,
        session_store: ChannelChatSessionStore,
        scheduler: Scheduler | None = None,
    ) -> None:
        self._service = objective_service
        self._session_store = session_store
        self._scheduler = scheduler

    async def handle_user_message(
        self,
        session_id: str,
        user_message: UserMessage,
        *,
        idempotency_key: str,
        budget: BudgetLimits | None = None,
    ) -> CommandResult:
        if not user_message.is_user_provided:
            raise ValueError(
                "SessionObjectiveGateway.handle_user_message requires an "
                "is_user_provided=True UserMessage"
            )

        active = await self._find_active_objective(session_id)
        if active is None:
            result = await self._service.create_objective(
                CreateObjectiveRequest(
                    session_id=session_id,
                    user_message=user_message,
                    budget=budget or DEFAULT_BUDGET_LIMITS,
                    idempotency_key=idempotency_key,
                )
            )
            await self._consume_fork_summary_if_needed(
                session_id,
                objective_id=result.objective_id,
            )
            return result

        return await self._service.submit_user_input(
            SubmitUserInputRequest(
                objective_id=active.objective_id,
                user_message=user_message,
                idempotency_key=idempotency_key,
            )
        )

    async def _find_active_objective(self, session_id: str) -> ObjectiveView | None:
        views = await self._service.list_by_session(session_id)
        for view in views:
            if not view.is_terminal:
                return view
        return None

    async def _consume_fork_summary_if_needed(
        self,
        session_id: str,
        *,
        objective_id: str,
    ) -> None:
        """Deliver a one-shot fork-context notice, then clear it (P2-04/P5-06).

        The notice targets the freshly created Objective's root Run.
        Because that Run is dispatched asynchronously by the outbox
        dispatcher, injection is best-effort: a short bounded retry covers
        the common case where dispatch completes within a few hundred
        milliseconds, and a failure is logged (not raised) otherwise so the
        summary is still cleared and the user-facing flow is not blocked.
        """
        session = await self._session_store.get_session(session_id)
        if session is None or not session.fork_context_summary:
            return
        summary = session.fork_context_summary

        if self._scheduler is not None:
            await self._try_inject_fork_notice(objective_id, summary)

        session.fork_context_summary = None
        await self._session_store.upsert_session(session)

    async def _try_inject_fork_notice(self, objective_id: str, summary: str) -> None:
        assert self._scheduler is not None
        notice = UserMessage.from_system(
            "This session was forked from a previous session. "
            f"Context summary from the source session:\n{summary}"
        )
        for attempt in range(1, _FORK_INJECT_ATTEMPTS + 1):
            view = await self._service.get_view(objective_id)
            root_run_id = (
                view.active_assignment.root_run_id
                if view is not None and view.active_assignment is not None
                else None
            )
            if root_run_id:
                try:
                    await self._scheduler.inject_user_message(root_run_id, notice)
                    return
                except ValueError:
                    pass
            if attempt < _FORK_INJECT_ATTEMPTS:
                await asyncio.sleep(_FORK_INJECT_RETRY_SECONDS)
        logger.warning(
            "objective_fork_summary_injection_skipped",
            objective_id=objective_id,
            reason="root_run_not_live_yet",
        )


__all__ = ["DEFAULT_BUDGET_LIMITS", "SessionObjectiveGateway"]
