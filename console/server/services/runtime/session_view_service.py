"""Console session read models assembled from storage and scheduler state."""

from agiwo.agent.models.input import UserMessage
from agiwo.agent.models.run import RunView
from agiwo.agent.storage.base import RunLogStorage
from agiwo.scheduler.engine import Scheduler

from server.models.metrics import RunMetricsSummary
from server.models.session import (
    ChannelChatContext,
    ChannelChatSessionStore,
    PageSlice,
    Session,
    SessionDetailRecord,
    SessionSummaryRecord,
)
from server.services.metrics import summarize_run_views_paginated


class SessionViewService:
    def __init__(
        self,
        *,
        run_storage: RunLogStorage,
        session_store: ChannelChatSessionStore | None,
        scheduler: Scheduler | None,
    ) -> None:
        self._run_storage = run_storage
        self._session_store = session_store
        self._scheduler = scheduler

    async def list_sessions(
        self,
        *,
        limit: int,
        offset: int,
        agent_id: str | None = None,
    ) -> PageSlice[SessionSummaryRecord]:
        sessions = await self._list_store_sessions(agent_id=agent_id)
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        total = len(sessions)
        page = sessions[offset : offset + limit]

        if not page:
            return PageSlice(
                items=[], limit=limit, offset=offset, has_more=False, total=total
            )

        page_ids = [s.id for s in page]
        run_counts, step_counts, latest_runs = await self._batch_fetch_summaries(
            page_ids
        )
        items = [
            self._assemble_summary(
                s,
                last_run=latest_runs.get(s.id),
                run_count=run_counts.get(s.id, 0),
                step_count=step_counts.get(s.id, 0),
                root_state_status=await self._get_root_state_status(s.id),
            )
            for s in page
        ]
        return PageSlice(
            items=items,
            limit=limit,
            offset=offset,
            has_more=offset + limit < total,
            total=total,
        )

    async def get_session_detail(self, session_id: str) -> SessionDetailRecord | None:
        session = await self._get_session(session_id)
        if session is None:
            return None

        metrics = await summarize_run_views_paginated(
            self._run_storage, session_id=session_id
        )
        last_run = await self._run_storage.get_latest_run_view(session_id)
        summary = self._assemble_summary(
            session,
            last_run=last_run,
            run_count=metrics.run_count,
            step_count=await self._run_storage.get_committed_step_count(session_id),
            root_state_status=await self._get_root_state_status(session_id),
            metrics=metrics,
        )
        scheduler_state = await self._get_scheduler_state(session_id)
        chat_context = await self._get_chat_context(session)

        return SessionDetailRecord(
            summary=summary,
            session=session,
            chat_context=chat_context,
            scheduler_state=scheduler_state,
        )

    # -- Internal helpers ------------------------------------------------------

    async def _batch_fetch_summaries(
        self, session_ids: list[str]
    ) -> tuple[dict[str, int], dict[str, int], dict[str, RunView | None]]:
        run_counts = await self._run_storage.batch_count_run_views(session_ids)
        step_counts = await self._run_storage.batch_get_committed_step_counts(
            session_ids
        )
        latest_runs = await self._run_storage.batch_get_latest_run_views(session_ids)
        return run_counts, step_counts, latest_runs

    @staticmethod
    def _assemble_summary(
        session: Session,
        *,
        last_run: RunView | None,
        run_count: int,
        step_count: int,
        root_state_status: str | None,
        metrics: RunMetricsSummary | None = None,
    ) -> SessionSummaryRecord:
        last_user_input = (
            UserMessage.to_transport_payload(last_run.last_user_input)
            if last_run is not None
            else None
        )
        last_response = (
            last_run.response[:200]
            if last_run is not None and last_run.response
            else None
        )
        return SessionSummaryRecord(
            session_id=session.id,
            base_agent_id=session.base_agent_id,
            last_user_input=last_user_input,
            last_response=last_response,
            run_count=run_count,
            step_count=step_count,
            metrics=metrics if metrics is not None else RunMetricsSummary(),
            created_at=session.created_at,
            updated_at=session.updated_at,
            chat_context_scope_id=session.chat_context_scope_id,
            created_by=session.created_by,
            root_state_status=root_state_status,
            source_session_id=session.source_session_id,
            fork_context_summary=session.fork_context_summary,
        )

    async def _list_store_sessions(
        self,
        *,
        agent_id: str | None = None,
    ) -> list[Session]:
        if self._session_store is None:
            return []
        if agent_id is not None:
            return await self._session_store.list_sessions_by_base_agent(agent_id)
        return await self._session_store.list_sessions()

    async def _get_session(self, session_id: str) -> Session | None:
        if self._session_store is None:
            return None
        return await self._session_store.get_session(session_id)

    async def _get_chat_context(self, session: Session) -> ChannelChatContext | None:
        if self._session_store is None or session.chat_context_scope_id is None:
            return None
        return await self._session_store.get_chat_context(session.chat_context_scope_id)

    async def _get_scheduler_state(self, session_id: str):
        if self._scheduler is None:
            return None
        return await self._scheduler.get_state(session_id)

    async def _get_root_state_status(self, session_id: str) -> str | None:
        if self._scheduler is None:
            return None
        state = await self._scheduler.get_state(session_id)
        if state is None:
            return None
        return (
            state.status.value if hasattr(state.status, "value") else str(state.status)
        )
