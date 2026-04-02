"""Console session read models assembled from storage and scheduler state."""

from dataclasses import dataclass

from agiwo.agent.models.input import UserMessage
from agiwo.agent.storage.base import RunStepStorage
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
from server.services.metrics import SessionAggregate, collect_session_aggregates


@dataclass(slots=True)
class SessionViewItem:
    aggregate: SessionAggregate | None = None
    session: Session | None = None
    chat_context: ChannelChatContext | None = None
    root_state_status: str | None = None


def _session_sort_key(item: SessionViewItem) -> tuple[str, str]:
    updated_at = None
    if item.session is not None:
        updated_at = item.session.updated_at
    elif item.aggregate is not None:
        updated_at = item.aggregate.updated_at
    serialized = updated_at.isoformat() if updated_at is not None else ""
    session_id = (
        item.session.id if item.session is not None else item.aggregate.session_id
    )
    return (serialized, session_id)


def _summary_record(item: SessionViewItem) -> SessionSummaryRecord:
    aggregate = item.aggregate
    session = item.session
    last_run = aggregate.last_run if aggregate is not None else None
    last_user_input = (
        UserMessage.to_transport_payload(last_run.user_input)
        if last_run is not None
        else None
    )
    return SessionSummaryRecord(
        session_id=session.id if session is not None else aggregate.session_id,
        agent_id=(
            aggregate.agent_id
            if aggregate is not None and aggregate.agent_id is not None
            else (session.base_agent_id if session is not None else None)
        ),
        last_user_input=last_user_input,
        last_response=(
            last_run.response_content[:200]
            if last_run is not None and last_run.response_content
            else None
        ),
        run_count=aggregate.metrics.run_count if aggregate is not None else 0,
        step_count=aggregate.metrics.step_count if aggregate is not None else 0,
        metrics=aggregate.metrics if aggregate is not None else RunMetricsSummary(),
        created_at=session.created_at if session is not None else aggregate.created_at,
        updated_at=session.updated_at if session is not None else aggregate.updated_at,
        chat_context_scope_id=session.chat_context_scope_id
        if session is not None
        else None,
        created_by=session.created_by if session is not None else None,
        base_agent_id=session.base_agent_id if session is not None else None,
        root_state_status=item.root_state_status,
        source_session_id=session.source_session_id if session is not None else None,
        fork_context_summary=session.fork_context_summary
        if session is not None
        else None,
    )


class SessionViewService:
    def __init__(
        self,
        *,
        run_storage: RunStepStorage,
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
        items = await self._collect_session_views(agent_id=agent_id)
        total = len(items)
        page = items[offset : offset + limit]
        return PageSlice(
            items=[_summary_record(item) for item in page],
            limit=limit,
            offset=offset,
            has_more=offset + limit < total,
            total=total,
        )

    async def get_session_detail(self, session_id: str) -> SessionDetailRecord | None:
        items = await self._collect_session_views(session_id=session_id)
        if not items:
            return None
        item = items[0]
        scheduler_state = await self._get_scheduler_state(
            item.session, session_id=session_id
        )
        return SessionDetailRecord(
            summary=_summary_record(item),
            session=item.session,
            chat_context=item.chat_context,
            scheduler_state=scheduler_state,
        )

    async def _collect_session_views(
        self,
        *,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> list[SessionViewItem]:
        store_sessions = await self._list_store_sessions(
            agent_id=agent_id, session_id=session_id
        )
        store_session_ids = {session.id for session in store_sessions}
        aggregate_map = await self._collect_aggregate_map(
            agent_id=agent_id,
            session_id=session_id,
            session_ids=store_session_ids,
        )
        view_map: dict[str, SessionViewItem] = {
            key: SessionViewItem(aggregate=value)
            for key, value in aggregate_map.items()
        }

        for session in store_sessions:
            item = view_map.get(session.id)
            if item is None:
                item = SessionViewItem()
                view_map[session.id] = item
            item.session = session
            if (
                self._session_store is not None
                and session.chat_context_scope_id is not None
            ):
                item.chat_context = await self._session_store.get_chat_context(
                    session.chat_context_scope_id
                )
            item.root_state_status = await self._get_root_state_status(session.id)

        if (
            session_id is not None
            and session_id in view_map
            and view_map[session_id].root_state_status is None
        ):
            view_map[session_id].root_state_status = await self._get_root_state_status(
                session_id
            )

        items = list(view_map.values())
        items.sort(key=_session_sort_key, reverse=True)
        return items

    async def _collect_aggregate_map(
        self,
        *,
        agent_id: str | None,
        session_id: str | None,
        session_ids: set[str],
    ) -> dict[str, SessionAggregate]:
        if session_id is not None:
            aggregates = await collect_session_aggregates(
                self._run_storage,
                session_id=session_id,
            )
        else:
            aggregates = await collect_session_aggregates(self._run_storage)
            if agent_id is not None:
                aggregates = (
                    [
                        aggregate
                        for aggregate in aggregates
                        if aggregate.session_id in session_ids
                    ]
                    if session_ids
                    else []
                )
        return {aggregate.session_id: aggregate for aggregate in aggregates}

    async def _list_store_sessions(
        self,
        *,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> list[Session]:
        if self._session_store is None:
            return []
        if session_id is not None:
            session = await self._session_store.get_session(session_id)
            return [session] if session is not None else []
        if agent_id is not None:
            return await self._session_store.list_sessions_by_base_agent(agent_id)
        return await self._session_store.list_sessions()

    async def _get_scheduler_state(
        self,
        session: Session | None,
        *,
        session_id: str,
    ):
        if self._scheduler is None:
            return None
        target_state_id = session.id if session is not None else session_id
        return await self._scheduler.get_state(target_state_id)

    async def _get_root_state_status(self, session_id: str) -> str | None:
        if self._scheduler is None:
            return None
        state = await self._scheduler.get_state(session_id)
        if state is None:
            return None
        return (
            state.status.value if hasattr(state.status, "value") else str(state.status)
        )
