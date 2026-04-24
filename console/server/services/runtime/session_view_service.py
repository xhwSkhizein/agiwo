"""Console session read models assembled from the run query facade and scheduler."""

from agiwo.agent import RuntimeDecisionState, RunView, UserMessage
from agiwo.scheduler.engine import Scheduler
from agiwo.scheduler.models import AgentState

from server.models.metrics import RunMetricsSummary
from server.models.session import (
    ChannelChatContext,
    ChannelChatSessionStore,
    PageSlice,
    RuntimeDecisionRecord,
    Session,
    SessionDetailRecord,
    SessionObservabilityRecord,
    SessionSummaryRecord,
)
from server.services.metrics import add_run_to_summary
from server.services.runtime.run_query_service import RunQueryService
from server.services.runtime.trace_query_service import TraceQueryService


class SessionViewService:
    def __init__(
        self,
        *,
        run_queries: RunQueryService,
        trace_queries: TraceQueryService,
        session_store: ChannelChatSessionStore | None,
        scheduler: Scheduler | None,
    ) -> None:
        self._run_queries = run_queries
        self._trace_queries = trace_queries
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

        stats = await self._run_queries.get_session_run_snapshot(session_id)
        metrics = self._build_metrics_summary(stats.run_views)
        scheduler_state = await self._get_scheduler_state(session_id)
        summary = self._assemble_summary(
            session,
            last_run=stats.run_views[0] if stats.run_views else None,
            run_count=len(stats.run_views),
            step_count=stats.committed_step_count,
            root_state_status=self._root_state_status(scheduler_state),
            metrics=metrics,
        )
        chat_context = await self._get_chat_context(session)

        return SessionDetailRecord(
            summary=summary,
            session=session,
            chat_context=chat_context,
            scheduler_state=scheduler_state,
            observability=await self._build_observability(
                session_id=session_id,
                runtime_decisions=stats.runtime_decisions,
            ),
        )

    # -- Internal helpers ------------------------------------------------------

    @staticmethod
    def _build_metrics_summary(run_views: list[RunView]) -> RunMetricsSummary:
        metrics = RunMetricsSummary()
        for run in run_views:
            add_run_to_summary(metrics, run)
        return metrics

    async def _batch_fetch_summaries(
        self, session_ids: list[str]
    ) -> tuple[dict[str, int], dict[str, int], dict[str, RunView | None]]:
        return await self._run_queries.batch_get_session_summaries(session_ids)

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
        return self._root_state_status(state)

    @staticmethod
    def _root_state_status(state: AgentState | None) -> str | None:
        if state is None:
            return None
        return (
            state.status.value if hasattr(state.status, "value") else str(state.status)
        )

    async def _build_observability(
        self,
        *,
        session_id: str,
        runtime_decisions: RuntimeDecisionState,
    ) -> SessionObservabilityRecord:
        return SessionObservabilityRecord(
            recent_traces=await self._trace_queries.list_session_recent_traces(
                session_id
            ),
            decision_events=self._build_runtime_decision_records(runtime_decisions),
        )

    @staticmethod
    def _build_runtime_decision_records(
        runtime_decisions: RuntimeDecisionState,
    ) -> list[RuntimeDecisionRecord]:
        decisions: list[RuntimeDecisionRecord] = []

        if runtime_decisions.latest_termination is not None:
            decision = runtime_decisions.latest_termination
            decisions.append(
                RuntimeDecisionRecord(
                    kind="termination",
                    sequence=decision.sequence,
                    run_id=decision.run_id,
                    agent_id=decision.agent_id,
                    created_at=decision.created_at,
                    summary=f"{decision.reason.value} via {decision.source}",
                    details={
                        "reason": decision.reason.value,
                        "phase": decision.phase,
                        "source": decision.source,
                    },
                )
            )

        if runtime_decisions.latest_compaction is not None:
            decision = runtime_decisions.latest_compaction
            decisions.append(
                RuntimeDecisionRecord(
                    kind="compaction",
                    sequence=decision.sequence,
                    run_id=decision.run_id,
                    agent_id=decision.agent_id,
                    created_at=decision.created_at,
                    summary=(
                        f"seq {decision.metadata.start_seq}-{decision.metadata.end_seq} "
                        f"{decision.metadata.before_token_estimate} -> "
                        f"{decision.metadata.after_token_estimate} tokens"
                    ),
                    details={
                        "start_sequence": decision.metadata.start_seq,
                        "end_sequence": decision.metadata.end_seq,
                        "before_token_estimate": decision.metadata.before_token_estimate,
                        "after_token_estimate": decision.metadata.after_token_estimate,
                        "message_count": decision.metadata.message_count,
                        "summary": decision.summary,
                    },
                )
            )

        if runtime_decisions.latest_step_back is not None:
            decision = runtime_decisions.latest_step_back
            decisions.append(
                RuntimeDecisionRecord(
                    kind="step_back",
                    sequence=decision.sequence,
                    run_id=decision.run_id,
                    agent_id=decision.agent_id,
                    created_at=decision.created_at,
                    summary=(
                        f"{decision.affected_count} results condensed, "
                        f"checkpoint at seq {decision.checkpoint_seq}"
                    ),
                    details={
                        "affected_count": decision.affected_count,
                        "checkpoint_seq": decision.checkpoint_seq,
                        "experience": decision.experience,
                    },
                )
            )

        if runtime_decisions.latest_rollback is not None:
            decision = runtime_decisions.latest_rollback
            decisions.append(
                RuntimeDecisionRecord(
                    kind="rollback",
                    sequence=decision.sequence,
                    run_id=decision.run_id,
                    agent_id=decision.agent_id,
                    created_at=decision.created_at,
                    summary=(
                        f"seq {decision.start_sequence}-{decision.end_sequence} hidden"
                    ),
                    details={
                        "start_sequence": decision.start_sequence,
                        "end_sequence": decision.end_sequence,
                        "reason": decision.reason,
                    },
                )
            )

        decisions.sort(
            key=lambda item: (item.created_at, item.sequence),
            reverse=True,
        )
        return decisions
