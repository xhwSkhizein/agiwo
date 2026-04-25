"""RunLog-backed query facade for Console run and step views."""

from dataclasses import dataclass

from agiwo.agent import RunLogStorage, RunView, RuntimeDecisionState, StepView
from agiwo.agent.models.log import (
    CompactionApplied,
    CompactionFailed,
    RunRolledBack,
    StepBackApplied,
    TerminationDecided,
)

from server.models.session import PageSlice
from server.models.session import RuntimeDecisionRecord
from server.services.runtime.runtime_observability import (
    build_runtime_decision_record_from_entry,
)


@dataclass(slots=True)
class SessionRunSnapshot:
    run_views: list[RunView]
    committed_step_count: int
    runtime_decisions: RuntimeDecisionState


@dataclass(slots=True)
class RunQueryService:
    run_storage: RunLogStorage

    async def list_runs(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        limit: int,
        offset: int,
    ) -> PageSlice[RunView]:
        runs = await self.run_storage.list_run_views(
            user_id=user_id,
            session_id=session_id,
            limit=limit + 1,
            offset=offset,
        )
        has_more = len(runs) > limit
        return PageSlice(
            items=runs[:limit],
            limit=limit,
            offset=offset,
            has_more=has_more,
            total=None,
        )

    async def get_run(self, run_id: str) -> RunView | None:
        return await self.run_storage.get_run_view(run_id)

    async def list_session_steps(
        self,
        session_id: str,
        *,
        start_seq: int | None = None,
        end_seq: int | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        limit: int,
        order: str,
    ) -> PageSlice[StepView]:
        raw_steps = await self.run_storage.list_step_views(
            session_id=session_id,
            start_seq=start_seq,
            end_seq=end_seq,
            run_id=run_id,
            agent_id=agent_id,
            include_hidden_from_context=False,
            limit=limit + 1,
            order=order,
        )
        has_more = len(raw_steps) > limit
        total = None
        if (
            start_seq is None
            and end_seq is None
            and run_id is None
            and agent_id is None
        ):
            total = len(
                await self.run_storage.list_step_views(
                    session_id=session_id,
                    include_hidden_from_context=False,
                    limit=100_000,
                )
            )
        return PageSlice(
            items=raw_steps[:limit],
            limit=limit,
            offset=0,
            has_more=has_more,
            total=total,
        )

    async def get_session_run_snapshot(self, session_id: str) -> SessionRunSnapshot:
        stats = await self.run_storage.get_session_run_stats(session_id)
        return SessionRunSnapshot(
            run_views=stats.run_views,
            committed_step_count=stats.committed_step_count,
            runtime_decisions=await self.run_storage.get_runtime_decision_state(
                session_id=session_id
            ),
        )

    async def get_runtime_decision_state(
        self,
        session_id: str,
        *,
        run_id: str | None = None,
        agent_id: str | None = None,
    ) -> RuntimeDecisionState:
        return await self.run_storage.get_runtime_decision_state(
            session_id=session_id,
            run_id=run_id,
            agent_id=agent_id,
        )

    async def list_runtime_decision_events(
        self,
        session_id: str,
        *,
        run_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 20,
    ) -> list[RuntimeDecisionRecord]:
        entries = await self.run_storage.list_entries(
            session_id=session_id,
            run_id=run_id,
            agent_id=agent_id,
            limit=100_000,
        )
        decision_entries = [
            entry
            for entry in entries
            if isinstance(
                entry,
                (
                    CompactionApplied,
                    CompactionFailed,
                    StepBackApplied,
                    RunRolledBack,
                    TerminationDecided,
                ),
            )
        ]
        decision_entries.sort(key=lambda entry: entry.sequence, reverse=True)
        return [
            build_runtime_decision_record_from_entry(entry)
            for entry in decision_entries[:limit]
        ]

    async def batch_get_session_summaries(
        self,
        session_ids: list[str],
    ) -> tuple[dict[str, int], dict[str, int], dict[str, RunView | None]]:
        run_counts = await self.run_storage.batch_count_run_views(session_ids)
        step_counts = await self.run_storage.batch_get_committed_step_counts(
            session_ids
        )
        latest_runs = await self.run_storage.batch_get_latest_run_views(session_ids)
        return run_counts, step_counts, latest_runs


__all__ = ["RunQueryService", "SessionRunSnapshot"]
