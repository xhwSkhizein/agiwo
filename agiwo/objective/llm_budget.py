"""Objective-owned LLM cost admission and idempotent usage recording."""

from collections.abc import Awaitable, Callable, Iterable
from datetime import datetime

from agiwo.agent import RunStatus
from agiwo.agent.budget_gate import (
    LlmAttemptAdmitRequest,
    LlmAttemptCostEvent,
    LlmBudgetDenied,
)
from agiwo.agent.models.log import LLMCallCompleted
from agiwo.objective.active_time import check_active_time, default_clock
from agiwo.objective.errors import BudgetBoundaryHit
from agiwo.objective.log import (
    FactBatch,
    ObjectiveFactKind,
    fact_budget_usage_recorded,
)
from agiwo.objective.models import (
    OBJECTIVE_TERMINAL,
    CommandReceiptStatus,
    ObjectiveStatus,
    utc_now,
)
from agiwo.objective.projection import project_objective
from agiwo.objective.store.base import CommandReceipt, ObjectiveStore, command_scope

RunCostReader = Callable[[str], Awaitable[float]]

_TERMINAL_RUN_STATUSES = frozenset(
    {
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.INTERRUPTED,
    }
)

_BLOCKED_STATUSES = frozenset(
    {
        ObjectiveStatus.DRAINING,
        ObjectiveStatus.BUDGET_PAUSED,
        ObjectiveStatus.USER_PAUSED,
        *OBJECTIVE_TERMINAL,
    }
)


def run_cost_from_view(run_view: object | None) -> float:
    """Read aggregated token cost from a RunView-like object."""
    if run_view is None:
        return 0.0
    metrics = getattr(run_view, "metrics", None)
    if metrics is None:
        return 0.0
    return max(0.0, float(getattr(metrics, "token_cost", 0.0) or 0.0))


def sum_llm_cost_from_run_log_entries(entries: Iterable[object]) -> float:
    """Sum observed LLM attempt costs from committed RunLog entries."""
    total = 0.0
    for entry in entries:
        if not isinstance(entry, LLMCallCompleted):
            continue
        if not entry.response_observed:
            continue
        metrics = entry.metrics
        if metrics is None or metrics.token_cost is None:
            continue
        total += max(0.0, float(metrics.token_cost))
    return total


class ObjectiveLlmBudgetGate:
    """Injected onto Agent at dispatch; never imported by agent internals."""

    def __init__(
        self,
        store: ObjectiveStore,
        *,
        objective_id: str,
        clock: Callable[[], datetime] | None = None,
        run_cost_reader: RunCostReader | None = None,
    ) -> None:
        self._store = store
        self._objective_id = objective_id
        self._clock = clock or default_clock
        self._run_cost_reader = run_cost_reader
        self._pending_by_run: dict[str, float] = {}
        self._pending_cost_usd: float = 0.0
        self._recorded_attempt_keys: set[str] = set()

    def _attempt_key(self, event: LlmAttemptCostEvent) -> str:
        return f"{event.run_id}:{event.logical_call_id}:{event.attempt_no}"

    async def _projected_used(self) -> tuple[float, float]:
        facts = await self._store.list_facts(objective_id=self._objective_id)
        view = project_objective(facts, objective_id=self._objective_id)
        if view is None:
            return 0.0, 0.0
        return view.budget.llm_cost_usd.used, view.budget.llm_cost_usd.limit

    async def check_before_attempt(self, request: LlmAttemptAdmitRequest) -> None:
        if request.objective_id != self._objective_id:
            raise LlmBudgetDenied(
                reason="objective_id_mismatch",
                details={
                    "expected": self._objective_id,
                    "actual": request.objective_id,
                },
            )
        facts = await self._store.list_facts(objective_id=self._objective_id)
        view = project_objective(facts, objective_id=self._objective_id)
        if view is None:
            raise LlmBudgetDenied(reason="objective_not_found")
        if view.status in _BLOCKED_STATUSES:
            raise LlmBudgetDenied(
                reason="objective_not_progressable",
                details={"status": view.status.value},
            )
        checked_at = self._clock()
        try:
            check_active_time(
                view,
                checked_at=checked_at,
                pending_action="start_llm_attempt",
            )
        except BudgetBoundaryHit as hit:
            raise LlmBudgetDenied(
                reason="active_seconds_exceeded",
                used=hit.used,
                limit=hit.limit,
                details=hit.details,
            ) from hit
        used = view.budget.llm_cost_usd.used + self._pending_cost_usd
        limit = view.budget.llm_cost_usd.limit
        ceiling = request.call_cost_ceiling
        if used + ceiling > limit + 1e-12:
            hit = BudgetBoundaryHit(
                dimension="llm_cost_usd",
                checked_at=checked_at,
                used=used,
                limit=limit,
                pending_action="start_llm_attempt",
                required=ceiling,
                call_cost_ceiling=ceiling,
                run_id=request.run_id,
                logical_call_id=request.logical_call_id,
                attempt_no=request.attempt_no,
            )
            raise LlmBudgetDenied(
                reason="llm_cost_ceiling_exceeded",
                used=used,
                limit=limit,
                call_cost_ceiling=ceiling,
                details=hit.details,
            )

    async def record_attempt_cost(self, event: LlmAttemptCostEvent) -> None:
        if event.objective_id != self._objective_id:
            return
        attempt_key = self._attempt_key(event)
        if attempt_key in self._recorded_attempt_keys:
            return
        cost = float(event.cost_usd) if event.response_observed else 0.0
        if cost < 0:
            cost = 0.0
        self._recorded_attempt_keys.add(attempt_key)
        self._pending_by_run[event.run_id] = (
            self._pending_by_run.get(event.run_id, 0.0) + cost
        )
        self._pending_cost_usd += cost

    async def _run_total_already_flushed(self, run_id: str) -> bool:
        facts = await self._store.list_facts(objective_id=self._objective_id)
        for fact in facts:
            if fact.kind != ObjectiveFactKind.BUDGET_USAGE_RECORDED:
                continue
            payload = fact.payload
            if (
                payload.get("source") == "llm_run_total"
                and payload.get("run_id") == run_id
            ):
                return True
        return False

    async def _resolve_run_cost(self, run_id: str, run_cost: float | None) -> float:
        if run_cost is not None:
            return max(0.0, float(run_cost))
        if self._run_cost_reader is None:
            return 0.0
        return max(0.0, float(await self._run_cost_reader(run_id)))

    async def seed_pending_from_run_log(
        self,
        run_id: str,
        *,
        run_cost: float | None = None,
    ) -> float:
        """Rebuild in-memory pending from RunLog when llm_run_total is not flushed."""
        if await self._run_total_already_flushed(run_id):
            return 0.0
        cost = await self._resolve_run_cost(run_id, run_cost)
        existing = self._pending_by_run.get(run_id, 0.0)
        if cost <= existing + 1e-12:
            return existing
        delta = cost - existing
        self._pending_by_run[run_id] = cost
        self._pending_cost_usd += delta
        return cost

    async def flush_run_usage(
        self,
        run_id: str,
        *,
        run_cost_override: float | None = None,
    ) -> float:
        """Append one Run-boundary BudgetUsageRecorded fact; clear in-memory pending."""
        if await self._run_total_already_flushed(run_id):
            pending = self._pending_by_run.pop(run_id, 0.0)
            self._pending_cost_usd = max(0.0, self._pending_cost_usd - pending)
            return 0.0

        if run_cost_override is not None:
            existing = self._pending_by_run.pop(run_id, 0.0)
            self._pending_cost_usd = max(0.0, self._pending_cost_usd - existing)
            pending = max(0.0, float(run_cost_override))
        else:
            pending = self._pending_by_run.pop(run_id, 0.0)
        if pending <= 0:
            self._pending_by_run.pop(run_id, None)
            return 0.0

        committed_used, _ = await self._projected_used()
        used_after = committed_used + pending
        now = utc_now()
        seq = await self._store.get_max_sequence(self._objective_id) + 1
        idempotency_key = f"llm_run_total:{run_id}"
        receipt = CommandReceipt(
            scope=command_scope(self._objective_id, "llm_cost"),
            idempotency_key=idempotency_key,
            request_hash=idempotency_key,
            status=CommandReceiptStatus.COMPLETED,
            response_payload={
                "cost_usd": pending,
                "used_after": used_after,
                "run_id": run_id,
            },
            created_at=now,
            completed_at=now,
        )
        fact = FactBatch(
            objective_id=self._objective_id,
            start_sequence=seq,
            now=now,
        ).add(
            fact_budget_usage_recorded(
                dimension="llm_cost_usd",
                delta=pending,
                used_after=used_after,
                provenance={
                    "source": "llm_run_total",
                    "run_id": run_id,
                },
            )
        )
        await self._store.commit_command(receipt=receipt, facts=[fact])
        self._pending_cost_usd = max(0.0, self._pending_cost_usd - pending)
        return pending


async def reconcile_missing_llm_run_totals(
    store: ObjectiveStore,
    runs: object | None,
    *,
    objective_ids: Iterable[str],
) -> int:
    """Flush terminal Runs that have RunLog cost but no llm_run_total fact."""
    if runs is None:
        return 0
    get_run_status = getattr(runs, "get_run_status", None)
    get_run_view = getattr(runs, "get_run_view", None)
    if not callable(get_run_status) or not callable(get_run_view):
        return 0

    flushed = 0
    for objective_id in objective_ids:
        facts = await store.list_facts(objective_id=objective_id)
        started_run_ids = {
            str(f.payload.get("run_id"))
            for f in facts
            if f.kind is ObjectiveFactKind.ROOT_RUN_STARTED and f.payload.get("run_id")
        }
        flushed_run_ids = {
            str(f.payload.get("run_id"))
            for f in facts
            if f.kind is ObjectiveFactKind.BUDGET_USAGE_RECORDED
            and f.payload.get("source") == "llm_run_total"
            and f.payload.get("run_id")
        }
        missing = started_run_ids - flushed_run_ids
        if not missing:
            continue

        async def _run_cost_reader(run_id: str) -> float:
            view = await get_run_view(run_id)
            return run_cost_from_view(view)

        gate = ObjectiveLlmBudgetGate(
            store,
            objective_id=objective_id,
            run_cost_reader=_run_cost_reader,
        )
        for run_id in missing:
            status = await get_run_status(run_id)
            if status not in _TERMINAL_RUN_STATUSES:
                continue
            view = await get_run_view(run_id)
            cost = run_cost_from_view(view)
            if cost <= 0:
                continue
            delta = await gate.flush_run_usage(run_id, run_cost_override=cost)
            if delta > 0:
                flushed += 1
    return flushed


__all__ = [
    "ObjectiveLlmBudgetGate",
    "RunCostReader",
    "reconcile_missing_llm_run_totals",
    "run_cost_from_view",
    "sum_llm_cost_from_run_log_entries",
]
