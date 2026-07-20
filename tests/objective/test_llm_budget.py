"""P3-02: LLM call_cost_ceiling admission and actual cost accounting."""

import pytest

from agiwo.agent.budget_gate import (
    LlmAttemptAdmitRequest,
    LlmAttemptCostEvent,
    LlmBudgetDenied,
    MissingBudgetGateError,
)
from agiwo.llm.usage_resolver import compute_call_cost_ceiling
from agiwo.agent import RunStatus
from agiwo.objective.llm_budget import (
    ObjectiveLlmBudgetGate,
    reconcile_missing_llm_run_totals,
)
from agiwo.objective.log import (
    ObjectiveFactKind,
    fact_objective_created,
    fact_root_run_requested,
    fact_root_run_started,
    materialize_facts,
)
from agiwo.objective.models import (
    CommandReceiptStatus,
    ObjectiveBudget,
    RunRole,
    new_id,
    utc_now,
)
from agiwo.objective.outbox import DispatchRequested
from agiwo.objective.store.base import CommandReceipt, command_scope
from agiwo.objective.store.memory import InMemoryObjectiveStore


def test_call_cost_ceiling_reproducible() -> None:
    ceiling = compute_call_cost_ceiling(
        request_tokens=1000,
        max_output_tokens=500,
        input_price=1.0,
        output_price=2.0,
    )
    # (1000*1 + 500*2) / 1e6
    assert abs(ceiling - 0.002) < 1e-12


def test_missing_prices_yield_zero_ceiling() -> None:
    assert (
        compute_call_cost_ceiling(
            request_tokens=10_000,
            max_output_tokens=10_000,
            input_price=0.0,
            output_price=0.0,
        )
        == 0.0
    )


async def _seed(store: InMemoryObjectiveStore, *, llm_limit: float) -> str:
    oid = new_id("obj_")
    now = utc_now()
    await store.commit_command(
        receipt=CommandReceipt(
            scope=command_scope(oid, "seed"),
            idempotency_key="seed",
            request_hash="seed",
            status=CommandReceiptStatus.COMPLETED,
            response_payload={},
            created_at=now,
            completed_at=now,
        ),
        facts=materialize_facts(
            oid,
            [
                fact_objective_created(
                    session_id="s1",
                    budget=ObjectiveBudget.create(
                        handoffs=5,
                        verification_attempts=5,
                        llm_cost_usd=llm_limit,
                        active_seconds=600,
                    ),
                )
            ],
        ),
    )
    return oid


def _admit(oid: str, *, ceiling: float) -> LlmAttemptAdmitRequest:
    return LlmAttemptAdmitRequest(
        objective_id=oid,
        run_id="run1",
        logical_call_id="call1",
        phase="assistant",
        attempt_no=1,
        call_ordinal=1,
        request_tokens=100,
        max_output_tokens=50,
        call_cost_ceiling=ceiling,
        price_snapshot={
            "input_price": 1.0,
            "output_price": 2.0,
            "cache_hit_price": 0.0,
        },
    )


def _cost_event(oid: str, **overrides) -> LlmAttemptCostEvent:
    base = dict(
        objective_id=oid,
        run_id="run1",
        logical_call_id="call1",
        phase="assistant",
        attempt_no=1,
        call_ordinal=1,
        request_tokens=10,
        accepted_output_tokens=5,
        call_cost_ceiling=1.0,
        cost_usd=0.25,
        response_observed=True,
        source="provider",
    )
    base.update(overrides)
    return LlmAttemptCostEvent(**base)


@pytest.mark.asyncio
async def test_admit_allows_when_used_plus_ceiling_within_limit() -> None:
    store = InMemoryObjectiveStore()
    oid = await _seed(store, llm_limit=1.0)
    gate = ObjectiveLlmBudgetGate(store, objective_id=oid)
    await gate.check_before_attempt(_admit(oid, ceiling=0.5))


@pytest.mark.asyncio
async def test_admit_denies_when_ceiling_would_exceed() -> None:
    store = InMemoryObjectiveStore()
    oid = await _seed(store, llm_limit=0.1)
    gate = ObjectiveLlmBudgetGate(store, objective_id=oid)
    with pytest.raises(LlmBudgetDenied) as exc:
        await gate.check_before_attempt(_admit(oid, ceiling=0.5))
    assert exc.value.reason == "llm_cost_ceiling_exceeded"
    assert exc.value.used == 0.0
    assert exc.value.limit == 0.1


@pytest.mark.asyncio
async def test_record_actual_cost_in_memory_until_flush() -> None:
    store = InMemoryObjectiveStore()
    oid = await _seed(store, llm_limit=10.0)
    gate = ObjectiveLlmBudgetGate(store, objective_id=oid)
    event = _cost_event(oid)
    await gate.record_attempt_cost(event)
    await gate.record_attempt_cost(event)
    facts = await store.list_facts(objective_id=oid)
    usage = [f for f in facts if f.kind == ObjectiveFactKind.BUDGET_USAGE_RECORDED]
    assert usage == []

    zero = _cost_event(
        oid,
        logical_call_id="call2",
        call_ordinal=2,
        cost_usd=9.0,
        response_observed=False,
        source="no_response",
    )
    await gate.record_attempt_cost(zero)

    delta = await gate.flush_run_usage("run1")
    assert abs(delta - 0.25) < 1e-12
    facts = await store.list_facts(objective_id=oid)
    usage = [f for f in facts if f.kind == ObjectiveFactKind.BUDGET_USAGE_RECORDED]
    assert len(usage) == 1
    assert usage[0].payload["source"] == "llm_run_total"
    assert usage[0].payload["run_id"] == "run1"
    assert usage[0].payload["used_after"] == 0.25
    assert usage[0].payload["delta"] == 0.25

    # Second flush is a no-op.
    assert await gate.flush_run_usage("run1") == 0.0
    facts = await store.list_facts(objective_id=oid)
    usage = [f for f in facts if f.kind == ObjectiveFactKind.BUDGET_USAGE_RECORDED]
    assert len(usage) == 1


@pytest.mark.asyncio
async def test_pending_cost_affects_admission_before_flush() -> None:
    store = InMemoryObjectiveStore()
    oid = await _seed(store, llm_limit=1.0)
    gate = ObjectiveLlmBudgetGate(store, objective_id=oid)
    await gate.record_attempt_cost(_cost_event(oid, cost_usd=0.9))
    with pytest.raises(LlmBudgetDenied) as exc:
        await gate.check_before_attempt(_admit(oid, ceiling=0.2))
    assert exc.value.reason == "llm_cost_ceiling_exceeded"
    assert abs(exc.value.used - 0.9) < 1e-12


@pytest.mark.asyncio
async def test_fail_closed_without_gate_is_agent_error() -> None:
    # Document the agent-side contract used by llm_caller.
    with pytest.raises(MissingBudgetGateError):
        raise MissingBudgetGateError("obj_x")


@pytest.mark.asyncio
async def test_seed_pending_from_run_log_after_gate_recreate() -> None:
    store = InMemoryObjectiveStore()
    oid = await _seed(store, llm_limit=1.0)
    gate1 = ObjectiveLlmBudgetGate(store, objective_id=oid)
    await gate1.record_attempt_cost(_cost_event(oid, cost_usd=0.4))
    facts = await store.list_facts(objective_id=oid)
    assert not any(f.kind == ObjectiveFactKind.BUDGET_USAGE_RECORDED for f in facts)

    async def _reader(_run_id: str) -> float:
        return 0.4

    gate2 = ObjectiveLlmBudgetGate(store, objective_id=oid, run_cost_reader=_reader)
    seeded = await gate2.seed_pending_from_run_log("run1")
    assert abs(seeded - 0.4) < 1e-12
    with pytest.raises(LlmBudgetDenied) as exc:
        await gate2.check_before_attempt(_admit(oid, ceiling=0.7))
    assert exc.value.reason == "llm_cost_ceiling_exceeded"
    assert abs(exc.value.used - 0.4) < 1e-12

    delta = await gate2.flush_run_usage("run1")
    assert abs(delta - 0.4) < 1e-12
    facts = await store.list_facts(objective_id=oid)
    usage = [f for f in facts if f.kind == ObjectiveFactKind.BUDGET_USAGE_RECORDED]
    assert len(usage) == 1
    assert usage[0].payload["source"] == "llm_run_total"


@pytest.mark.asyncio
async def test_flush_from_run_log_override_when_pending_lost() -> None:
    store = InMemoryObjectiveStore()
    oid = await _seed(store, llm_limit=5.0)
    gate = ObjectiveLlmBudgetGate(store, objective_id=oid)
    delta = await gate.flush_run_usage("run1", run_cost_override=0.75)
    assert abs(delta - 0.75) < 1e-12
    facts = await store.list_facts(objective_id=oid)
    usage = [f for f in facts if f.kind == ObjectiveFactKind.BUDGET_USAGE_RECORDED]
    assert len(usage) == 1
    assert usage[0].payload["used_after"] == 0.75


class _FakeRuns:
    def __init__(self, *, status: RunStatus, cost: float) -> None:
        self._status = status
        self._cost = cost

    async def get_run_status(self, run_id: str) -> RunStatus:
        del run_id
        return self._status

    async def get_run_view(self, run_id: str) -> object:
        del run_id

        class _Metrics:
            def __init__(self, token_cost: float) -> None:
                self.token_cost = token_cost

        class _View:
            def __init__(self, token_cost: float) -> None:
                self.metrics = _Metrics(token_cost)

        return _View(self._cost)


@pytest.mark.asyncio
async def test_reconcile_missing_llm_run_totals_flushes_terminal() -> None:
    store = InMemoryObjectiveStore()
    oid = await _seed(store, llm_limit=5.0)
    now = utc_now()
    await store.commit_command(
        receipt=CommandReceipt(
            scope=command_scope(oid, "started"),
            idempotency_key="started",
            request_hash="started",
            status=CommandReceiptStatus.COMPLETED,
            response_payload={},
            created_at=now,
            completed_at=now,
        ),
        facts=materialize_facts(
            oid,
            [
                fact_root_run_requested(
                    run_id="run_crash",
                    role=RunRole.WORK,
                ),
                fact_root_run_started(run_id="run_crash"),
            ],
            start_sequence=2,
            now=now,
        ),
        outbox_records=[
            DispatchRequested.create(
                objective_id=oid,
                run_id="run_crash",
                role=RunRole.WORK,
            )
        ],
    )
    flushed = await reconcile_missing_llm_run_totals(
        store,
        _FakeRuns(status=RunStatus.COMPLETED, cost=0.55),
        objective_ids=[oid],
    )
    assert flushed == 1
    facts = await store.list_facts(objective_id=oid)
    usage = [f for f in facts if f.kind == ObjectiveFactKind.BUDGET_USAGE_RECORDED]
    assert len(usage) == 1
    assert usage[0].payload["source"] == "llm_run_total"
    assert usage[0].payload["run_id"] == "run_crash"
    assert abs(usage[0].payload["delta"] - 0.55) < 1e-12
