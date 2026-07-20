"""P6-01: crash injection around outbox claim / startup reconcile."""

import pytest

from agiwo.agent import RunStatus
from agiwo.agent.models.run import RunStatus as AgentRunStatus
from agiwo.objective.log import (
    fact_objective_created,
    fact_root_run_requested,
    fact_run_outcome,
    materialize_facts,
)
from agiwo.objective.models import (
    CommandReceiptStatus,
    DispatchStatus,
    ObjectiveBudget,
    RunOutcome,
    RunRole,
    utc_now,
)
from agiwo.objective.outbox import DispatchRequested
from agiwo.objective.recovery import RecoveryKind, reconcile_startup
from agiwo.objective.store.base import CommandReceipt
from agiwo.objective.store.memory import InMemoryObjectiveStore
from agiwo.objective.store.sqlite import SQLiteObjectiveStore


class _FakeRuns:
    def __init__(self, statuses: dict[str, RunStatus | None]) -> None:
        self._statuses = statuses

    async def get_run_status(self, run_id: str) -> RunStatus | None:
        return self._statuses.get(run_id)

    async def get_run_view(self, run_id: str):
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_claim_then_restart_redispatches_without_duplicate_run(
    backend: str, tmp_path
) -> None:
    if backend == "sqlite":
        store: InMemoryObjectiveStore | SQLiteObjectiveStore = SQLiteObjectiveStore(
            str(tmp_path / "recovery.db")
        )
        await store.connect()
    else:
        store = InMemoryObjectiveStore()

    record = DispatchRequested.create(
        objective_id="obj_crash",
        run_id="run_crash_1",
        role=RunRole.WORK,
    )
    # Seed pending outbox (test-only direct insert for crash boundary).
    if isinstance(store, InMemoryObjectiveStore):
        store._outbox[record.dispatch_id] = record  # noqa: SLF001
    else:
        await store.commit_command(
            receipt=CommandReceipt(
                scope="test",
                idempotency_key=f"seed-{record.dispatch_id}",
                request_hash="seed",
                status=CommandReceiptStatus.COMPLETED,
                response_payload={},
                created_at=utc_now(),
                completed_at=utc_now(),
            ),
            facts=[],
            outbox_records=[record],
        )

    claimed = await store.claim_dispatch(owner="owner-crash", lease_seconds=60)
    assert claimed is not None
    assert claimed.status is DispatchStatus.CLAIMED

    # Process "dies": rebuild store handle for sqlite, keep memory object.
    if backend == "sqlite":
        await store.close()
        store = SQLiteObjectiveStore(str(tmp_path / "recovery.db"))
        await store.connect()

    report = await reconcile_startup(
        store,
        runs=_FakeRuns({}),  # runtime gone → redispatch
        apply=True,
        owner="owner-crash",
    )
    assert report.ok
    assert any(a.kind is RecoveryKind.REDISPATCH for a in report.actions)

    pending = await store.list_pending_dispatches(objective_id="obj_crash")
    assert len(pending) == 1
    assert pending[0].status is DispatchStatus.PENDING
    assert pending[0].run_id == "run_crash_1"
    # Still one outbox record — no duplicate dispatch_id/run_id invented.
    assert pending[0].dispatch_id == record.dispatch_id

    if isinstance(store, SQLiteObjectiveStore):
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_outcome_present_completes_outbox_on_restart(
    backend: str, tmp_path
) -> None:
    """Crash after Outcome commit but before outbox complete → COMPLETE_OUTBOX."""
    if backend == "sqlite":
        store: InMemoryObjectiveStore | SQLiteObjectiveStore = SQLiteObjectiveStore(
            str(tmp_path / "outcome.db")
        )
        await store.connect()
    else:
        store = InMemoryObjectiveStore()

    now = utc_now()
    objective_id = "obj_outcome"
    seed = DispatchRequested.create(
        objective_id=objective_id,
        run_id="run_out",
        role=RunRole.WORK,
    )
    facts = materialize_facts(
        objective_id,
        [
            fact_objective_created(
                session_id="sess-out",
                budget=ObjectiveBudget.create(
                    handoffs=5,
                    verification_attempts=3,
                    llm_cost_usd=1.0,
                    active_seconds=600,
                ),
            ),
            fact_root_run_requested(run_id="run_out", role=RunRole.WORK),
            fact_run_outcome(
                outcome=RunOutcome(
                    run_id="run_out",
                    role=RunRole.WORK,
                    terminal_status=AgentRunStatus.COMPLETED,
                    reason="done",
                    report="report",
                    outcome_id="out_1",
                ),
            ),
        ],
        now=now,
    )
    await store.commit_command(
        receipt=CommandReceipt(
            scope="test",
            idempotency_key=f"seed-outcome-{backend}",
            request_hash="seed",
            status=CommandReceiptStatus.COMPLETED,
            response_payload={},
            created_at=now,
            completed_at=now,
        ),
        facts=facts,
        outbox_records=[seed],
    )
    claimed = await store.claim_dispatch(owner="owner-out", lease_seconds=60)
    assert claimed is not None
    # Dispatcher reached "dispatched" but never marked completed (crash gap).
    await store.complete_dispatch(
        dispatch_id=claimed.dispatch_id,
        owner="owner-out",
        status="dispatched",
    )

    if backend == "sqlite":
        await store.close()
        store = SQLiteObjectiveStore(str(tmp_path / "outcome.db"))
        await store.connect()

    report = await reconcile_startup(
        store,
        runs=_FakeRuns({"run_out": RunStatus.COMPLETED}),
        apply=True,
        owner="owner-out",
    )
    assert report.ok
    assert any(a.kind is RecoveryKind.COMPLETE_OUTBOX for a in report.actions)
    pending = await store.list_pending_dispatches(objective_id=objective_id)
    assert pending == []

    if isinstance(store, SQLiteObjectiveStore):
        await store.close()
