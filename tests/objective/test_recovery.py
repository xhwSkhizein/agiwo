"""P3-06: startup outbox / Run reconciliation."""

import pytest

from agiwo.agent import RunStatus
from agiwo.objective.models import DispatchStatus, RunRole
from agiwo.objective.outbox import DispatchRequested
from agiwo.objective.recovery import RecoveryKind, classify_dispatch, reconcile_startup
from agiwo.objective.store.memory import InMemoryObjectiveStore


class _FakeRuns:
    def __init__(self, statuses: dict[str, RunStatus | None]) -> None:
        self._statuses = statuses

    async def get_run_status(self, run_id: str) -> RunStatus | None:
        return self._statuses.get(run_id)

    async def get_run_view(self, run_id: str):
        return None


@pytest.mark.asyncio
async def test_classify_no_run_redispatch() -> None:
    store = InMemoryObjectiveStore()
    record = DispatchRequested.create(
        objective_id="o1",
        run_id="r1",
        role=RunRole.WORK,
    )
    # Seed via commit is heavy; put directly for unit classify.
    store._outbox[record.dispatch_id] = record  # noqa: SLF001
    action = await classify_dispatch(store, record, _FakeRuns({}))
    assert action.kind is RecoveryKind.REDISPATCH


@pytest.mark.asyncio
async def test_classify_paused_waits() -> None:
    store = InMemoryObjectiveStore()
    record = DispatchRequested.create(
        objective_id="o1",
        run_id="r1",
        role=RunRole.WORK,
    ).with_updates(status=DispatchStatus.DISPATCHED)
    store._outbox[record.dispatch_id] = record  # noqa: SLF001
    action = await classify_dispatch(store, record, _FakeRuns({"r1": RunStatus.PAUSED}))
    assert action.kind is RecoveryKind.WAIT_PAUSED


@pytest.mark.asyncio
async def test_reconcile_startup_releases_claimed() -> None:
    store = InMemoryObjectiveStore()
    record = DispatchRequested.create(
        objective_id="o1",
        run_id="r1",
        role=RunRole.WORK,
    )
    # Claim it.
    store._outbox[record.dispatch_id] = record  # noqa: SLF001
    claimed = await store.claim_dispatch(owner="owner-a", lease_seconds=60)
    assert claimed is not None
    report = await reconcile_startup(
        store, runs=_FakeRuns({}), apply=True, owner="owner-a"
    )
    assert report.ok
    assert any(a.kind is RecoveryKind.REDISPATCH for a in report.actions)
    pending = await store.list_pending_dispatches(objective_id="o1")
    assert len(pending) == 1
    assert pending[0].status is DispatchStatus.PENDING
