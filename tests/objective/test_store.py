"""P1-03 memory ObjectiveStore contract tests."""

import asyncio
from datetime import timedelta

import pytest

from agiwo.objective.errors import IdempotencyConflict, StoreError
from agiwo.objective.log import (
    fact_objective_created,
    fact_objective_status_changed,
    fact_root_run_requested,
    materialize_facts,
)
from agiwo.objective.models import (
    CommandReceiptStatus,
    ObjectiveBudget,
    ObjectiveStatus,
    RunRole,
    utc_now,
)
from agiwo.objective.outbox import DispatchRequested
from agiwo.objective.store.base import CommandReceipt, SlotMutation
from agiwo.objective.store.memory import InMemoryObjectiveStore


def _budget() -> ObjectiveBudget:
    return ObjectiveBudget.create(
        handoffs=3, verification_attempts=2, llm_cost_usd=1.0, active_seconds=60
    )


def _receipt(
    scope: str, key: str, h: str, payload: dict | None = None
) -> CommandReceipt:
    return CommandReceipt(
        scope=scope,
        idempotency_key=key,
        request_hash=h,
        status=CommandReceiptStatus.COMPLETED,
        response_payload=payload or {"ok": True},
    )


@pytest.mark.asyncio
async def test_facts_and_outbox_atomic_success() -> None:
    store = InMemoryObjectiveStore()
    oid = "obj1"
    fact = fact_objective_created(session_id="s1", budget=_budget()).materialize(
        objective_id=oid, sequence=1
    )
    dispatch = DispatchRequested.create(
        objective_id=oid,
        run_id="r1",
        role=RunRole.WORK,
    )
    await store.commit_command(
        receipt=_receipt("scope", "k1", "h1"),
        facts=[fact],
        slot_mutation=SlotMutation(action="acquire", session_id="s1", objective_id=oid),
        outbox_records=[dispatch],
    )
    facts = await store.list_facts(objective_id=oid)
    assert len(facts) == 1
    pending = await store.list_pending_dispatches(objective_id=oid)
    assert len(pending) == 1
    slot = await store.get_session_slot("s1")
    assert slot is not None
    assert slot.objective_id == oid


@pytest.mark.asyncio
async def test_atomic_rollback_on_duplicate_sequence() -> None:
    store = InMemoryObjectiveStore()
    oid = "obj1"
    f1 = fact_objective_created(session_id="s1", budget=_budget()).materialize(
        objective_id=oid, sequence=1
    )
    await store.commit_command(
        receipt=_receipt("scope", "k1", "h1"),
        facts=[f1],
        slot_mutation=SlotMutation(action="acquire", session_id="s1", objective_id=oid),
    )
    f2 = fact_objective_created(session_id="s1", budget=_budget()).materialize(
        objective_id=oid, sequence=1
    )
    with pytest.raises(StoreError):
        await store.commit_command(
            receipt=_receipt("scope", "k2", "h2"),
            facts=[f2],
        )
    assert await store.get_max_sequence(oid) == 1


@pytest.mark.asyncio
async def test_session_slot_unique_concurrent_create() -> None:
    store = InMemoryObjectiveStore()

    async def create(oid: str, key: str) -> str | None:
        fact = fact_objective_created(session_id="s1", budget=_budget()).materialize(
            objective_id=oid, sequence=1
        )
        try:
            await store.commit_command(
                receipt=_receipt("session:s1:objective:create", key, "hash-" + oid),
                facts=[fact],
                slot_mutation=SlotMutation(
                    action="acquire", session_id="s1", objective_id=oid
                ),
            )
            return oid
        except StoreError:
            return None

    results = await asyncio.gather(create("o1", "k1"), create("o2", "k2"))
    winners = [r for r in results if r is not None]
    assert len(winners) == 1
    slot = await store.get_session_slot("s1")
    assert slot is not None
    assert slot.objective_id == winners[0]


@pytest.mark.asyncio
async def test_terminal_release_allows_next_objective() -> None:
    store = InMemoryObjectiveStore()
    await store.commit_command(
        receipt=_receipt("scope", "k1", "h1"),
        facts=materialize_facts(
            "o1",
            [fact_objective_created(session_id="s1", budget=_budget())],
        ),
        slot_mutation=SlotMutation(
            action="acquire", session_id="s1", objective_id="o1"
        ),
    )
    await store.commit_command(
        receipt=_receipt("scope", "k2", "h2"),
        facts=materialize_facts(
            "o1",
            [
                fact_objective_status_changed(
                    from_status=ObjectiveStatus.CREATED,
                    to_status=ObjectiveStatus.FAILED,
                    reason="abandoned",
                )
            ],
            start_sequence=2,
        ),
        slot_mutation=SlotMutation(
            action="release", session_id="s1", objective_id="o1"
        ),
    )
    assert await store.get_session_slot("s1") is None
    await store.commit_command(
        receipt=_receipt("scope", "k3", "h3"),
        facts=materialize_facts(
            "o2",
            [fact_objective_created(session_id="s1", budget=_budget())],
        ),
        slot_mutation=SlotMutation(
            action="acquire", session_id="s1", objective_id="o2"
        ),
    )
    slot = await store.get_session_slot("s1")
    assert slot is not None
    assert slot.objective_id == "o2"


@pytest.mark.asyncio
async def test_non_terminal_slot_release_rejected() -> None:
    store = InMemoryObjectiveStore()
    await store.commit_command(
        receipt=_receipt("scope", "k1", "h1"),
        facts=materialize_facts(
            "o1",
            [fact_objective_created(session_id="s1", budget=_budget())],
        ),
        slot_mutation=SlotMutation(
            action="acquire", session_id="s1", objective_id="o1"
        ),
    )
    with pytest.raises(StoreError, match="non-terminal"):
        await store.commit_command(
            receipt=_receipt("scope", "k2", "h2"),
            facts=[],
            slot_mutation=SlotMutation(
                action="release", session_id="s1", objective_id="o1"
            ),
        )
    assert await store.get_session_slot("s1") is not None


@pytest.mark.asyncio
async def test_root_run_requested_requires_dispatch_outbox() -> None:
    store = InMemoryObjectiveStore()
    await store.commit_command(
        receipt=_receipt("scope", "k1", "h1"),
        facts=materialize_facts(
            "o1",
            [fact_objective_created(session_id="s1", budget=_budget())],
        ),
        slot_mutation=SlotMutation(
            action="acquire", session_id="s1", objective_id="o1"
        ),
    )
    with pytest.raises(StoreError, match="DispatchRequested"):
        await store.commit_command(
            receipt=_receipt("scope", "k2", "h2"),
            facts=materialize_facts(
                "o1",
                [fact_root_run_requested(run_id="r1", role=RunRole.WORK)],
                start_sequence=2,
            ),
        )
    dispatch = DispatchRequested.create(
        objective_id="o1",
        run_id="r1",
        role=RunRole.WORK,
    )
    await store.commit_command(
        receipt=_receipt("scope", "k3", "h3"),
        facts=materialize_facts(
            "o1",
            [fact_root_run_requested(run_id="r1", role=RunRole.WORK)],
            start_sequence=2,
        ),
        outbox_records=[dispatch],
    )
    pending = await store.list_pending_dispatches(objective_id="o1")
    assert len(pending) == 1


@pytest.mark.asyncio
async def test_receipt_replay_and_conflict() -> None:
    store = InMemoryObjectiveStore()
    r1 = await store.commit_command(
        receipt=_receipt("scope", "key", "hash-a", {"v": 1}),
        facts=[],
    )
    r2 = await store.commit_command(
        receipt=_receipt("scope", "key", "hash-a", {"v": 999}),
        facts=[],
    )
    assert r2.response_payload == r1.response_payload
    with pytest.raises(IdempotencyConflict):
        await store.commit_command(
            receipt=_receipt("scope", "key", "hash-b"),
            facts=[],
        )


@pytest.mark.asyncio
async def test_outbox_claim_lease_renew_complete() -> None:
    store = InMemoryObjectiveStore()
    dispatch = DispatchRequested.create(
        objective_id="o1",
        run_id="r1",
        role=RunRole.WORK,
    )
    await store.commit_command(
        receipt=_receipt("scope", "k1", "h1"),
        facts=materialize_facts(
            "o1",
            [fact_objective_created(session_id="s1", budget=_budget())],
        ),
        outbox_records=[dispatch],
    )
    claimed = await store.claim_dispatch(owner="worker-1", lease_seconds=30)
    assert claimed is not None
    assert claimed.lease_owner == "worker-1"
    # second claim while lease active
    none = await store.claim_dispatch(owner="worker-2", lease_seconds=30)
    assert none is None
    renewed = await store.renew_dispatch_lease(
        dispatch_id=claimed.dispatch_id, owner="worker-1", lease_seconds=60
    )
    assert renewed.lease_owner == "worker-1"
    with pytest.raises(StoreError):
        await store.renew_dispatch_lease(
            dispatch_id=claimed.dispatch_id, owner="worker-2"
        )
    completed = await store.complete_dispatch(
        dispatch_id=claimed.dispatch_id, owner="worker-1", status="dispatched"
    )
    assert completed.status.value == "dispatched"
    pending = await store.list_pending_dispatches()
    assert pending == []


@pytest.mark.asyncio
async def test_lease_expiry_allows_reclaim() -> None:
    store = InMemoryObjectiveStore()
    dispatch = DispatchRequested.create(
        objective_id="o1",
        run_id="r1",
        role=RunRole.WORK,
    )
    await store.commit_command(
        receipt=_receipt("scope", "k1", "h1"),
        facts=materialize_facts(
            "o1",
            [fact_objective_created(session_id="s1", budget=_budget())],
        ),
        outbox_records=[dispatch],
    )
    t0 = utc_now()
    claimed = await store.claim_dispatch(owner="w1", lease_seconds=1, now=t0)
    assert claimed is not None

    later = t0 + timedelta(seconds=2)
    reclaimed = await store.claim_dispatch(owner="w2", lease_seconds=10, now=later)
    assert reclaimed is not None
    assert reclaimed.lease_owner == "w2"
    assert reclaimed.attempt == claimed.attempt + 1
