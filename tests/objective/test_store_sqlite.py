"""P1-03 SQLite ObjectiveStore tests."""

import asyncio

import pytest

from agiwo.agent.models.config import RunLogStorageConfig
from agiwo.objective.errors import StoreError, UnsupportedStorageBackend

from agiwo.objective.log import fact_objective_created, materialize_facts
from agiwo.objective.models import (
    CommandReceiptStatus,
    ObjectiveBudget,
    RunRole,
)
from agiwo.objective.outbox import DispatchRequested
from agiwo.objective.store.base import CommandReceipt, SlotMutation
from agiwo.objective.store.factory import create_objective_store
from agiwo.objective.store.sqlite import SQLiteObjectiveStore


def _budget() -> ObjectiveBudget:
    return ObjectiveBudget.create(
        handoffs=3, verification_attempts=2, llm_cost_usd=1.0, active_seconds=60
    )


@pytest.mark.asyncio
async def test_sqlite_persist_and_reopen(tmp_path) -> None:
    db = str(tmp_path / "obj.db")
    store = SQLiteObjectiveStore(db_path=db)
    await store.connect()
    fact = fact_objective_created(session_id="s1", budget=_budget()).materialize(
        objective_id="o1", sequence=1
    )
    dispatch = DispatchRequested.create(
        objective_id="o1",
        run_id="r1",
        role=RunRole.WORK,
    )
    await store.commit_command(
        receipt=CommandReceipt(
            scope="scope",
            idempotency_key="k1",
            request_hash="h1",
            status=CommandReceiptStatus.COMPLETED,
            response_payload={"ok": True},
        ),
        facts=[fact],
        slot_mutation=SlotMutation(
            action="acquire", session_id="s1", objective_id="o1"
        ),
        outbox_records=[dispatch],
    )
    await store.close()

    store2 = SQLiteObjectiveStore(db_path=db)
    await store2.connect()
    facts = await store2.list_facts(objective_id="o1")
    assert len(facts) == 1
    pending = await store2.list_pending_dispatches(objective_id="o1")
    assert len(pending) == 1
    slot = await store2.get_session_slot("s1")
    assert slot is not None
    receipt = await store2.get_receipt(scope="scope", idempotency_key="k1")
    assert receipt is not None
    await store2.close()


@pytest.mark.asyncio
async def test_sqlite_shares_db_file_with_run_log_schema(tmp_path) -> None:
    from agiwo.agent.storage.sqlite import SQLiteRunLogStorage  # noqa: PLC0415

    db = str(tmp_path / "shared.db")
    run_log = SQLiteRunLogStorage(db_path=db)
    await run_log.connect()
    obj = SQLiteObjectiveStore(db_path=db)
    await obj.connect()
    # Independent tables: objective insert does not touch run_log
    await obj.commit_command(
        receipt=CommandReceipt(
            scope="s",
            idempotency_key="k",
            request_hash="h",
            status=CommandReceiptStatus.COMPLETED,
            response_payload={},
        ),
        facts=materialize_facts(
            "o1",
            [fact_objective_created(session_id="s1", budget=_budget())],
        ),
    )
    assert await run_log.get_max_sequence("s1") == 0
    assert await obj.get_max_sequence("o1") == 1
    await obj.close()
    await run_log.close()


def test_factory_fail_closed_for_mongodb() -> None:
    with pytest.raises(UnsupportedStorageBackend):
        create_objective_store(RunLogStorageConfig(storage_type="mongodb", config={}))


def test_factory_memory_and_sqlite(tmp_path) -> None:
    mem = create_objective_store(RunLogStorageConfig(storage_type="memory"))
    assert mem is not None
    sql = create_objective_store(
        RunLogStorageConfig(
            storage_type="sqlite",
            config={"db_path": str(tmp_path / "x.db")},
        )
    )
    assert isinstance(sql, SQLiteObjectiveStore)


@pytest.mark.asyncio
async def test_sqlite_concurrent_session_slot(tmp_path) -> None:

    db = str(tmp_path / "slot.db")
    store = SQLiteObjectiveStore(db_path=db)
    await store.connect()

    async def create(oid: str, key: str):
        try:
            await store.commit_command(
                receipt=CommandReceipt(
                    scope="session:s1:objective:create",
                    idempotency_key=key,
                    request_hash="hash-" + oid,
                    status=CommandReceiptStatus.COMPLETED,
                    response_payload={"objective_id": oid},
                ),
                facts=[
                    fact_objective_created(
                        session_id="s1", budget=_budget()
                    ).materialize(objective_id=oid, sequence=1)
                ],
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
    await store.close()


@pytest.mark.asyncio
async def test_sqlite_receipt_replay_across_reopen(tmp_path) -> None:
    db = str(tmp_path / "rcpt.db")
    store = SQLiteObjectiveStore(db_path=db)
    await store.connect()
    await store.commit_command(
        receipt=CommandReceipt(
            scope="session:s1:objective:create",
            idempotency_key="k1",
            request_hash="hash-a",
            status=CommandReceiptStatus.COMPLETED,
            response_payload={"objective_id": "o1", "status": "CREATED"},
        ),
        facts=materialize_facts(
            "o1",
            [fact_objective_created(session_id="s1", budget=_budget())],
        ),
        slot_mutation=SlotMutation(
            action="acquire", session_id="s1", objective_id="o1"
        ),
    )
    await store.close()

    store2 = SQLiteObjectiveStore(db_path=db)
    await store2.connect()
    again = await store2.commit_command(
        receipt=CommandReceipt(
            scope="session:s1:objective:create",
            idempotency_key="k1",
            request_hash="hash-a",
            status=CommandReceiptStatus.COMPLETED,
            response_payload={"objective_id": "SHOULD_NOT_WIN"},
        ),
        facts=[],
    )
    assert again.response_payload["objective_id"] == "o1"
    assert (
        isinstance(again.status, type(CommandReceiptStatus.COMPLETED))
        or again.status == CommandReceiptStatus.COMPLETED
    )
    await store2.close()
