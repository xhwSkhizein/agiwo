"""Regression: shared SQLite connection must not nest BEGIN IMMEDIATE."""

import asyncio

import pytest

from agiwo.agent.models.log import RunStarted
from agiwo.agent.storage.sqlite import SQLiteRunLogStorage
from agiwo.objective.log import fact_objective_created
from agiwo.objective.models import (
    CommandReceiptStatus,
    ObjectiveBudget,
    RunRole,
    new_id,
)
from agiwo.objective.outbox import DispatchRequested
from agiwo.objective.store.base import CommandReceipt, SlotMutation
from agiwo.objective.store.sqlite import SQLiteObjectiveStore
from agiwo.utils.sqlite_pool import get_shared_connection, reset_sqlite_pool
from agiwo.utils.storage_support.sqlite_runtime import begin_immediate


@pytest.fixture(autouse=True)
def _reset_pool():
    reset_sqlite_pool()
    yield
    reset_sqlite_pool()


@pytest.mark.asyncio
async def test_concurrent_objective_commit_and_run_log_allocate(tmp_path) -> None:
    db = str(tmp_path / "shared.db")
    objective = SQLiteObjectiveStore(db_path=db)
    run_log = SQLiteRunLogStorage(db_path=db)
    await objective.connect()
    await run_log.connect()

    budget = ObjectiveBudget.create(
        handoffs=3, verification_attempts=2, llm_cost_usd=1.0, active_seconds=60
    )

    async def commit_many() -> None:
        for i in range(20):
            oid = f"obj_{i}"
            await objective.commit_command(
                receipt=CommandReceipt(
                    scope="scope",
                    idempotency_key=new_id(),
                    request_hash=f"h{i}",
                    status=CommandReceiptStatus.COMPLETED,
                    response_payload={},
                ),
                facts=[
                    fact_objective_created(
                        session_id=f"sess_{i}",
                        budget=budget,
                    ).materialize(objective_id=oid, sequence=1)
                ],
                slot_mutation=SlotMutation(
                    action="acquire",
                    session_id=f"sess_{i}",
                    objective_id=oid,
                ),
                outbox_records=[
                    DispatchRequested.create(
                        objective_id=oid,
                        run_id=f"run_{i}",
                        role=RunRole.WORK,
                    )
                ],
            )

    async def allocate_many() -> None:
        for i in range(40):
            seq = await run_log.allocate_sequence("shared-session")
            assert seq == i + 1

    await asyncio.gather(commit_many(), allocate_many())

    # Also exercise append under contention with a second session's allocates.
    async def append_started() -> None:
        for i in range(10):
            seq = await run_log.allocate_sequence("append-session")
            await run_log.append_entries(
                [
                    RunStarted(
                        sequence=seq,
                        session_id="append-session",
                        run_id=f"run_append_{i}",
                        agent_id="agent",
                    )
                ]
            )

    async def allocate_other() -> None:
        for i in range(20):
            seq = await run_log.allocate_sequence("other-session")
            assert seq == i + 1

    await asyncio.gather(append_started(), allocate_other())

    await objective.close()
    await run_log.close()


@pytest.mark.asyncio
async def test_begin_immediate_recovers_from_stale_transaction(tmp_path) -> None:
    reset_sqlite_pool()
    db = str(tmp_path / "stale.db")
    conn = await get_shared_connection(db)
    await conn.execute("BEGIN IMMEDIATE")
    assert conn.in_transaction
    # Nested BEGIN would fail; helper must roll back and reopen cleanly.
    await begin_immediate(conn)
    assert conn.in_transaction
    await conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER)")
    await conn.commit()
    reset_sqlite_pool()
