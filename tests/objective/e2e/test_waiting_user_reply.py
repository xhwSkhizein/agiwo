"""P6-01: WAITING_USER reply creates a fresh WORK root Run (not Run resume)."""

import pytest

from agiwo.agent import Agent
from agiwo.agent.models.config import AgentConfig
from agiwo.agent.models.input import ContentPart, ContentType, UserMessage
from agiwo.agent.models.run import RunStatus
from agiwo.llm.base import Model, StreamChunk
from agiwo.objective import (
    ObjectiveService,
    SubmitUserInputRequest,
)
from agiwo.objective.history import append_objective_user_input_to_history
from agiwo.objective.log import (
    fact_objective_created,
    fact_objective_status_changed,
    fact_objective_user_input,
    fact_root_run_requested,
    fact_root_run_started,
    fact_run_outcome,
    fact_waiting_interval_started,
    materialize_facts,
)
from agiwo.objective.models import (
    CommandReceiptStatus,
    ObjectiveBudget,
    ObjectiveStatus,
    ObjectiveUserInput,
    RunOutcome,
    RunRole,
    new_id,
    utc_now,
)
from agiwo.objective.outbox import DispatchRequested
from agiwo.objective.store.base import CommandReceipt, SlotMutation
from agiwo.objective.store.memory import InMemoryObjectiveStore
from agiwo.objective.store.sqlite import SQLiteObjectiveStore
from agiwo.scheduler import Scheduler

from tests.objective.e2e.invariants import assert_objective_invariants


class _StubModel(Model):
    def __init__(self) -> None:
        super().__init__(id="stub", name="stub", temperature=0.0)

    async def arun_stream(self, messages, tools=None):
        del messages, tools
        yield StreamChunk(content="ok")
        yield StreamChunk(finish_reason="stop")


def _user(text: str) -> UserMessage:
    return UserMessage(
        content=[ContentPart(type=ContentType.TEXT, text=text)],
        is_user_provided=True,
    )


async def _seed_waiting_user(store, *, session_id: str, objective_id: str) -> None:
    budget = ObjectiveBudget.create(
        handoffs=10,
        verification_attempts=5,
        llm_cost_usd=5.0,
        active_seconds=600,
    )
    now = utc_now()
    facts = materialize_facts(
        objective_id,
        [
            fact_objective_created(session_id=session_id, budget=budget),
            fact_objective_user_input(
                user_input=ObjectiveUserInput(
                    input_id="inp_seed",
                    message=_user("goal"),
                    created_at=now,
                ),
            ),
            fact_root_run_requested(run_id="run_seed", role=RunRole.WORK),
            fact_root_run_started(run_id="run_seed"),
            fact_run_outcome(
                outcome=RunOutcome(
                    run_id="run_seed",
                    role=RunRole.WORK,
                    terminal_status=RunStatus.COMPLETED,
                    reason="needs_user",
                    report="please clarify",
                    outcome_id="out_seed",
                ),
            ),
            fact_waiting_interval_started(
                reason="expects_user_reply",
                started_at=now,
            ),
            fact_objective_status_changed(
                from_status=ObjectiveStatus.RUNNING,
                to_status=ObjectiveStatus.WAITING_USER,
                reason="expects_user_reply",
            ),
        ],
        now=now,
    )
    seed_dispatch = DispatchRequested.create(
        objective_id=objective_id,
        run_id="run_seed",
        role=RunRole.WORK,
    )
    await store.commit_command(
        receipt=CommandReceipt(
            scope="test",
            idempotency_key=f"seed-{objective_id}",
            request_hash="seed",
            status=CommandReceiptStatus.COMPLETED,
            response_payload={},
            created_at=now,
            completed_at=now,
        ),
        facts=facts,
        slot_mutation=SlotMutation(
            action="acquire",
            session_id=session_id,
            objective_id=objective_id,
            acquired_at=now,
        ),
        outbox_records=[seed_dispatch],
    )
    # Prior root Run already has an Outcome; close its outbox so reconcile
    # does not treat the seed dispatch as still open.
    claimed = await store.claim_dispatch(owner="seed", lease_seconds=60)
    assert claimed is not None
    await store.complete_dispatch(
        dispatch_id=claimed.dispatch_id,
        owner="seed",
        status="completed",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_waiting_user_reply_creates_fresh_work_root_run(
    backend: str, tmp_path
) -> None:
    if backend == "sqlite":
        store: InMemoryObjectiveStore | SQLiteObjectiveStore = SQLiteObjectiveStore(
            str(tmp_path / "waiting.db")
        )
        await store.connect()
    else:
        store = InMemoryObjectiveStore()

    session_id = "sess-waiting"
    objective_id = "obj_waiting"
    await _seed_waiting_user(store, session_id=session_id, objective_id=objective_id)

    scheduler = Scheduler()
    await scheduler.start()
    agent = Agent(
        AgentConfig(name="waiting", description="waiting"),
        model=_StubModel(),
        id=session_id,
    )

    async def provider(_sid: str) -> Agent:
        return agent

    seed_input = ObjectiveUserInput(input_id="inp_seed", message=_user("goal"))
    await append_objective_user_input_to_history(
        agent,
        session_id=session_id,
        user_input=seed_input,
        objective_id=objective_id,
    )

    service = ObjectiveService(
        store,
        scheduler=scheduler,
        default_agent_provider=provider,
    )
    try:
        result = await service.submit_user_input(
            SubmitUserInputRequest(
                objective_id=objective_id,
                user_message=_user("here is the clarification"),
                idempotency_key=new_id(),
            )
        )
        assert result.payload["continued_root_run"] is True
        assert result.payload["run_id"]
        assert result.payload["run_id"] != "run_seed"
        assert result.status == ObjectiveStatus.RUNNING.value

        view = await service.get_view(objective_id)
        assert view is not None
        assert view.status is ObjectiveStatus.RUNNING
        assert_objective_invariants(view)
        work = [r for r in view.root_runs if r.role is RunRole.WORK]
        assert len(work) == 2
        assert view.active_root_run is not None
        assert view.active_root_run.run_id == result.payload["run_id"]

        pending = await store.list_pending_dispatches(objective_id=objective_id)
        assert len(pending) == 1
        assert pending[0].run_id == result.payload["run_id"]
        assert pending[0].role is RunRole.WORK
    finally:
        await scheduler.stop()
        await agent.close()
        if isinstance(store, SQLiteObjectiveStore):
            await store.close()
