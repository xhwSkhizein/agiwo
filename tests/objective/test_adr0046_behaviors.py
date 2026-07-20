"""ADR 0046 frozen behaviors: latch, complexity, history, thin input."""

import asyncio
import json
from collections.abc import AsyncIterator

import pytest

from agiwo.agent import Agent
from agiwo.agent.models.config import AgentConfig
from agiwo.agent.models.input import ContentPart, ContentType, UserMessage
from agiwo.llm.base import Model, StreamChunk
from agiwo.objective import (
    BudgetLimits,
    CreateObjectiveRequest,
    ObjectiveService,
)
from agiwo.objective.complexity import (
    COMPLEXITY_PLANNING_THRESHOLD,
    assess_entry_complexity,
    planning_notice_for_score,
)
from agiwo.objective.dispatch import OutboxDispatcher
from agiwo.objective.finalization import should_deliver
from agiwo.objective.history import (
    append_objective_user_input_to_history,
    collect_history_input_ids_from_agent,
)
from agiwo.objective.input import render_run_input
from agiwo.objective.log import (
    fact_entry_complexity_assessed,
    fact_objective_created,
    fact_objective_user_input,
    fact_verification_required_set,
    materialize_facts,
)
from agiwo.objective.models import (
    CommandReceiptStatus,
    DispatchStatus,
    HandoffDecision,
    HandoffTarget,
    ObjectiveBudget,
    ObjectiveStatus as ObjStatus,
    ObjectiveUserInput,
    RunRole,
    new_id,
)
from agiwo.objective.outbox import DispatchRequested
from agiwo.objective.plan_latch import maybe_commit_verification_required
from agiwo.objective.projection import ObjectiveView, project_objective
from agiwo.objective.store.base import CommandReceipt
from agiwo.objective.store.memory import InMemoryObjectiveStore
from agiwo.scheduler import Scheduler


def _user(text: str = "help me") -> UserMessage:
    return UserMessage(
        content=[ContentPart(type=ContentType.TEXT, text=text)],
        is_user_provided=True,
    )


def _budget() -> BudgetLimits:
    return BudgetLimits(
        handoffs=5,
        verification_attempts=3,
        llm_cost_usd=2.0,
        active_seconds=600,
    )


def _view(*, verification_required: bool = False) -> ObjectiveView:
    return ObjectiveView(
        objective_id="obj_1",
        session_id="sess_1",
        status=ObjStatus.RUNNING,
        budget=ObjectiveBudget.create(
            handoffs=5,
            verification_attempts=3,
            llm_cost_usd=1.0,
            active_seconds=600,
        ),
        verification_required=verification_required,
    )


class _ScoreModel(Model):
    def __init__(self, score: int) -> None:
        super().__init__(id="score", name="score", temperature=0.0)
        self._score = score

    async def arun_stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        del messages, tools
        yield StreamChunk(content=json.dumps({"score": self._score}))
        yield StreamChunk(finish_reason="stop")


class _StubModel(Model):
    def __init__(self) -> None:
        super().__init__(id="stub", name="stub", temperature=0.0)

    async def arun_stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        del messages, tools
        yield StreamChunk(content="ok")
        yield StreamChunk(finish_reason="stop")


def test_verification_required_latch_projects_true() -> None:
    budget = ObjectiveBudget.create(
        handoffs=1, verification_attempts=1, llm_cost_usd=1, active_seconds=10
    )
    facts = materialize_facts(
        "o1",
        [
            fact_objective_created(session_id="s1", budget=budget),
            fact_verification_required_set(run_id="r1", milestone_id="m1"),
        ],
    )
    view = project_objective(facts)
    assert view is not None
    assert view.verification_required is True


def test_should_deliver_false_when_verification_required() -> None:
    decision = HandoffDecision(target=HandoffTarget.USER, expects_reply=False)
    assert not should_deliver(
        decision,
        role=RunRole.WORK,
        verification_required=True,
    )


@pytest.mark.asyncio
async def test_complexity_notice_only_when_score_above_threshold() -> None:
    low = await assess_entry_complexity("simple", model=_ScoreModel(2))
    high = await assess_entry_complexity("complex", model=_ScoreModel(7))
    assert low == 2
    assert high == 7
    assert (
        planning_notice_for_score(low, threshold=COMPLEXITY_PLANNING_THRESHOLD) is None
    )
    notice = planning_notice_for_score(high, threshold=COMPLEXITY_PLANNING_THRESHOLD)
    assert notice is not None
    assert "update_plan" in notice


@pytest.mark.asyncio
async def test_complexity_assessment_skipped_without_model() -> None:
    assert await assess_entry_complexity("anything", model=None) is None


def test_thin_run_input_without_verification_required() -> None:
    view = ObjectiveView(
        objective_id="obj_1",
        session_id="sess_1",
        status=ObjStatus.CREATED,
        budget=ObjectiveBudget.create(
            handoffs=1, verification_attempts=1, llm_cost_usd=1, active_seconds=10
        ),
        user_inputs=(
            ObjectiveUserInput(
                input_id="inp_1",
                message=_user("hello thin"),
            ),
        ),
    )
    message, template, digest = render_run_input(
        view,
        role=RunRole.WORK,
        run_id="run_1",
        thin=True,
    )
    assert template == ""
    assert digest == ""
    assert "hello thin" in message.extract_text()
    assert "Run boundary" not in message.extract_text()


@pytest.mark.asyncio
async def test_create_writes_history_before_commit() -> None:
    store = InMemoryObjectiveStore()
    scheduler = Scheduler()
    await scheduler.start()
    agent = Agent(
        AgentConfig(name="t", description="t"), model=_StubModel(), id="sess_hist"
    )

    async def provider(session_id: str) -> Agent:
        del session_id
        return agent

    service = ObjectiveService(
        store,
        scheduler=scheduler,
        default_agent_provider=provider,
        complexity_model=_ScoreModel(2),
    )
    try:
        result = await service.create_objective(
            CreateObjectiveRequest(
                session_id="sess_hist",
                user_message=_user("seed history"),
                budget=_budget(),
                idempotency_key=new_id(),
            )
        )
        history_ids = await collect_history_input_ids_from_agent(
            agent,
            session_id="sess_hist",
        )
        view = await service.get_view(result.objective_id)
        assert view is not None
        assert view.user_inputs[0].input_id in history_ids
        assert view.entry_complexity_score == 2
    finally:
        await scheduler.stop()
        await agent.close()


@pytest.mark.asyncio
async def test_create_fails_when_agent_unavailable_with_scheduler() -> None:
    store = InMemoryObjectiveStore()
    scheduler = Scheduler()
    await scheduler.start()

    async def provider(_session_id: str) -> Agent | None:
        return None

    service = ObjectiveService(
        store,
        scheduler=scheduler,
        default_agent_provider=provider,
    )
    try:
        with pytest.raises(Exception) as exc:
            await service.create_objective(
                CreateObjectiveRequest(
                    session_id="sess_missing",
                    user_message=_user(),
                    budget=_budget(),
                    idempotency_key=new_id(),
                )
            )
        assert "agent_provider_returned_none" in str(exc.value)
    finally:
        await scheduler.stop()


@pytest.mark.asyncio
async def test_history_gap_marks_dispatch_failed() -> None:
    store = InMemoryObjectiveStore()
    scheduler = Scheduler()
    await scheduler.start()
    budget = ObjectiveBudget.create(
        handoffs=1, verification_attempts=1, llm_cost_usd=1, active_seconds=10
    )
    objective_id = "obj_gap"
    inp_id = "inp_gap"
    tagged = UserMessage(
        content=[ContentPart(type=ContentType.TEXT, text="gap")],
        is_user_provided=True,
        objective_input_id=inp_id,
    )
    await store.commit_command(
        receipt=CommandReceipt(
            scope="x",
            idempotency_key="seed",
            request_hash="h",
            status=CommandReceiptStatus.COMPLETED,
            response_payload={},
        ),
        facts=materialize_facts(
            objective_id,
            [
                fact_objective_created(session_id="sess_gap", budget=budget),
                fact_objective_user_input(
                    user_input=ObjectiveUserInput(input_id=inp_id, message=tagged),
                ),
            ],
        ),
        outbox_records=[],
    )

    dispatch = DispatchRequested.create(
        objective_id=objective_id,
        run_id="run_gap",
        role=RunRole.WORK,
        run_input=UserMessage.from_system("go").to_dict(),
        template_hash="",
        session_id="sess_gap",
        state_id="sess_gap",
    )
    await store.commit_command(
        receipt=CommandReceipt(
            scope="outbox",
            idempotency_key="dispatch",
            request_hash="d",
            status=CommandReceiptStatus.COMPLETED,
            response_payload={},
        ),
        facts=[],
        outbox_records=[dispatch],
    )

    agent = Agent(
        AgentConfig(name="t", description="t"), model=_StubModel(), id="sess_gap"
    )
    # Partial history: bootstrap only runs when history has zero objective markers.
    other = ObjectiveUserInput(
        input_id="inp_other",
        message=UserMessage(
            content=[ContentPart(type=ContentType.TEXT, text="other")],
            is_user_provided=True,
            objective_input_id="inp_other",
        ),
    )
    await append_objective_user_input_to_history(
        agent,
        session_id="sess_gap",
        user_input=other,
        objective_id=objective_id,
    )

    async def provider(_session_id: str) -> Agent:
        return agent

    dispatcher = OutboxDispatcher(
        store=store,
        scheduler=scheduler,
        agent_provider=provider,
        poll_interval=0.01,
    )
    await dispatcher.start()
    try:
        for _ in range(50):
            row = await store.get_dispatch(dispatch.dispatch_id)
            assert row is not None
            if row.status is DispatchStatus.FAILED:
                assert "history_gap" in (row.last_error or "")
                return
            await asyncio.sleep(0.05)
        row = await store.get_dispatch(dispatch.dispatch_id)
        raise AssertionError(
            f"expected failed dispatch, got {row.status if row else None}"
        )
    finally:
        await dispatcher.stop()
        await scheduler.stop()
        await agent.close()


@pytest.mark.asyncio
async def test_plan_latch_commits_once() -> None:
    store = InMemoryObjectiveStore()
    budget = ObjectiveBudget.create(
        handoffs=1, verification_attempts=1, llm_cost_usd=1, active_seconds=10
    )
    await store.commit_command(
        receipt=CommandReceipt(
            scope="x",
            idempotency_key="seed",
            request_hash="h",
            status=CommandReceiptStatus.COMPLETED,
            response_payload={},
        ),
        facts=materialize_facts(
            "obj_latch",
            [fact_objective_created(session_id="s1", budget=budget)],
        ),
    )
    assert await maybe_commit_verification_required(
        store,
        objective_id="obj_latch",
        run_id="run_1",
        milestone_id="m1",
    )
    view = project_objective(
        await store.list_facts(objective_id="obj_latch"),
        objective_id="obj_latch",
    )
    assert view is not None
    assert view.verification_required is True
    assert not await maybe_commit_verification_required(
        store,
        objective_id="obj_latch",
        run_id="run_1",
    )


@pytest.mark.asyncio
async def test_entry_complexity_fact_does_not_set_verification_required() -> None:
    budget = ObjectiveBudget.create(
        handoffs=1, verification_attempts=1, llm_cost_usd=1, active_seconds=10
    )
    facts = materialize_facts(
        "o1",
        [
            fact_objective_created(session_id="s1", budget=budget),
            fact_entry_complexity_assessed(
                score=9,
                threshold=COMPLEXITY_PLANNING_THRESHOLD,
            ),
        ],
    )
    view = project_objective(facts)
    assert view is not None
    assert view.entry_complexity_score == 9
    assert view.verification_required is False
