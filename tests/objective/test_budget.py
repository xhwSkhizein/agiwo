"""P3-01: ObjectiveBudget ledger, handoff/verification consumption, adjust floor."""

import asyncio
import inspect

import agiwo.objective.budget as budget_mod
import pytest

from agiwo.agent.models.finalization import RunFinalizationResult
from agiwo.agent.models.input import ContentPart, ContentType, UserMessage
from agiwo.objective import (
    AdjustBudgetRequest,
    BudgetBoundaryHit,
    BudgetLimits,
    CreateObjectiveRequest,
    ObjectiveService,
    ValidationError,
)
from agiwo.objective.budget import (
    budget_agent_projection,
    check_and_plan_consumption,
    consumption_for_decision,
)
from agiwo.objective.errors import StoreError
from agiwo.objective.log import (
    fact_budget_usage_recorded,
    fact_objective_created,
    fact_root_run_requested,
    materialize_facts,
)
from agiwo.objective.models import (
    CommandReceiptStatus,
    HandoffDecision,
    HandoffTarget,
    ObjectiveBudget,
    RunRole,
    new_id,
    utc_now,
)
from agiwo.objective.outbox import DispatchRequested
from agiwo.objective.store.base import CommandReceipt, command_scope
from agiwo.objective.store.memory import InMemoryObjectiveStore


def _user(text: str = "please help") -> UserMessage:
    return UserMessage(
        content=[ContentPart(type=ContentType.TEXT, text=text)],
        is_user_provided=True,
    )


def _limits(
    *,
    handoffs: float = 5,
    verification_attempts: float = 3,
    llm_cost_usd: float = 2.0,
    active_seconds: float = 600,
) -> BudgetLimits:
    return BudgetLimits(
        handoffs=handoffs,
        verification_attempts=verification_attempts,
        llm_cost_usd=llm_cost_usd,
        active_seconds=active_seconds,
    )


def _finalization(
    *, target: str, expects_reply: bool | None = None
) -> RunFinalizationResult:
    decision: dict = {"target": target}
    if expects_reply is not None:
        decision["expects_reply"] = expects_reply
    return RunFinalizationResult(report="done", decision=decision)


def _receipt(scope: str, key: str) -> CommandReceipt:
    now = utc_now()
    return CommandReceipt(
        scope=scope,
        idempotency_key=key,
        request_hash=key,
        status=CommandReceiptStatus.COMPLETED,
        response_payload={},
        created_at=now,
        completed_at=now,
    )


async def _seed_active_root_run(
    store: InMemoryObjectiveStore,
    *,
    handoffs: float,
    verification_attempts: float = 5,
    role: RunRole = RunRole.WORK,
) -> tuple[str, str]:
    oid = new_id("obj_")
    run_id = new_id("run_")
    budget = ObjectiveBudget.create(
        handoffs=handoffs,
        verification_attempts=verification_attempts,
        llm_cost_usd=10.0,
        active_seconds=3600,
    )
    await store.commit_command(
        receipt=_receipt(command_scope(oid, "seed"), "seed"),
        facts=materialize_facts(
            oid,
            [
                fact_objective_created(session_id="s-budget", budget=budget),
                fact_root_run_requested(run_id=run_id, role=role),
            ],
        ),
        outbox_records=[
            DispatchRequested.create(
                objective_id=oid,
                run_id=run_id,
                role=role,
            )
        ],
    )
    return oid, run_id


def test_create_rejects_zero_and_negative_limits() -> None:
    with pytest.raises(ValidationError):
        BudgetLimits(
            handoffs=0,
            verification_attempts=1,
            llm_cost_usd=1,
            active_seconds=1,
        ).to_budget()
    with pytest.raises(ValidationError):
        ObjectiveBudget.create(
            handoffs=-1,
            verification_attempts=1,
            llm_cost_usd=1,
            active_seconds=1,
        )


@pytest.mark.asyncio
async def test_create_requires_all_four_finite_limits() -> None:
    service = ObjectiveService(InMemoryObjectiveStore())
    result = await service.create_objective(
        CreateObjectiveRequest(
            session_id="s1",
            user_message=_user(),
            budget=_limits(),
            idempotency_key="k1",
        )
    )
    view = await service.get_view(result.objective_id)
    assert view is not None
    assert view.budget.handoffs.limit == 5
    assert view.budget.verification_attempts.used == 0
    agent_proj = view.budget_for_agent()
    assert agent_proj["max_handoffs"] == 5
    assert agent_proj["remaining_handoffs"] == 5


def test_consumption_targets() -> None:
    agent = HandoffDecision(target=HandoffTarget.AGENT)
    verifier = HandoffDecision(target=HandoffTarget.VERIFIER)
    user = HandoffDecision(target=HandoffTarget.USER, expects_reply=True)
    assert consumption_for_decision(agent).handoffs_delta == 1
    assert consumption_for_decision(agent).verification_attempts_delta == 0
    plan = consumption_for_decision(verifier)
    assert plan.handoffs_delta == 1
    assert plan.verification_attempts_delta == 1
    assert consumption_for_decision(user).is_empty


def test_verifier_atomic_check_no_partial() -> None:
    base = ObjectiveBudget.create(
        handoffs=1,
        verification_attempts=1,
        llm_cost_usd=1,
        active_seconds=1,
    )
    budget = ObjectiveBudget(
        handoffs=base.handoffs,
        verification_attempts=base.verification_attempts.with_used(1),
        llm_cost_usd=base.llm_cost_usd,
        active_seconds=base.active_seconds,
    )
    with pytest.raises(BudgetBoundaryHit) as exc:
        check_and_plan_consumption(
            budget,
            HandoffDecision(target=HandoffTarget.VERIFIER),
        )
    assert exc.value.dimension == "verification_attempts"
    # handoffs still available; check does not mutate the snapshot
    assert budget.handoffs.used == 0
    assert budget.handoffs.remaining == 1


@pytest.mark.asyncio
async def test_agent_handoff_consumes_once_and_creates_next() -> None:
    store = InMemoryObjectiveStore()
    service = ObjectiveService(store)
    oid, run_id = await _seed_active_root_run(store, handoffs=2)
    await service.apply_finalization(
        objective_id=oid,
        run_id=run_id,
        finalization=_finalization(target="agent"),
    )
    view = await service.get_view(oid)
    assert view is not None
    assert view.budget.handoffs.used == 1
    assert view.budget.verification_attempts.used == 0
    assert view.active_root_run is not None
    assert view.active_root_run.role is RunRole.WORK


@pytest.mark.asyncio
async def test_user_target_does_not_consume_handoffs() -> None:
    store = InMemoryObjectiveStore()
    service = ObjectiveService(store)
    oid, run_id = await _seed_active_root_run(store, handoffs=2)
    await service.apply_finalization(
        objective_id=oid,
        run_id=run_id,
        finalization=_finalization(target="user", expects_reply=True),
    )
    view = await service.get_view(oid)
    assert view is not None
    assert view.budget.handoffs.used == 0
    assert view.active_root_run is None


@pytest.mark.asyncio
async def test_verifier_consumes_both_dimensions() -> None:
    store = InMemoryObjectiveStore()
    service = ObjectiveService(store)
    oid, run_id = await _seed_active_root_run(
        store,
        handoffs=3,
        verification_attempts=2,
        role=RunRole.WORK,
    )
    await service.apply_finalization(
        objective_id=oid,
        run_id=run_id,
        finalization=_finalization(target="verifier"),
    )
    view = await service.get_view(oid)
    assert view is not None
    assert view.budget.handoffs.used == 1
    assert view.budget.verification_attempts.used == 1
    assert view.active_root_run is not None
    assert view.active_root_run.role is RunRole.VERIFICATION


@pytest.mark.asyncio
async def test_budget_boundary_commits_outcome_without_next_assignment() -> None:
    store = InMemoryObjectiveStore()
    service = ObjectiveService(store)
    oid, run_id = await _seed_active_root_run(store, handoffs=1)
    # Consume the only handoff first.
    await service.apply_finalization(
        objective_id=oid,
        run_id=run_id,
        finalization=_finalization(target="agent"),
    )
    view = await service.get_view(oid)
    assert view is not None
    assert view.budget.handoffs.used == 1
    work = view.active_root_run
    assert work is not None

    with pytest.raises(BudgetBoundaryHit) as exc:
        await service.apply_finalization(
            objective_id=oid,
            run_id=work.run_id,
            finalization=_finalization(target="agent"),
        )
    assert exc.value.dimension == "handoffs"

    view = await service.get_view(oid)
    assert view is not None
    assert view.budget.handoffs.used == 1
    assert len(view.outcomes) == 2
    assert view.active_root_run is None
    assert view.status.value == "BUDGET_PAUSED"
    pending = await store.list_pending_dispatches()
    # Seed dispatch for run1 plus follow-on dispatch for run2; no third outbox.
    assert len([p for p in pending if p.role is RunRole.WORK]) == 2


@pytest.mark.asyncio
async def test_verifier_boundary_no_partial_usage_facts() -> None:
    store = InMemoryObjectiveStore()
    service = ObjectiveService(store)
    oid, run_id = await _seed_active_root_run(
        store,
        handoffs=5,
        verification_attempts=1,
        role=RunRole.WORK,
    )
    # Exhaust verification via a prior usage fact.
    await store.commit_command(
        receipt=_receipt(command_scope(oid, "usage"), "u1"),
        facts=materialize_facts(
            oid,
            [
                fact_budget_usage_recorded(
                    dimension="verification_attempts",
                    delta=1,
                    used_after=1,
                )
            ],
            start_sequence=3,
        ),
    )
    with pytest.raises(BudgetBoundaryHit):
        await service.apply_finalization(
            objective_id=oid,
            run_id=run_id,
            finalization=_finalization(target="verifier"),
        )
    view = await service.get_view(oid)
    assert view is not None
    assert view.budget.handoffs.used == 0
    assert view.budget.verification_attempts.used == 1


@pytest.mark.asyncio
async def test_concurrent_usage_commits_only_budget_allows() -> None:
    store = InMemoryObjectiveStore()
    oid = new_id("obj_")
    budget = ObjectiveBudget.create(
        handoffs=1,
        verification_attempts=5,
        llm_cost_usd=1,
        active_seconds=60,
    )
    await store.commit_command(
        receipt=_receipt(command_scope(oid, "seed"), "seed"),
        facts=materialize_facts(
            oid,
            [fact_objective_created(session_id="s1", budget=budget)],
        ),
    )

    async def try_consume(key: str) -> str:
        try:
            await store.commit_command(
                receipt=_receipt(command_scope(oid, "usage"), key),
                facts=materialize_facts(
                    oid,
                    [
                        fact_budget_usage_recorded(
                            dimension="handoffs",
                            delta=1,
                            used_after=1,
                        )
                    ],
                    start_sequence=2,
                ),
            )
            return "ok"
        except StoreError:
            return "store"
        except Exception as exc:  # noqa: BLE001
            return type(exc).__name__

    results = await asyncio.gather(try_consume("a"), try_consume("b"))
    assert results.count("ok") == 1
    assert results.count("store") == 1
    view_facts = await store.list_facts(objective_id=oid)
    usage = [f for f in view_facts if f.kind.value == "BudgetUsageRecorded"]
    assert len(usage) == 1


@pytest.mark.asyncio
async def test_adjust_budget_rejects_below_used_and_is_idempotent() -> None:
    service = ObjectiveService(InMemoryObjectiveStore())
    created = await service.create_objective(
        CreateObjectiveRequest(
            session_id="s1",
            user_message=_user(),
            budget=_limits(handoffs=3),
            idempotency_key="c1",
        )
    )
    # create_objective wrote sequences 1-2 (created + user input).
    store = service._store  # noqa: SLF001
    await store.commit_command(
        receipt=_receipt(command_scope(created.objective_id, "u"), "u"),
        facts=materialize_facts(
            created.objective_id,
            [
                fact_budget_usage_recorded(
                    dimension="handoffs",
                    delta=2,
                    used_after=2,
                )
            ],
            start_sequence=3,
        ),
    )
    with pytest.raises(ValidationError):
        await service.adjust_budget(
            AdjustBudgetRequest(
                objective_id=created.objective_id,
                idempotency_key="adj-low",
                handoffs=1,
            )
        )
    req = AdjustBudgetRequest(
        objective_id=created.objective_id,
        idempotency_key="adj-ok",
        handoffs=10,
    )
    r1 = await service.adjust_budget(req)
    r2 = await service.adjust_budget(req)
    assert r2.replayed is True
    assert r1.payload["budget"]["handoffs"]["limit"] == 10
    view = await service.get_view(created.objective_id)
    assert view is not None
    assert view.budget.handoffs.limit == 10
    assert view.budget.handoffs.used == 2


def test_agent_projection_is_read_only_and_no_agent_mutate_api() -> None:
    budget = ObjectiveBudget.create(
        handoffs=3,
        verification_attempts=2,
        llm_cost_usd=1,
        active_seconds=60,
    )
    proj = budget_agent_projection(budget)
    assert set(proj) == {
        "max_handoffs",
        "used_handoffs",
        "remaining_handoffs",
        "max_verification_attempts",
        "used_verification_attempts",
        "remaining_verification_attempts",
        "max_llm_cost_usd",
        "used_llm_cost_usd",
        "remaining_llm_cost_usd",
        "max_active_seconds",
        "used_active_seconds",
        "remaining_active_seconds",
        "historical_active_seconds",
    }
    mutate_names = {
        name
        for name in dir(budget_mod)
        if name.startswith(("set_", "raise_", "increase_", "mutate_"))
    }
    assert mutate_names == set()
    # User boundary keeps adjust_budget; agent package must not import objective.
    assert hasattr(ObjectiveService, "adjust_budget")
    assert inspect.iscoroutinefunction(ObjectiveService.adjust_budget)
