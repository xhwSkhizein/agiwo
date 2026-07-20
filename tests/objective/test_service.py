"""P1-04 ObjectiveService facade contract tests."""

import asyncio

import pytest

from agiwo.agent.models.input import ContentPart, ContentType, UserMessage
from agiwo.objective import (
    BudgetLimits,
    CreateObjectiveRequest,
    ExternalizeUserInputRequest,
    IdempotencyConflict,
    InvariantViolation,
    ObjectiveService,
    ObjectiveStatus,
    PauseObjectiveRequest,
    ResumeObjectiveRequest,
    SubmitUserInputRequest,
    create_objective_store,
)
from agiwo.objective.log import (
    fact_objective_created,
    fact_objective_status_changed,
    fact_objective_user_input,
    materialize_facts,
)
from agiwo.objective.models import (
    AdjustBudgetRequest,
    CommandReceiptStatus,
    ObjectiveBudget,
    ObjectiveUserInput,
    new_id,
)
from agiwo.objective.store.base import CommandReceipt, SlotMutation
from agiwo.objective.store.memory import InMemoryObjectiveStore


def _user(text: str = "please help") -> UserMessage:
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


@pytest.mark.asyncio
async def test_create_requires_real_user_and_budget() -> None:
    service = ObjectiveService(InMemoryObjectiveStore())
    with pytest.raises(Exception):
        await service.create_objective(
            CreateObjectiveRequest(
                session_id="s1",
                user_message=UserMessage.from_system("nope"),
                budget=_budget(),
                idempotency_key="k1",
            )
        )
    result = await service.create_objective(
        CreateObjectiveRequest(
            session_id="s1",
            user_message=_user(),
            budget=_budget(),
            idempotency_key="k1",
        )
    )
    assert result.status == ObjectiveStatus.CREATED.value
    view = await service.get_view(result.objective_id)
    assert view is not None
    assert view.session_id == "s1"
    assert len(view.user_inputs) == 1


@pytest.mark.asyncio
async def test_create_idempotent_replay_and_conflict() -> None:
    service = ObjectiveService(InMemoryObjectiveStore())
    req = CreateObjectiveRequest(
        session_id="s1",
        user_message=_user(),
        budget=_budget(),
        idempotency_key="same-key",
    )
    r1 = await service.create_objective(req)
    r2 = await service.create_objective(req)
    assert r2.replayed is True
    assert r2.objective_id == r1.objective_id
    with pytest.raises(IdempotencyConflict):
        await service.create_objective(
            CreateObjectiveRequest(
                session_id="s1",
                user_message=_user("different"),
                budget=_budget(),
                idempotency_key="same-key",
            )
        )


@pytest.mark.asyncio
async def test_concurrent_create_one_session_slot() -> None:
    store = InMemoryObjectiveStore()
    service = ObjectiveService(store)

    async def create(key: str, text: str):
        try:
            return await service.create_objective(
                CreateObjectiveRequest(
                    session_id="s1",
                    user_message=_user(text),
                    budget=_budget(),
                    idempotency_key=key,
                )
            )
        except Exception as exc:  # noqa: BLE001
            return exc

    results = await asyncio.gather(create("k1", "a"), create("k2", "b"))
    successes = [r for r in results if not isinstance(r, Exception)]
    assert len(successes) == 1
    slot = await store.get_session_slot("s1")
    assert slot is not None
    assert slot.objective_id == successes[0].objective_id


@pytest.mark.asyncio
async def test_terminal_rejects_user_input() -> None:

    store = InMemoryObjectiveStore()
    service = ObjectiveService(store)
    oid = "obj_term"
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
            oid,
            [
                fact_objective_created(session_id="s1", budget=budget),
                fact_objective_user_input(
                    user_input=ObjectiveUserInput(input_id="i1", message=_user()),
                ),
                fact_objective_status_changed(
                    from_status=ObjectiveStatus.CREATED,
                    to_status=ObjectiveStatus.RUNNING,
                    reason="go",
                ),
                fact_objective_status_changed(
                    from_status=ObjectiveStatus.RUNNING,
                    to_status=ObjectiveStatus.COMPLETED,
                    reason="done",
                ),
            ],
        ),
        slot_mutation=SlotMutation(action="acquire", session_id="s1", objective_id=oid),
    )
    # release as terminal would
    await store.commit_command(
        receipt=CommandReceipt(
            scope="x",
            idempotency_key="rel",
            request_hash="hr",
            status=CommandReceiptStatus.COMPLETED,
            response_payload={},
        ),
        facts=[],
        slot_mutation=SlotMutation(action="release", session_id="s1", objective_id=oid),
    )
    with pytest.raises(InvariantViolation):
        await service.submit_user_input(
            SubmitUserInputRequest(
                objective_id=oid,
                user_message=_user("more"),
                idempotency_key=new_id(),
            )
        )


@pytest.mark.asyncio
async def test_externalize_idempotent_and_keeps_original(tmp_path) -> None:
    service = ObjectiveService(
        InMemoryObjectiveStore(),
        artifacts_root=tmp_path,
    )
    created = await service.create_objective(
        CreateObjectiveRequest(
            session_id="s1",
            user_message=_user("very long user text"),
            budget=_budget(),
            idempotency_key="c1",
        )
    )
    view = await service.get_view(created.objective_id)
    assert view is not None
    input_id = view.user_inputs[0].input_id
    r1 = await service.externalize_user_input(
        ExternalizeUserInputRequest(
            objective_id=created.objective_id,
            input_id=input_id,
            summary="summary",
            idempotency_key="ext1",
        )
    )
    r2 = await service.externalize_user_input(
        ExternalizeUserInputRequest(
            objective_id=created.objective_id,
            input_id=input_id,
            summary="summary",
            idempotency_key="ext1",
        )
    )
    assert r2.replayed is True
    assert r2.payload["artifact_id"] == r1.payload["artifact_id"]
    view2 = await service.get_view(created.objective_id)
    assert view2 is not None
    assert view2.user_inputs[0].message.extract_text() == "very long user text"
    assert len(view2.externalized_inputs) == 1


@pytest.mark.asyncio
async def test_adjust_budget() -> None:
    service = ObjectiveService(InMemoryObjectiveStore())
    created = await service.create_objective(
        CreateObjectiveRequest(
            session_id="s1",
            user_message=_user(),
            budget=_budget(),
            idempotency_key="c1",
        )
    )
    result = await service.adjust_budget(
        AdjustBudgetRequest(
            objective_id=created.objective_id,
            idempotency_key="b1",
            handoffs=20,
        )
    )
    assert result.payload["budget"]["handoffs"]["limit"] == 20
    view = await service.get_view(created.objective_id)
    assert view is not None
    assert view.budget.handoffs.limit == 20


@pytest.mark.asyncio
async def test_public_import_surface() -> None:
    import agiwo.objective as obj  # noqa: PLC0415

    public = set(obj.__all__)
    assert "ObjectiveService" in public
    assert "create_objective_store" in public
    assert "ObjectiveView" in public
    # Internal types not exported
    assert "ObjectiveLogEntry" not in public
    assert "DispatchRequested" not in public
    assert "project_objective" not in public


@pytest.mark.asyncio
async def test_list_by_session() -> None:
    service = ObjectiveService(create_objective_store())
    await service.create_objective(
        CreateObjectiveRequest(
            session_id="s1",
            user_message=_user("one"),
            budget=_budget(),
            idempotency_key="a",
        )
    )
    views = await service.list_by_session("s1")
    assert len(views) == 1


@pytest.mark.asyncio
async def test_idle_pause_and_resume_without_active_run() -> None:
    store = InMemoryObjectiveStore()
    service = ObjectiveService(store)
    created = await service.create_objective(
        CreateObjectiveRequest(
            session_id="s1",
            user_message=_user(),
            budget=_budget(),
            idempotency_key="c1",
        )
    )
    # No Assignment yet → idle drain completes to USER_PAUSED without Scheduler.
    paused = await service.pause(
        PauseObjectiveRequest(
            objective_id=created.objective_id,
            idempotency_key="p1",
        )
    )
    assert paused.status == ObjectiveStatus.USER_PAUSED.value
    view = await service.get_view(created.objective_id)
    assert view is not None
    assert view.status == ObjectiveStatus.USER_PAUSED

    resumed = await service.resume(
        ResumeObjectiveRequest(
            objective_id=created.objective_id,
            idempotency_key="r1",
        )
    )
    assert resumed.status == ObjectiveStatus.RUNNING.value
    view = await service.get_view(created.objective_id)
    assert view is not None
    assert view.status == ObjectiveStatus.RUNNING
    assert view.current_active_started_at is not None
