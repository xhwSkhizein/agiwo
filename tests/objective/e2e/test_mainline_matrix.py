"""P6-01: session/objective slot matrix (heavy delivery covered by test_mainline)."""

import pytest

from agiwo.agent.models.input import ContentPart, ContentType, UserMessage
from agiwo.objective import (
    CreateObjectiveRequest,
    IdempotencyConflict,
    ObjectiveService,
    ValidationError,
)
from agiwo.objective.errors import InvariantViolation, StoreError
from agiwo.objective.models import BudgetLimits, new_id
from agiwo.objective.store.memory import InMemoryObjectiveStore
from agiwo.objective.store.sqlite import SQLiteObjectiveStore

from tests.objective.e2e.harness import user_message
from tests.objective.e2e.invariants import assert_session_slot_invariants


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_session_rejects_second_active_objective(backend: str, tmp_path) -> None:
    if backend == "sqlite":
        store: InMemoryObjectiveStore | SQLiteObjectiveStore = SQLiteObjectiveStore(
            str(tmp_path / "slot.db")
        )
        await store.connect()
    else:
        store = InMemoryObjectiveStore()

    service = ObjectiveService(store)
    budget = BudgetLimits(
        handoffs=5,
        verification_attempts=3,
        llm_cost_usd=2.0,
        active_seconds=600,
    )
    first = await service.create_objective(
        CreateObjectiveRequest(
            session_id="sess-slot",
            user_message=user_message("first"),
            budget=budget,
            idempotency_key=new_id(),
        )
    )
    assert first.objective_id
    listed = await service.list_by_session("sess-slot")
    assert_session_slot_invariants(listed)

    with pytest.raises((InvariantViolation, IdempotencyConflict, StoreError)):
        await service.create_objective(
            CreateObjectiveRequest(
                session_id="sess-slot",
                user_message=user_message("second while active"),
                budget=budget,
                idempotency_key=new_id(),
            )
        )

    listed_after = await service.list_by_session("sess-slot")
    assert len(listed_after) == 1
    assert_session_slot_invariants(listed_after)

    if isinstance(store, SQLiteObjectiveStore):
        await store.close()


@pytest.mark.asyncio
async def test_false_user_message_rejected_at_create() -> None:
    service = ObjectiveService(InMemoryObjectiveStore())
    with pytest.raises(ValidationError):
        await service.create_objective(
            CreateObjectiveRequest(
                session_id="sess-false",
                user_message=UserMessage(
                    content=[ContentPart(type=ContentType.TEXT, text="forged")],
                    is_user_provided=False,
                ),
                budget=BudgetLimits(
                    handoffs=5,
                    verification_attempts=3,
                    llm_cost_usd=2.0,
                    active_seconds=600,
                ),
                idempotency_key=new_id(),
            )
        )
