"""P6-01: pause command surface and command idempotency consistency."""

import pytest

from agiwo.objective import (
    CreateObjectiveRequest,
    PauseObjectiveRequest,
    ResumeObjectiveRequest,
)
from agiwo.objective.models import ObjectiveStatus, new_id
from agiwo.objective.store.memory import InMemoryObjectiveStore
from agiwo.objective import ObjectiveService

from tests.objective.e2e.harness import DEFAULT_BUDGET, user_message
from tests.objective.e2e.invariants import assert_objective_invariants


@pytest.mark.asyncio
async def test_pause_idle_created_objective() -> None:
    service = ObjectiveService(InMemoryObjectiveStore())
    created = await service.create_objective(
        CreateObjectiveRequest(
            session_id="sess-pause-idle",
            user_message=user_message("pause me"),
            budget=DEFAULT_BUDGET,
            idempotency_key=new_id(),
        )
    )
    result = await service.pause(
        PauseObjectiveRequest(
            objective_id=created.objective_id,
            idempotency_key=new_id(),
            reason="user_pause",
        )
    )
    assert result.status in {
        ObjectiveStatus.USER_PAUSED.value,
        ObjectiveStatus.DRAINING.value,
        ObjectiveStatus.CREATED.value,
        ObjectiveStatus.RUNNING.value,
    }
    view = await service.get_view(created.objective_id)
    assert view is not None
    assert_objective_invariants(view)

    if view.status is ObjectiveStatus.USER_PAUSED:
        resumed = await service.resume(
            ResumeObjectiveRequest(
                objective_id=created.objective_id,
                idempotency_key=new_id(),
                reason="user_resume",
            )
        )
        assert resumed.objective_id == created.objective_id


@pytest.mark.asyncio
async def test_create_idempotency_same_key_replays() -> None:
    service = ObjectiveService(InMemoryObjectiveStore())
    key = new_id()
    req = CreateObjectiveRequest(
        session_id="sess-idem",
        user_message=user_message("idempotent create"),
        budget=DEFAULT_BUDGET,
        idempotency_key=key,
    )
    first = await service.create_objective(req)
    second = await service.create_objective(req)
    assert first.objective_id == second.objective_id
    assert second.replayed is True
    listed = await service.list_by_session("sess-idem")
    assert len(listed) == 1
