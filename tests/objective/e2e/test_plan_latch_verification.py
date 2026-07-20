"""E2E: update_plan milestone → verification_required latch → verifier delivery."""

import json

import pytest

from agiwo.objective import CreateObjectiveRequest
from agiwo.objective.log import ObjectiveFactKind
from agiwo.objective.models import ObjectiveStatus, RunRole, new_id

from tests.objective.e2e.harness import (
    DEFAULT_BUDGET,
    ScriptedToolCallModel,
    finalization_json,
    objective_e2e_runtime,
    user_message,
    wait_completed,
    wait_for,
)


def _update_plan_call(*, call_id: str, milestone_status: str = "pending") -> dict:
    return {
        "index": 0,
        "id": call_id,
        "type": "function",
        "function": {
            "name": "update_plan",
            "arguments": json.dumps(
                {
                    "changes": [
                        {
                            "id": "inspect",
                            "description": "Inspect the deliverable",
                            "status": milestone_status,
                        }
                    ]
                }
            ),
        },
    }


@pytest.mark.asyncio
async def test_update_plan_latch_forces_verification_chain() -> None:
    responses: list[str | dict] = [
        _update_plan_call(call_id="plan-1"),
        "work report",
        _update_plan_call(call_id="plan-2", milestone_status="completed"),
        "completed report",
        finalization_json(
            target="user",
            expects_reply=False,
        ),
        # Last assistant text is the Outcome report; finalization JSON has no report.
        "verified and delivered",
        finalization_json(
            target="user",
            expects_reply=False,
        ),
    ]

    async with objective_e2e_runtime(
        responses=responses,
        model=ScriptedToolCallModel(responses),
    ) as runtime:
        created = await runtime.service.create_objective(
            CreateObjectiveRequest(
                session_id=runtime.session_id,
                user_message=user_message("please plan and complete the task"),
                budget=DEFAULT_BUDGET,
                idempotency_key=new_id(),
            )
        )
        objective_id = created.objective_id

        async def latch_fact_present() -> bool:
            facts = await runtime.store.list_facts(objective_id=objective_id)
            return any(
                f.kind is ObjectiveFactKind.VERIFICATION_REQUIRED_SET for f in facts
            )

        # Scripted runs finish in <1s; do not require observing RUNNING.
        await wait_for(latch_fact_present, timeout_seconds=12.0)

        view = await wait_completed(runtime.service, objective_id)
        assert view.status is ObjectiveStatus.COMPLETED
        assert view.verification_required is True
        assert view.delivery_report == "verified and delivered"
        roles = [r.role for r in view.root_runs]
        assert RunRole.WORK in roles
        assert RunRole.VERIFICATION in roles

        facts = await runtime.store.list_facts(objective_id=objective_id)
        latch_seq = next(
            f.sequence
            for f in facts
            if f.kind is ObjectiveFactKind.VERIFICATION_REQUIRED_SET
        )
        first_decision_seq = min(
            f.sequence for f in facts if f.kind is ObjectiveFactKind.DECISION_ACCEPTED
        )
        assert latch_seq < first_decision_seq
