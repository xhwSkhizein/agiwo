"""Regression: simple delivery and verification_required coercion."""

import pytest

from agiwo.agent.models.finalization import (
    RunFinalizationResult,
    parse_finalization_json,
)
from agiwo.agent.models.run import RunStatus
from agiwo.objective.finalization import (
    map_finalization_to_outcome,
    should_deliver,
)
from agiwo.objective.models import (
    HandoffDecision,
    HandoffTarget,
    ObjectiveBudget,
    ObjectiveStatus,
    RunRole,
)
from agiwo.objective.projection import ObjectiveView, RootRunView


def _view(
    *,
    role: RunRole,
    verification_required: bool = False,
) -> ObjectiveView:
    root_run = RootRunView(
        run_id="run_1",
        role=role,
        status=RunStatus.RUNNING,
    )
    return ObjectiveView(
        objective_id="obj_1",
        session_id="sess_1",
        status=ObjectiveStatus.RUNNING,
        budget=ObjectiveBudget.create(
            handoffs=5,
            verification_attempts=3,
            llm_cost_usd=1.0,
            active_seconds=600,
        ),
        verification_required=verification_required,
        root_runs=(root_run,),
    )


def test_parse_finalization_defaults_missing_expects_reply_to_true() -> None:
    result = parse_finalization_json(
        '{"decision":{"target":"user"},'
        '"new_contributions":[],"contribution_annotations":[],'
        '"objective_update":null,"artifact_refs":[],"carry_forward":[]}',
        report="done",
    )
    assert result.report == "done"
    assert result.decision["target"] == "user"
    assert result.decision["expects_reply"] is True


def test_parse_finalization_rejects_model_report_field() -> None:
    with pytest.raises(ValueError, match="must not include report"):
        parse_finalization_json(
            '{"report":"sneaky","decision":{"target":"user","expects_reply":false},'
            '"new_contributions":[],"contribution_annotations":[],'
            '"objective_update":null,"artifact_refs":[],"carry_forward":[]}',
            report="done",
        )


def test_simple_work_can_deliver_when_not_verification_required() -> None:
    decision = HandoffDecision(target=HandoffTarget.USER, expects_reply=False)
    assert should_deliver(
        decision,
        role=RunRole.WORK,
        verification_required=False,
    )
    assert should_deliver(
        decision,
        role=RunRole.VERIFICATION,
        verification_required=True,
    )
    assert should_deliver(
        decision,
        role=RunRole.VERIFICATION,
        verification_required=False,
    )
    assert not should_deliver(
        decision,
        role=RunRole.WORK,
        verification_required=True,
    )


def test_work_user_delivery_coerced_when_verification_required() -> None:
    outcome, decision = map_finalization_to_outcome(
        view=_view(role=RunRole.WORK, verification_required=True),
        run_id="run_1",
        role=RunRole.WORK,
        result=RunFinalizationResult(
            report="done",
            decision={"target": "user", "expects_reply": False},
        ),
    )
    assert decision.target is HandoffTarget.VERIFIER
    assert outcome.decision.target is HandoffTarget.VERIFIER
    assert decision.reason == "verification_required_coerced_to_verifier"
