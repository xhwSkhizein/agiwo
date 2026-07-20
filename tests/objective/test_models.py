"""P1-01 domain model unit tests."""

import pytest

from agiwo.agent.models.input import ContentPart, ContentType, UserMessage
from agiwo.agent.models.run import RunStatus
from agiwo.objective.errors import InvalidStateTransition, ValidationError
from agiwo.objective.models import (
    ARTIFACT_INLINE_CONTENT_MAX_BYTES,
    Artifact,
    ArtifactRef,
    CurrentGoalAnalysis,
    HandoffDecision,
    HandoffTarget,
    ObjectiveBudget,
    ObjectiveStatus,
    ObjectiveUpdateSpec,
    ObjectiveUserInput,
    RunOutcome,
    RunRole,
    assert_single_active_objective,
    assert_single_active_root_run,
    is_objective_resumable,
    is_root_run_resumable,
    validate_objective_transition,
)


def _user(text: str = "hello") -> UserMessage:
    return UserMessage(
        content=[ContentPart(type=ContentType.TEXT, text=text)],
        is_user_provided=True,
    )


def test_objective_status_transitions_table() -> None:
    validate_objective_transition(ObjectiveStatus.CREATED, ObjectiveStatus.RUNNING)
    validate_objective_transition(ObjectiveStatus.RUNNING, ObjectiveStatus.DRAINING)
    validate_objective_transition(ObjectiveStatus.USER_PAUSED, ObjectiveStatus.RUNNING)
    with pytest.raises(InvalidStateTransition):
        validate_objective_transition(
            ObjectiveStatus.COMPLETED, ObjectiveStatus.RUNNING
        )
    with pytest.raises(InvalidStateTransition):
        validate_objective_transition(ObjectiveStatus.FAILED, ObjectiveStatus.CREATED)
    with pytest.raises(InvalidStateTransition):
        validate_objective_transition(
            ObjectiveStatus.CREATED, ObjectiveStatus.COMPLETED
        )


def test_root_run_resumable() -> None:
    assert is_root_run_resumable(RunStatus.PAUSED)
    assert not is_root_run_resumable(RunStatus.RUNNING)
    assert not is_root_run_resumable(None)
    assert is_objective_resumable(ObjectiveStatus.USER_PAUSED)
    assert not is_objective_resumable(ObjectiveStatus.COMPLETED)


def test_session_and_root_run_cardinality() -> None:
    with pytest.raises(Exception) as exc:
        assert_single_active_objective(
            session_id="s1",
            active_objective_ids=["o1", "o2"],
        )
    assert exc.value.code == "invariant_violation"
    with pytest.raises(Exception) as exc2:
        assert_single_active_root_run(
            objective_id="o1",
            active_run_ids=["r1", "r2"],
        )
    assert exc2.value.code == "invariant_violation"
    assert_single_active_objective(session_id="s1", active_objective_ids=["o1"])
    assert_single_active_root_run(objective_id="o1", active_run_ids=["r1"])


def test_handoff_decision_targets() -> None:
    HandoffDecision(target=HandoffTarget.AGENT)
    HandoffDecision(target=HandoffTarget.VERIFIER)
    HandoffDecision(target=HandoffTarget.USER, expects_reply=True)
    HandoffDecision(target=HandoffTarget.USER, expects_reply=False)
    with pytest.raises(ValidationError):
        HandoffDecision(target=HandoffTarget.USER)
    with pytest.raises(ValidationError):
        HandoffDecision(target=HandoffTarget.AGENT, expects_reply=True)


def test_outcome_requires_run_id_and_terminal() -> None:
    with pytest.raises(ValidationError):
        RunOutcome(
            run_id="",
            role=RunRole.WORK,
            terminal_status=RunStatus.COMPLETED,
            reason="done",
            report="report",
        )
    with pytest.raises(ValidationError):
        RunOutcome(
            run_id="r1",
            role=RunRole.WORK,
            terminal_status=RunStatus.PAUSED,
            reason="x",
            report="r",
        )
    with pytest.raises(ValidationError):
        RunOutcome(
            run_id="r1",
            role=RunRole.WORK,
            terminal_status=RunStatus.RUNNING,
            reason="x",
            report="r",
        )
    ok = RunOutcome(
        run_id="r1",
        role=RunRole.WORK,
        terminal_status=RunStatus.COMPLETED,
        reason="done",
        report="plain text report",
    )
    assert ok.report == "plain text report"


def test_objective_user_input_requires_real_user_message() -> None:
    with pytest.raises(ValidationError):
        ObjectiveUserInput(
            input_id="i1",
            message=UserMessage.from_system("system notice"),
        )
    ok = ObjectiveUserInput(input_id="i1", message=_user())
    assert ok.message.is_user_provided


def test_artifact_is_file_index_not_report() -> None:
    art = Artifact(
        artifact_id="art1",
        path="sessions/s1/artifacts/art1.txt",
        summary="big input",
        source_input_id="i1",
        content_hash="abc",
    )
    assert "decision" not in art.to_dict()
    with pytest.raises(ValidationError):
        Artifact(
            artifact_id="art2",
            path="sessions/s1/artifacts/big.txt",
            summary="x",
            inline_content="x" * (ARTIFACT_INLINE_CONTENT_MAX_BYTES + 1),
        )
    ref = ArtifactRef(artifact_id="art1")
    assert ref.path is None
    with pytest.raises(ValidationError):
        ArtifactRef()


def test_current_goal_analysis_forbids_plan_budget_fields() -> None:
    CurrentGoalAnalysis(revision=1, intent="do X")
    with pytest.raises(ValidationError):
        CurrentGoalAnalysis.from_dict(
            {"revision": 1, "intent": "x", "budget": {"handoffs": 1}}
        )
    with pytest.raises(ValidationError):
        ObjectiveUpdateSpec.from_dict(
            {
                "expected_revision": 1,
                "intent": "x",
                "run_plan": [],
            }
        )


def test_budget_dimensions_limit_only() -> None:
    budget = ObjectiveBudget.create(
        handoffs=3,
        verification_attempts=2,
        llm_cost_usd=1.5,
        active_seconds=600,
    )
    assert budget.handoffs.used == 0
    assert not hasattr(budget.handoffs, "reserved")
    with pytest.raises(ValidationError):
        ObjectiveBudget.create(
            handoffs=0,
            verification_attempts=1,
            llm_cost_usd=1,
            active_seconds=1,
        )
