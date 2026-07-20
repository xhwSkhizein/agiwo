"""P1-02 projection tests."""

import pytest

from agiwo.agent.models.input import ContentPart, ContentType, UserMessage
from agiwo.agent.models.run import RunStatus
from agiwo.objective.errors import ProjectionError
from agiwo.objective.log import (
    FactBatch,
    expand_outcome_derived_facts,
    fact_artifact_registered,
    fact_context_capacity_exceeded,
    fact_contribution_annotated,
    fact_contribution_created,
    fact_current_goal_revised,
    fact_objective_created,
    fact_objective_delivered,
    fact_objective_status_changed,
    fact_objective_user_input,
    fact_root_run_paused,
    fact_root_run_requested,
    fact_root_run_started,
    fact_run_outcome,
    fact_user_input_externalized,
    fact_verification_required_set,
    materialize_facts,
)
from agiwo.objective.models import (
    Artifact,
    ContributionAnnotation,
    CurrentGoalAnalysis,
    NewContributionSpec,
    ObjectiveBudget,
    ObjectiveContribution,
    ObjectiveStatus,
    ObjectiveUpdateSpec,
    ObjectiveUserInput,
    RunOutcome,
    RunRole,
)
from agiwo.objective.projection import project_objective


def _budget() -> ObjectiveBudget:
    return ObjectiveBudget.create(
        handoffs=5,
        verification_attempts=3,
        llm_cost_usd=2.0,
        active_seconds=600,
    )


def _user(text: str = "do the work") -> UserMessage:
    return UserMessage(
        content=[ContentPart(type=ContentType.TEXT, text=text)],
        is_user_provided=True,
    )


def test_project_created_running_completed_path() -> None:
    budget = _budget()
    facts = materialize_facts(
        "o1",
        [
            fact_objective_created(session_id="s1", budget=budget),
            fact_objective_user_input(
                user_input=ObjectiveUserInput(input_id="i1", message=_user()),
            ),
            fact_root_run_requested(
                run_id="r1",
                role=RunRole.WORK,
            ),
            fact_root_run_started(
                run_id="r1",
            ),
        ],
    )
    view = project_objective(facts)
    assert view is not None
    assert view.status == ObjectiveStatus.RUNNING
    assert view.active_root_run is not None
    assert view.active_root_run.status == RunStatus.RUNNING
    assert len(view.user_inputs) == 1

    outcome = RunOutcome(
        run_id="r1",
        role=RunRole.WORK,
        terminal_status=RunStatus.COMPLETED,
        reason="done",
        report="final report text",
        outcome_id="out1",
    )
    facts2 = facts + materialize_facts(
        "o1",
        [
            fact_run_outcome(outcome=outcome),
            fact_objective_delivered(
                final_outcome_id="out1",
                report="final report text",
            ),
        ],
        start_sequence=5,
    )
    view2 = project_objective(facts2)
    assert view2 is not None
    assert view2.status == ObjectiveStatus.COMPLETED
    assert view2.delivery_report == "final report text"
    assert view2.root_runs[0].outcome is not None


def test_illegal_status_transition_fails() -> None:
    budget = _budget()
    facts = materialize_facts(
        "o1",
        [
            fact_objective_created(session_id="s1", budget=budget),
            fact_objective_status_changed(
                from_status=ObjectiveStatus.CREATED,
                to_status=ObjectiveStatus.COMPLETED,
                reason="bad",
            ),
        ],
    )
    with pytest.raises(Exception):
        project_objective(facts)


def test_terminal_reopen_fails() -> None:
    budget = _budget()
    facts = materialize_facts(
        "o1",
        [
            fact_objective_created(session_id="s1", budget=budget),
            fact_objective_status_changed(
                from_status=ObjectiveStatus.CREATED,
                to_status=ObjectiveStatus.RUNNING,
                reason="start",
            ),
            fact_objective_status_changed(
                from_status=ObjectiveStatus.RUNNING,
                to_status=ObjectiveStatus.COMPLETED,
                reason="done",
            ),
            fact_objective_status_changed(
                from_status=ObjectiveStatus.COMPLETED,
                to_status=ObjectiveStatus.RUNNING,
                reason="reopen",
            ),
        ],
    )
    with pytest.raises(Exception):
        project_objective(facts)


def test_run_outcome_uniqueness() -> None:
    budget = _budget()
    base = materialize_facts(
        "o1",
        [
            fact_objective_created(session_id="s1", budget=budget),
            fact_root_run_requested(
                run_id="r1",
                role=RunRole.WORK,
            ),
            fact_root_run_started(
                run_id="r1",
            ),
        ],
    )
    outcome = RunOutcome(
        run_id="r1",
        role=RunRole.WORK,
        terminal_status=RunStatus.COMPLETED,
        reason="ok",
        report="r",
        outcome_id="out1",
    )
    view = project_objective(
        base
        + materialize_facts("o1", [fact_run_outcome(outcome=outcome)], start_sequence=4)
    )
    assert view is not None
    assert view.root_runs[0].status == RunStatus.COMPLETED

    with pytest.raises(ProjectionError):
        project_objective(
            base
            + materialize_facts(
                "o1",
                [
                    fact_run_outcome(outcome=outcome),
                    fact_run_outcome(
                        outcome=RunOutcome(
                            run_id="r1",
                            role=RunRole.WORK,
                            terminal_status=RunStatus.COMPLETED,
                            reason="again",
                            report="r2",
                            outcome_id="out2",
                        ),
                    ),
                ],
                start_sequence=4,
            )
        )


def test_context_capacity_and_externalize_keeps_original_input() -> None:
    budget = _budget()
    facts = materialize_facts(
        "o1",
        [
            fact_objective_created(session_id="s1", budget=budget),
            fact_objective_user_input(
                user_input=ObjectiveUserInput(
                    input_id="i1", message=_user("long text")
                ),
            ),
            fact_context_capacity_exceeded(
                context_limit_tokens=100,
                estimated_input_tokens=200,
                externalizable_input_ids=["i1"],
            ),
            fact_artifact_registered(
                artifact=Artifact(
                    artifact_id="art1",
                    path="sessions/s1/artifacts/art1.txt",
                    summary="long text summary",
                    source_input_id="i1",
                    content_hash="h1",
                ),
            ),
            fact_user_input_externalized(
                input_id="i1",
                artifact_id="art1",
            ),
        ],
    )
    view = project_objective(facts)
    assert view is not None
    assert view.status == ObjectiveStatus.WAITING_USER
    assert len(view.user_inputs) == 1
    assert view.user_inputs[0].message.extract_text() == "long text"
    assert len(view.externalized_inputs) == 1
    assert view.externalized_inputs[0].path.endswith("art1.txt")


def test_sequence_gap_fails() -> None:
    budget = _budget()
    facts = [
        fact_objective_created(session_id="s1", budget=budget).materialize(
            objective_id="o1", sequence=1
        ),
        fact_objective_user_input(
            user_input=ObjectiveUserInput(input_id="i1", message=_user()),
        ).materialize(objective_id="o1", sequence=3),
    ]
    with pytest.raises(ProjectionError):
        project_objective(facts)


def test_two_active_root_runs_fail() -> None:
    budget = _budget()
    facts = materialize_facts(
        "o1",
        [
            fact_objective_created(session_id="s1", budget=budget),
            fact_root_run_requested(
                run_id="r1",
                role=RunRole.WORK,
            ),
            fact_root_run_requested(
                run_id="r2",
                role=RunRole.WORK,
            ),
        ],
    )
    with pytest.raises(ProjectionError):
        project_objective(facts)


def test_delivered_requires_existing_outcome() -> None:
    budget = _budget()
    facts = materialize_facts(
        "o1",
        [
            fact_objective_created(session_id="s1", budget=budget),
            fact_objective_delivered(
                final_outcome_id="missing",
                report="r",
            ),
        ],
    )
    with pytest.raises(ProjectionError):
        project_objective(facts)


def test_outcome_contributions_and_goal_revision_projection() -> None:
    """Outcome-sourced contributions and goal revisions become facts then project."""

    budget = _budget()
    base = materialize_facts(
        "o1",
        [
            fact_objective_created(session_id="s1", budget=budget),
            fact_root_run_requested(
                run_id="r1",
                role=RunRole.WORK,
            ),
            fact_root_run_started(
                run_id="r1",
            ),
            fact_contribution_created(
                contribution=ObjectiveContribution(
                    contribution_id="c1", content="found X", summary="X"
                ),
                source_run_id="r1",
            ),
            fact_contribution_annotated(
                annotation=ContributionAnnotation(
                    contribution_id="c1",
                    annotation="still relevant",
                    from_run_id="r1",
                    time=__import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    ),
                ),
            ),
            fact_current_goal_revised(
                analysis=CurrentGoalAnalysis(revision=1, intent="do X carefully"),
            ),
        ],
    )
    outcome = RunOutcome(
        run_id="r1",
        role=RunRole.WORK,
        terminal_status=RunStatus.COMPLETED,
        reason="done",
        report="report",
        outcome_id="out1",
        new_contributions=(NewContributionSpec(content="found Y", summary="Y"),),
        objective_update=ObjectiveUpdateSpec(
            expected_revision=1, intent="refine scope"
        ),
    )
    derived = expand_outcome_derived_facts(outcome=outcome)
    batch = FactBatch(objective_id="o1", start_sequence=7)
    batch.add_many(derived)
    batch.add(fact_run_outcome(outcome=outcome))
    facts = base + batch.facts
    view = project_objective(facts)
    assert view is not None
    assert len(view.contributions) == 2
    assert {c.content for c in view.contributions} == {"found X", "found Y"}
    assert view.contributions[0].annotations or view.contributions[1].annotations
    assert view.current_goal is not None
    assert view.current_goal.revision == 2
    assert view.current_goal.intent == "refine scope"
    assert view.user_inputs == ()


def test_paused_root_run_rejects_outcome() -> None:
    budget = _budget()
    facts = materialize_facts(
        "o1",
        [
            fact_objective_created(session_id="s1", budget=budget),
            fact_root_run_requested(
                run_id="r1",
                role=RunRole.WORK,
            ),
            fact_root_run_started(
                run_id="r1",
            ),
            fact_root_run_paused(
                run_id="r1",
                reason="user pause",
            ),
            fact_run_outcome(
                outcome=RunOutcome(
                    run_id="r1",
                    role=RunRole.WORK,
                    terminal_status=RunStatus.COMPLETED,
                    reason="bad",
                    report="r",
                    outcome_id="out1",
                ),
            ),
        ],
    )
    with pytest.raises(ProjectionError):
        project_objective(facts)


def test_verification_required_latch() -> None:
    budget = _budget()
    facts = materialize_facts(
        "o1",
        [
            fact_objective_created(session_id="s1", budget=budget),
            fact_verification_required_set(
                run_id="r1",
                milestone_id="m1",
            ),
        ],
    )
    view = project_objective(facts)
    assert view is not None
    assert view.verification_required is True
