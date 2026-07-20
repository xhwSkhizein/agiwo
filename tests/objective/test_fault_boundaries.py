"""P4-02/P4-03: mechanical fault boundaries respect handoff quotas."""

from agiwo.agent.models.finalization import mechanical_user_boundary_result
from agiwo.agent.models.run import RunStatus
from agiwo.agent.retry import (
    ExecutionFault,
    FaultDisposition,
    RunBlockingFaultError,
    finalization_for_blocking_fault,
)
from agiwo.objective.budget import consumption_for_decision
from agiwo.objective.finalization import map_finalization_to_outcome
from agiwo.objective.log import (
    fact_objective_created,
    fact_objective_status_changed,
    fact_root_run_requested,
    fact_root_run_started,
    materialize_facts,
)
from agiwo.objective.models import (
    HandoffTarget,
    ObjectiveBudget,
    ObjectiveStatus,
    RunRole,
)
from agiwo.objective.projection import project_objective


def _running_view():
    facts = materialize_facts(
        "o1",
        [
            fact_objective_created(
                session_id="s1",
                budget=ObjectiveBudget.create(
                    handoffs=1,
                    verification_attempts=1,
                    llm_cost_usd=1,
                    active_seconds=60,
                ),
            ),
            fact_objective_status_changed(
                from_status=ObjectiveStatus.CREATED,
                to_status=ObjectiveStatus.RUNNING,
                reason="start",
            ),
            fact_root_run_requested(run_id="r1", role=RunRole.WORK),
            fact_root_run_started(run_id="r1"),
        ],
    )
    return project_objective(facts, objective_id="o1")


def test_user_boundary_ignores_handoff_quota() -> None:
    view = _running_view()
    assert view is not None
    finalization = mechanical_user_boundary_result(
        "auth failed", reason="system_non_retryable"
    )
    outcome, decision = map_finalization_to_outcome(
        view=view,
        run_id="r1",
        role=RunRole.WORK,
        result=finalization,
        terminal_status=RunStatus.INTERRUPTED,
    )
    assert decision.target is HandoffTarget.USER
    assert decision.expects_reply is True
    plan = consumption_for_decision(decision)
    assert plan.is_empty
    assert outcome.terminal_status is RunStatus.INTERRUPTED


def test_retry_exhausted_consumes_agent_handoff() -> None:
    facts = materialize_facts(
        "o1",
        [
            fact_objective_created(
                session_id="s1",
                budget=ObjectiveBudget.create(
                    handoffs=2,
                    verification_attempts=1,
                    llm_cost_usd=1,
                    active_seconds=60,
                ),
            ),
            fact_objective_status_changed(
                from_status=ObjectiveStatus.CREATED,
                to_status=ObjectiveStatus.RUNNING,
                reason="start",
            ),
            fact_root_run_requested(run_id="r1", role=RunRole.WORK),
            fact_root_run_started(run_id="r1"),
        ],
    )
    view = project_objective(facts, objective_id="o1")
    assert view is not None
    fault = ExecutionFault(
        operation="llm",
        disposition=FaultDisposition.RETRYABLE,
        run_blocking=True,
    )
    finalization = finalization_for_blocking_fault(
        RunBlockingFaultError(fault, attempts=[fault, fault], exhausted=True),
        carry_forward=[{"id": "m1", "description": "x", "status": "active"}],
    )
    outcome, decision = map_finalization_to_outcome(
        view=view,
        run_id="r1",
        role=RunRole.WORK,
        result=finalization,
        terminal_status=RunStatus.INTERRUPTED,
    )
    assert decision.target is HandoffTarget.AGENT
    assert not consumption_for_decision(decision).is_empty
    assert outcome.terminal_status is RunStatus.INTERRUPTED
