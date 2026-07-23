"""Wave E-01: mechanical CompletionGates matrix unit tests."""

import pytest

from agiwo.agent.completion_gates import (
    AllowComplete,
    CompletionGates,
    Continue,
    evaluate_mechanical,
)
from agiwo.agent.models.plan import Milestone, RunPlan


def _plan(*statuses: str) -> RunPlan:
    return RunPlan(
        milestones=[
            Milestone(id=f"m{i}", description=f"task {i}", status=status)
            for i, status in enumerate(statuses)
        ]
    )


@pytest.mark.parametrize(
    ("plan", "workers", "expected"),
    [
        (_plan(), frozenset(), AllowComplete),
        (_plan("completed"), frozenset(), AllowComplete),
        (_plan("pending"), frozenset(), Continue),
        (_plan("active"), frozenset(), Continue),
        (_plan("completed", "pending"), frozenset(), Continue),
        (_plan(), frozenset({"worker-a"}), Continue),
        (_plan("pending"), frozenset({"worker-a"}), Continue),
        (_plan("completed"), frozenset({"worker-a"}), Continue),
    ],
)
def test_mechanical_gate_matrix(plan, workers, expected) -> None:
    decision = evaluate_mechanical(plan=plan, active_worker_ids=workers)
    assert isinstance(decision, expected)
    if isinstance(decision, Continue):
        if workers:
            assert "Unfinished Workers remain" in decision.feedback_text
        if any(m.status in {"pending", "active"} for m in plan.milestones):
            assert "unfinished milestones" in decision.feedback_text


@pytest.mark.asyncio
async def test_semantic_gate_stub_not_invoked_by_default() -> None:
    called = False

    class _BlockingSemantic:
        async def evaluate(self, *, plan, active_worker_ids):
            del plan, active_worker_ids
            nonlocal called
            called = True
            return Continue(feedback_text="semantic block")

    gates = CompletionGates(enable_semantic=False, semantic=_BlockingSemantic())
    decision = await gates.evaluate(plan=_plan(), active_worker_ids=frozenset())
    assert isinstance(decision, AllowComplete)
    assert called is False
    assert gates.semantic_enabled is False


@pytest.mark.asyncio
async def test_semantic_gate_runs_only_when_enabled() -> None:
    class _BlockingSemantic:
        async def evaluate(self, *, plan, active_worker_ids):
            del plan, active_worker_ids
            return Continue(feedback_text="semantic block")

    gates = CompletionGates(enable_semantic=True, semantic=_BlockingSemantic())
    decision = await gates.evaluate(plan=_plan(), active_worker_ids=frozenset())
    assert isinstance(decision, Continue)
    assert decision.feedback_text == "semantic block"
