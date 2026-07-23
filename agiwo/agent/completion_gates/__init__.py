"""Completion gates — mechanical first (G-v1a), semantic seam reserved."""

from agiwo.agent.completion_gates.mechanical import (
    build_mechanical_feedback,
    evaluate_mechanical,
    open_milestones,
)
from agiwo.agent.completion_gates.models import AllowComplete, Continue, GateDecision
from agiwo.agent.completion_gates.semantic import (
    DisabledSemanticCompletionGate,
    SemanticCompletionGate,
)
from agiwo.agent.models.plan import RunPlan


class CompletionGates:
    """Evaluate stop boundary checks before a ROOT Run may complete."""

    def __init__(
        self,
        *,
        enable_semantic: bool = False,
        semantic: SemanticCompletionGate | None = None,
    ) -> None:
        self._enable_semantic = enable_semantic
        self._semantic = semantic or DisabledSemanticCompletionGate()

    @property
    def semantic_enabled(self) -> bool:
        return self._enable_semantic

    async def evaluate(
        self,
        *,
        plan: RunPlan,
        active_worker_ids: frozenset[str],
    ) -> GateDecision:
        mechanical = evaluate_mechanical(
            plan=plan,
            active_worker_ids=active_worker_ids,
        )
        if isinstance(mechanical, Continue):
            return mechanical
        if not self._enable_semantic:
            return AllowComplete()
        semantic = await self._semantic.evaluate(
            plan=plan,
            active_worker_ids=active_worker_ids,
        )
        if semantic is None:
            return AllowComplete()
        return semantic


__all__ = [
    "AllowComplete",
    "CompletionGates",
    "Continue",
    "DisabledSemanticCompletionGate",
    "GateDecision",
    "SemanticCompletionGate",
    "build_mechanical_feedback",
    "evaluate_mechanical",
    "open_milestones",
]
