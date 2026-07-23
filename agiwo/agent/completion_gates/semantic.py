"""Semantic completion gate seam (I2) — reserved, default off in G-v1a."""

from typing import Protocol

from agiwo.agent.completion_gates.models import GateDecision
from agiwo.agent.models.plan import RunPlan


class SemanticCompletionGate(Protocol):
    """Optional LLM-backed gate; not invoked unless explicitly enabled."""

    async def evaluate(
        self,
        *,
        plan: RunPlan,
        active_worker_ids: frozenset[str],
    ) -> GateDecision | None:
        """Return a gate decision, or ``None`` to defer to AllowComplete."""


class DisabledSemanticCompletionGate:
    """Default semantic gate implementation — never blocks and makes no LLM calls."""

    async def evaluate(
        self,
        *,
        plan: RunPlan,
        active_worker_ids: frozenset[str],
    ) -> GateDecision | None:
        del plan, active_worker_ids
        return None


__all__ = ["DisabledSemanticCompletionGate", "SemanticCompletionGate"]
