"""Runtime wiring for completion gates inside the Loop."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field


@dataclass
class CompletionGateContext:
    """Optional Loop-bound dependencies for mechanical gate checks."""

    active_worker_ids: Callable[[], frozenset[str]] = field(
        default_factory=lambda: lambda: frozenset()
    )
    on_gate_feedback: Callable[[str], Awaitable[None]] | None = None


__all__ = ["CompletionGateContext"]
