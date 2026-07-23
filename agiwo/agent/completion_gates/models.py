"""Completion gate decision models (G-v1a mechanical + semantic seam)."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AllowComplete:
    """Mechanical and semantic checks passed; the Run may stop."""


@dataclass(frozen=True, slots=True)
class Continue:
    """Stop blocked; feedback should re-enter the Loop queue."""

    feedback_text: str


GateDecision = AllowComplete | Continue


__all__ = ["AllowComplete", "Continue", "GateDecision"]
