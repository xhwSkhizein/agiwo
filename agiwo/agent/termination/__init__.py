"""Termination-summary helpers for interrupted agent runs."""

from agiwo.agent.termination.prompts import (
    DEFAULT_TERMINATION_USER_PROMPT,
    TERMINATION_SUMMARY_REASONS,
    format_termination_reason,
    render_termination_summary_prompt,
)
from agiwo.agent.termination.limits import (
    check_non_recoverable_limits,
    check_post_llm_limits,
)
from agiwo.agent.termination.run_limit import RunLimitDecision, RunLimitPolicy


def __getattr__(name: str) -> object:
    if name == "maybe_generate_termination_summary":
        from agiwo.agent.termination.summarizer import (  # noqa: PLC0415
            maybe_generate_termination_summary,
        )

        return maybe_generate_termination_summary
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "check_non_recoverable_limits",
    "check_post_llm_limits",
    "DEFAULT_TERMINATION_USER_PROMPT",
    "TERMINATION_SUMMARY_REASONS",
    "format_termination_reason",
    "maybe_generate_termination_summary",
    "render_termination_summary_prompt",
    "RunLimitDecision",
    "RunLimitPolicy",
]
