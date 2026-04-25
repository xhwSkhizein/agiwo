"""Review Enforcer — trigger detection and system-review notice injection."""

from enum import Enum
import re

from agiwo.agent.models.review import Milestone, ReviewState

_SYSTEM_REVIEW_BLOCK_RE = re.compile(
    r"\n*<system-review>\s*.*?\s*</system-review>\s*",
    re.DOTALL,
)


class ReviewTrigger(Enum):
    """Which condition fired the review."""

    NONE = "none"
    STEP_INTERVAL = "step_interval"
    CONSECUTIVE_ERRORS = "consecutive_errors"
    MILESTONE_SWITCH = "milestone_switch"


def _build_review_notice(
    milestone: Milestone | None,
    step_count: int,
    *,
    trigger_reason: str,
    review_advice: str | None = None,
) -> str:
    """Build the <system-review> notice text."""
    if milestone is not None:
        milestone_text = f'Active milestone: "{milestone.description}"'
    else:
        milestone_text = (
            "No active milestone declared. Consider using declare_milestones."
        )

    inner_text = (
        f"{milestone_text}\n\n"
        f"Trigger: {trigger_reason}\n"
        f"Steps since last review: {step_count}\n"
    )
    if review_advice:
        inner_text += f"Hook advice: {review_advice}\n"

    inner_text += (
        "\n"
        "Question: Do your recent steps meaningfully advance the current goal?\n"
        "If not, use review_trajectory to:\n"
        "  1. Indicate misalignment (aligned=false)\n"
        "  2. Provide a concise experience summary of what was learned\n"
        "If aligned, use review_trajectory with aligned=true and a brief note."
    )

    return f"\n\n<system-review>\n{inner_text}\n</system-review>"


def check_review_trigger(
    *,
    state: ReviewState,
    enabled: bool,
    is_error: bool,
    step_interval: int,
    error_threshold: int,
    tool_name: str = "",
) -> ReviewTrigger:
    """Check if a review should be triggered. Returns the trigger type or NONE."""
    if not enabled:
        return ReviewTrigger.NONE
    if tool_name == "review_trajectory":
        return ReviewTrigger.NONE

    # Milestone switch (agent just completed/activated a milestone).
    # Initial milestone declarations should not immediately trigger a review.
    if (
        state.pending_review_reason == "milestone_switch"
        and tool_name != "declare_milestones"
    ):
        return ReviewTrigger.MILESTONE_SWITCH

    # Consecutive errors
    if is_error and state.consecutive_errors >= error_threshold:
        return ReviewTrigger.CONSECUTIVE_ERRORS

    # Tool count since last checkpoint/review.
    if state.review_count_since_checkpoint >= step_interval:
        return ReviewTrigger.STEP_INTERVAL

    return ReviewTrigger.NONE


def inject_system_review(
    content: str,
    milestone: Milestone | None,
    step_count: int,
    *,
    trigger_reason: str,
    review_advice: str | None = None,
) -> str:
    """Append a <system-review> notice to the tool result content.

    This always appends the generated notice block.
    """
    notice = _build_review_notice(
        milestone,
        step_count,
        trigger_reason=trigger_reason,
        review_advice=review_advice,
    )
    return content + notice


def strip_system_review_notices(content: str) -> str:
    """Remove prompt-visible system-review notices from tool result content."""
    return _SYSTEM_REVIEW_BLOCK_RE.sub("", content).rstrip()


__all__ = [
    "ReviewTrigger",
    "check_review_trigger",
    "inject_system_review",
    "strip_system_review_notices",
]
