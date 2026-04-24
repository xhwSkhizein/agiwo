"""Review Enforcer — trigger detection and system-review notice injection."""

from enum import Enum

from agiwo.agent.models.review import Milestone, ReviewState


class ReviewTrigger(Enum):
    """Which condition fired the review."""

    NONE = "none"
    STEP_INTERVAL = "step_interval"
    CONSECUTIVE_ERRORS = "consecutive_errors"
    MILESTONE_SWITCH = "milestone_switch"


def _build_review_notice(
    milestone: Milestone | None,
    step_count: int,
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
        f"Steps since last review: {step_count}\n\n"
        f"Question: Do your recent steps meaningfully advance the current goal?\n"
        f"If not, use review_trajectory to:\n"
        f"  1. Indicate misalignment (aligned=false)\n"
        f"  2. Provide a concise experience summary of what was learned\n"
        f"If aligned, use review_trajectory with aligned=true and a brief note."
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
    current_seq: int = 0,
) -> ReviewTrigger:
    """Check if a review should be triggered. Returns the trigger type or NONE."""
    if not enabled:
        return ReviewTrigger.NONE
    if tool_name == "review_trajectory":
        return ReviewTrigger.NONE

    # Milestone switch (agent just completed/activated a milestone).
    # Initial milestone declarations should not immediately trigger a review.
    if state.is_review_pending and tool_name != "declare_milestones":
        return ReviewTrigger.MILESTONE_SWITCH

    # Consecutive errors
    if is_error and state.consecutive_errors >= error_threshold:
        return ReviewTrigger.CONSECUTIVE_ERRORS

    # Step interval since last review
    steps_since_review = current_seq - state.last_review_seq
    if steps_since_review >= step_interval:
        return ReviewTrigger.STEP_INTERVAL

    return ReviewTrigger.NONE


def inject_system_review(
    content: str,
    milestone: Milestone | None,
    step_count: int,
) -> str:
    """Append a <system-review> notice to the tool result content.

    Returns content unchanged when no notice should be injected.
    """
    notice = _build_review_notice(milestone, step_count)
    return content + notice


__all__ = [
    "ReviewTrigger",
    "check_review_trigger",
    "inject_system_review",
]
