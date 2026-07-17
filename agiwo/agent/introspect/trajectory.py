"""Trajectory introspection trigger and outcome rules."""

import re

from agiwo.agent.introspect.apply import parse_tool_usefulness_output
from agiwo.agent.introspect.models import (
    IntrospectionNotice,
    IntrospectionOutcome,
    IntrospectionState,
    IntrospectionTriggerReason,
    Milestone,
)
from agiwo.agent.models.plan import RunPlan
from agiwo.tool.base import ToolResult

_SYSTEM_REVIEW_BLOCK_RE = re.compile(
    r"\n*<system-review>\s*.*?\s*</system-review>\s*",
    re.DOTALL,
)


def _build_review_notice(
    milestone: Milestone | None,
    step_count: int,
    *,
    trigger_reason: str,
    review_advice: str | None = None,
) -> str:
    milestone_text = (
        f'Active milestone: "{milestone.description}"'
        if milestone is not None
        else "No active milestone declared. Consider using update_plan."
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
        "Use review_trajectory with aligned=true when on track, or aligned=false "
        "with a concise experience summary when drifted. Optionally score each tool "
        "in the review window via tool_usefulness (0=harmful, 1=low, 2=useful, "
        "3=critical). Omit tools you cannot judge."
    )
    return f"\n\n<system-review>\n{inner_text}\n</system-review>"


def append_system_review_notice(
    content: str,
    milestone: Milestone | None,
    step_count: int,
    *,
    trigger_reason: str,
    review_advice: str | None = None,
) -> str:
    return content + _build_review_notice(
        milestone,
        step_count,
        trigger_reason=trigger_reason,
        review_advice=review_advice,
    )


def strip_system_review_notices(content: str) -> str:
    return _SYSTEM_REVIEW_BLOCK_RE.sub("", content).rstrip()


def has_system_review_notice(content: object) -> bool:
    return isinstance(content, str) and "<system-review>" in content


def has_prompt_visible_system_review(messages: list[dict[str, object]]) -> bool:
    for message in messages:
        if message.get("role") != "tool":
            continue
        if has_system_review_notice(message.get("content")):
            return True
    return False


def maybe_build_introspection_notice(
    result: ToolResult,
    plan: RunPlan,
    state: IntrospectionState,
    *,
    step_interval: int,
    review_on_error: bool,
    has_visible_notice: bool = False,
    error_threshold: int = 2,
) -> IntrospectionNotice | None:
    if result.tool_name == "review_trajectory":
        return None
    state.review_count_since_boundary += 1
    if result.is_success:
        state.consecutive_errors = 0
    else:
        state.consecutive_errors += 1

    if state.notice_requested or has_visible_notice:
        return None

    trigger_reason: IntrospectionTriggerReason | None = None
    if state.pending_milestone_switch and result.tool_name != "update_plan":
        trigger_reason = "milestone_switch"
    elif (
        review_on_error
        and not result.is_success
        and state.consecutive_errors >= error_threshold
    ):
        trigger_reason = "consecutive_errors"
    elif state.review_count_since_boundary >= step_interval:
        trigger_reason = "step_interval"

    if trigger_reason is None:
        return None
    state.pending_milestone_switch = False
    state.notice_requested = True
    milestone = plan.active_milestone
    return IntrospectionNotice(
        content=append_system_review_notice(
            result.content or "",
            milestone,
            state.review_count_since_boundary,
            trigger_reason=trigger_reason,
        ),
        active_milestone=milestone,
        step_count=state.review_count_since_boundary,
        trigger_reason=trigger_reason,
    )


def parse_introspection_outcome(
    result: ToolResult,
    plan: RunPlan,
    *,
    current_seq: int,
    assistant_step_id: str | None,
    tool_step_id: str | None,
) -> IntrospectionOutcome | None:
    del assistant_step_id
    if result.tool_name != "review_trajectory" or not result.is_success:
        return None
    output = result.output if isinstance(result.output, dict) else {}
    aligned = output.get("aligned")
    experience_value = output.get("experience")
    experience = experience_value if isinstance(experience_value, str) else None
    if experience == "":
        experience = None
    return IntrospectionOutcome(
        aligned=aligned if isinstance(aligned, bool) else None,
        boundary_seq=current_seq,
        experience=experience,
        tool_usefulness=parse_tool_usefulness_output(output.get("tool_usefulness")),
        active_milestone_id=plan.active_milestone_id,
        review_tool_call_id=result.tool_call_id or None,
        review_step_id=tool_step_id,
    )


__all__ = [
    "append_system_review_notice",
    "has_prompt_visible_system_review",
    "has_system_review_notice",
    "maybe_build_introspection_notice",
    "parse_introspection_outcome",
    "strip_system_review_notices",
]
