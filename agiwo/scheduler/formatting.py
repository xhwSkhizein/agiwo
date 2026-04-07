"""Shared formatting helpers for scheduler-facing child-agent text."""

from agiwo.agent import UserInput
from agiwo.scheduler.models import (
    PendingEvent,
    SchedulerEventType,
    WakeCondition,
    WakeType,
)

SHUTDOWN_SUMMARY_TASK = (
    "System shutdown requested. Please produce a final summary "
    "report of all work done so far."
)


def format_child_results_summary(
    *,
    header: str,
    succeeded: dict[str, str],
    failed: dict[str, str],
    closing_instruction: str,
    progress_line: str | None = None,
) -> str:
    """Format a child-results report used by wake and timeout messages."""
    lines = [header]
    if progress_line:
        lines.extend(["", progress_line])
    if succeeded:
        lines.extend(["", "## Successful Results"])
        lines.extend(f"- [{cid}] {summary}" for cid, summary in succeeded.items())
    if failed:
        lines.extend(["", "## Failed Agents"])
        lines.extend(f"- [{cid}] FAILED: {reason}" for cid, reason in failed.items())
    lines.extend(["", closing_instruction])
    return "\n".join(lines)


def build_child_result_detail_lines(
    *,
    result: str | None = None,
    explain: str | None = None,
    failure_reason: str | None = None,
    periodic: bool = False,
    result_as_block: bool = False,
) -> list[str]:
    """Build consistently worded detail lines for child-agent outcomes."""
    lines: list[str] = []
    if periodic:
        lines.append("(Periodic check completed)")
    if explain:
        lines.append(f"Sleep reason: {explain}")
    if result:
        if result_as_block:
            lines.append(f"Result:\n{result}")
        else:
            lines.append(f"Result: {result}")
    if failure_reason:
        lines.append(f"Failure reason: {failure_reason}")
    return lines


def build_wake_message(
    wake_condition: WakeCondition | None,
    succeeded: dict[str, str],
    failed: dict[str, str],
) -> str:
    """Build the user-facing message when a sleeping agent is woken."""
    if wake_condition is None:
        return "You have been woken up. Please continue your task."

    if wake_condition.type == WakeType.WAITSET:
        done = len(succeeded) + len(failed)
        return format_child_results_summary(
            header=f"Child agents completed ({done}/{len(wake_condition.wait_for)}).",
            succeeded=succeeded,
            failed=failed,
            closing_instruction=(
                "Please synthesize a final response based on the successful "
                "results above."
            ),
        )
    if wake_condition.type == WakeType.TIMER:
        return "The scheduled delay has elapsed. Please continue your task."
    if wake_condition.type == WakeType.PERIODIC:
        return (
            "A scheduled periodic check has triggered. "
            "Please check progress and decide whether to continue waiting "
            "or produce a final result."
        )
    return "You have been woken up. Please continue your task."


def build_timeout_message(
    wake_condition: WakeCondition | None,
    succeeded: dict[str, str],
    failed: dict[str, str],
) -> str:
    """Build the user-facing message when a wait times out."""
    done = len(succeeded) + len(failed)
    return format_child_results_summary(
        header="Wait timeout reached.",
        succeeded=succeeded,
        failed=failed,
        progress_line=f"Completed children: {done}/{len(wake_condition.wait_for) if wake_condition else 0}",
        closing_instruction=(
            "Please produce a summary report with whatever results are available."
        ),
    )


def build_events_message(events: tuple[PendingEvent, ...]) -> str:
    """Build the user-facing message for pending event notifications."""
    lines = [f"You have {len(events)} new notification(s):\n"]
    for event in events:
        event_label = event.event_type.value.replace("_", " ").title()
        child_id = event.payload.get(
            "child_agent_id", event.source_agent_id or "unknown"
        )
        lines.append(f"### {event_label} - Agent: {child_id}")
        if event.event_type == SchedulerEventType.CHILD_SLEEP_RESULT:
            lines.extend(
                build_child_result_detail_lines(
                    result=event.payload.get("result", ""),
                    explain=event.payload.get("explain"),
                    periodic=event.payload.get("periodic", False),
                    result_as_block=True,
                )
            )
        elif event.event_type == SchedulerEventType.CHILD_COMPLETED:
            lines.extend(
                build_child_result_detail_lines(
                    result=event.payload.get("result", ""),
                    result_as_block=True,
                )
            )
        elif event.event_type == SchedulerEventType.CHILD_FAILED:
            lines.extend(
                build_child_result_detail_lines(
                    failure_reason=event.payload.get("reason", "Unknown failure")
                )
            )
        elif event.event_type == SchedulerEventType.USER_HINT:
            hint = event.payload.get("hint", "")
            if hint:
                lines.append(f"User hint: {hint}")
        lines.append("")
    lines.append(
        "Please review these notifications and take appropriate action "
        "(e.g., summarize results for the user, cancel stuck agents, etc.)."
    )
    return "\n".join(lines)


def summarize_text(text: str | None, limit: int) -> str | None:
    """Trim long text consistently for scheduler responses."""
    if text is None:
        return None
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


_FORK_NOTICE = (
    "<system-notice>\n"
    "You are a forked child agent. Your conversation history has been "
    "inherited from the parent agent. Do NOT use spawn_agent — it is "
    "unavailable to you. Complete the following task directly.\n"
    "</system-notice>"
)


def build_fork_task_notice(task: UserInput) -> str:
    """Wrap a task with a system notice for forked child agents."""
    task_text = task if isinstance(task, str) else str(task)
    return f"{_FORK_NOTICE}\n\n{task_text}"
