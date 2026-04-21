"""Shared formatting helpers for scheduler-facing child-agent text."""

from agiwo.agent import (
    ChannelContext,
    ContentPart,
    ContentType,
    UserInput,
    UserMessage,
)
from agiwo.scheduler.models import (
    PendingEvent,
    SchedulerEventType,
    WakeCondition,
    WakeType,
)


def _system_notice(content: str) -> str:
    """Render a ``<system-notice>`` tag.

    Local mirror of ``agiwo.agent.prompt.system_notice`` so the scheduler
    layer does not reach into internal agent execution modules (enforced by
    ``repo_guard AGW003``).
    """
    return f"<system-notice>{content}</system-notice>"


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


def build_events_message(events: tuple[PendingEvent, ...]) -> UserMessage:  # noqa: PLR0915
    """Build the user-facing message for pending event notifications.

    Returns a ``UserMessage`` so that attachments and ``ChannelContext``
    carried by USER_HINT events survive into the next run.  Child-agent
    events are rendered into the narrative text block; USER_HINT events
    contribute both text and non-text ``ContentPart`` entries.
    """
    lines = [f"You have {len(events)} new notification(s):", ""]
    extra_parts: list[ContentPart] = []
    channel_context: ChannelContext | None = None
    for event in events:
        event_label = event.event_type.value.replace("_", " ").title()
        child_id = event.source_agent_id or "unknown"
        header = f"### {event_label} - Agent: {child_id}"
        if event.event_type == SchedulerEventType.CHILD_SLEEP_RESULT:
            payload = event.get_payload_child_sleep_result()
            if payload:
                child_id = payload.child_agent_id
                header = f"### {event_label} - Agent: {child_id}"
                lines.append(header)
                lines.extend(
                    build_child_result_detail_lines(
                        result=payload.result,
                        explain=payload.explain,
                        periodic=payload.periodic,
                        result_as_block=True,
                    )
                )
            else:
                lines.append(header)
                lines.append("(invalid payload)")
        elif event.event_type == SchedulerEventType.CHILD_COMPLETED:
            payload = event.get_payload_child_completed()
            if payload:
                child_id = payload.child_agent_id
                header = f"### {event_label} - Agent: {child_id}"
                lines.append(header)
                lines.extend(
                    build_child_result_detail_lines(
                        result=payload.result,
                        result_as_block=True,
                    )
                )
            else:
                lines.append(header)
                lines.append("(invalid payload)")
        elif event.event_type == SchedulerEventType.CHILD_FAILED:
            payload = event.get_payload_child_failed()
            if payload:
                child_id = payload.child_agent_id
                header = f"### {event_label} - Agent: {child_id}"
                lines.append(header)
                lines.extend(
                    build_child_result_detail_lines(failure_reason=payload.reason)
                )
            else:
                lines.append(header)
                lines.append("(invalid payload)")
        elif event.event_type == SchedulerEventType.USER_HINT:
            lines.append(header)
            hint_message = _decode_user_hint_message(event)
            if hint_message is None:
                lines.append("User hint: (undecodable payload)")
            else:
                for part in hint_message.content:
                    if part.type == ContentType.TEXT:
                        if part.text and part.text.strip():
                            lines.append(f"User hint: {part.text.strip()}")
                    else:
                        extra_parts.append(part)
                if channel_context is None and hint_message.context is not None:
                    channel_context = hint_message.context
            lines.append("")
        else:
            lines.append(header)
            lines.append("")
            continue
        if event.event_type != SchedulerEventType.USER_HINT:
            lines.append("")
    lines.append(
        "Please review these notifications and take appropriate action "
        "(e.g., summarize results for the user, cancel stuck agents, etc.)."
    )
    text_body = "\n".join(lines)
    parts: list[ContentPart] = [ContentPart(type=ContentType.TEXT, text=text_body)]
    parts.extend(extra_parts)
    return UserMessage(content=parts, context=channel_context)


def _decode_user_hint_message(event: PendingEvent) -> UserMessage | None:
    payload = event.get_payload_user_hint()
    if payload is None:
        return None
    stored = payload.user_input
    decoded = UserMessage.from_storage_value(stored)
    if decoded is None:
        return None
    message = UserMessage.from_value(decoded)
    return message if message.has_content() else None


def summarize_text(text: str | None, limit: int) -> str | None:
    """Trim long text consistently for scheduler responses."""
    if text is None:
        return None
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


_FORK_NOTICE = _system_notice(
    "You are a forked child agent. Your conversation history has been "
    "inherited from the parent agent. Do NOT use spawn_agent — it is "
    "unavailable to you. Complete the following task directly."
)


def build_fork_task_notice(task: UserInput) -> str:
    """Wrap a task with a system notice for forked child agents."""
    task_text = UserMessage.serialize(task)
    return f"{_FORK_NOTICE}\n\n{task_text}"
