"""Shared formatting helpers for scheduler-facing child-agent text."""


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


def summarize_text(text: str | None, limit: int) -> str | None:
    """Trim long text consistently for scheduler responses."""
    if text is None:
        return None
    if len(text) <= limit:
        return text
    return text[:limit] + "..."
