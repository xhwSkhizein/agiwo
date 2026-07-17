"""Optional experimental hints from trajectory review for compaction."""

from agiwo.agent.introspect.models import ToolUsefulnessEntry


def format_experimental_usefulness_hint(
    entries: list[ToolUsefulnessEntry],
) -> str:
    if not entries:
        return ""
    lines = [
        "Experimental/unverified trajectory review usefulness scores "
        "(do not treat as deterministic delete rules):"
    ]
    for entry in entries:
        label = entry.tool_name or "tool"
        if entry.score is None:
            lines.append(f"- {entry.tool_call_id} ({label}): unknown")
        else:
            lines.append(f"- {entry.tool_call_id} ({label}): score {entry.score}")
    return "\n".join(lines)


__all__ = ["format_experimental_usefulness_hint"]
