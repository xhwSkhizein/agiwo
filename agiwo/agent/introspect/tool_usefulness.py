"""Parse tool usefulness payloads from review_trajectory output."""

from agiwo.agent.introspect.models import ToolUsefulnessEntry


def parse_tool_usefulness_output(
    raw_entries: object,
) -> list[ToolUsefulnessEntry]:
    if not isinstance(raw_entries, list):
        return []
    entries: list[ToolUsefulnessEntry] = []
    for item in raw_entries:
        if not isinstance(item, dict):
            continue
        tool_call_id = item.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            continue
        tool_name = item.get("tool_name")
        score = item.get("score")
        if score is not None and not isinstance(score, int):
            continue
        entries.append(
            ToolUsefulnessEntry(
                tool_call_id=tool_call_id,
                tool_name=tool_name if isinstance(tool_name, str) else None,
                score=score,
            )
        )
    return entries


__all__ = ["parse_tool_usefulness_output"]
