"""Deterministic Run report summarization for SessionIntent."""

REPORT_SUMMARY_MAX_LEN = 500


def summarize_run_report(
    *,
    response: str | None,
    max_len: int = REPORT_SUMMARY_MAX_LEN,
) -> str:
    """Return a truncated user-visible Run outcome summary."""
    if not response:
        return ""
    text = response.strip()
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return f"{text[: max_len - 3]}..."


__all__ = ["REPORT_SUMMARY_MAX_LEN", "summarize_run_report"]
