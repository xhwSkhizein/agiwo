"""
Channel utilities — shared helpers used by channel service implementations.
"""

from agiwo.agent import (
    AgentStreamItem,
    RunCompletedEvent,
    RunFailedEvent,
    StepDeltaEvent,
)
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)

_MAX_CHUNK_LEN = 6000
_MAX_LOG_TEXT_LEN = 1200


def truncate_for_log(text: str, max_len: int = _MAX_LOG_TEXT_LEN) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "...[truncated]"


async def safe_close_all(*closables: object) -> None:
    """Close multiple resources, logging and suppressing individual errors."""
    for obj in closables:
        try:
            close_fn = getattr(obj, "close", None)
            if close_fn is not None:
                await close_fn()
        except Exception:  # noqa: BLE001 — must not leak during shutdown
            logger.warning(
                "resource_close_failed",
                resource=type(obj).__name__,
                exc_info=True,
            )


def split_text_into_chunks(text: str, max_len: int = _MAX_CHUNK_LEN) -> list[str]:
    if len(text) <= max_len:
        return [text]

    raw_chunks: list[str] = []
    current_pos = 0
    total_len = len(text)

    while current_pos < total_len:
        if total_len - current_pos <= max_len:
            raw_chunks.append(text[current_pos:])
            break

        chunk_end = current_pos + max_len
        last_newline = text.rfind("\n", current_pos, chunk_end)
        if last_newline > current_pos:
            chunk_end = last_newline + 1

        raw_chunks.append(text[current_pos:chunk_end])
        current_pos = chunk_end

    total = len(raw_chunks)
    return [
        chunk + f"\n\n[续 {i + 1}/{total}]" if i < total - 1 else chunk
        for i, chunk in enumerate(raw_chunks)
    ]


def extract_stream_text(item: AgentStreamItem) -> str | None:
    if isinstance(item, StepDeltaEvent):
        return item.delta.content
    if isinstance(item, RunCompletedEvent):
        if not item.response:
            return None
        if item.depth == 0:
            return item.response
        return (
            f"<notice>agent_id={item.agent_id}, status=completed</notice>\n"
            f"{item.response}"
        )
    if isinstance(item, RunFailedEvent):
        if item.depth == 0:
            return item.error
        return f"<notice>agent_id={item.agent_id}, status=failed</notice>\n{item.error}"
    return None
