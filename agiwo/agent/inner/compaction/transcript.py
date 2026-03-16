import json
from datetime import datetime
from pathlib import Path

import aiofiles

from agiwo.config.settings import settings


async def save_transcript(
    messages: list[dict[str, object]],
    agent_id: str,
    session_id: str,
    start_seq: int,
    end_seq: int,
    root_path: str | None = None,
) -> str:
    """Persist compacted source messages to a transcript file."""
    root = root_path or settings.root_path
    transcript_dir = Path(root) / "transcripts" / agent_id / session_id
    transcript_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{start_seq}_{end_seq}_{date_str}.jsonl"
    filepath = transcript_dir / filename

    async with aiofiles.open(filepath, "w", encoding="utf-8") as handle:
        for message in messages:
            await handle.write(
                json.dumps(message, ensure_ascii=False, default=str) + "\n"
            )

    return str(filepath)


__all__ = ["save_transcript"]
