"""Compact-operation domain types."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class CompactMetadata:
    """Metadata for a single compact operation."""

    session_id: str
    agent_id: str
    start_seq: int
    end_seq: int
    before_token_estimate: int
    after_token_estimate: int
    message_count: int
    transcript_path: str
    analysis: dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    compact_model: str = ""
    compact_tokens: int = 0

    def get_summary(self) -> str:
        return self.analysis.get("summary", "")


__all__ = ["CompactMetadata"]
