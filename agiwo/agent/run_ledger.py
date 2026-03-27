import time
from dataclasses import dataclass, field
from typing import Any

from agiwo.agent.compact_types import CompactMetadata
from agiwo.agent.types import TerminationReason


@dataclass
class RunLedger:
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_schemas: list[dict[str, Any]] | None = None
    start_time: float = field(default_factory=time.time)
    termination_reason: TerminationReason | None = None
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    token_cost: float = 0.0
    steps_count: int = 0
    tool_calls_count: int = 0
    assistant_steps_count: int = 0
    response_content: str | None = None
    last_compact_metadata: CompactMetadata | None = None
