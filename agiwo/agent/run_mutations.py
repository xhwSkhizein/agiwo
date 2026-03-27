from collections.abc import Sequence
from typing import Any

from agiwo.agent.compact_types import CompactMetadata
from agiwo.agent.run_state import RunContext
from agiwo.agent.types import TerminationReason


def replace_messages(state: RunContext, messages: Sequence[dict[str, Any]]) -> None:
    state.ledger.messages = list(messages)


def set_tool_schemas(
    state: RunContext,
    tool_schemas: Sequence[dict[str, Any]] | None,
) -> None:
    state.ledger.tool_schemas = list(tool_schemas) if tool_schemas is not None else None


def record_compaction_metadata(
    state: RunContext,
    metadata: CompactMetadata | None,
) -> None:
    state.ledger.last_compact_metadata = metadata


def set_termination_reason(
    state: RunContext,
    reason: TerminationReason,
) -> None:
    state.ledger.termination_reason = reason
