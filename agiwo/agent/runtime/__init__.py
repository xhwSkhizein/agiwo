"""Runtime context and helpers for agent execution."""

from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.runtime.state_ops import (
    append_message,
    record_compaction_metadata,
    replace_messages,
    set_termination_reason,
    set_tool_schemas,
    track_step_state,
)
from agiwo.agent.runtime.step_committer import commit_step

__all__ = [
    "RunContext",
    "SessionRuntime",
    "append_message",
    "commit_step",
    "record_compaction_metadata",
    "replace_messages",
    "set_termination_reason",
    "set_tool_schemas",
    "track_step_state",
]
