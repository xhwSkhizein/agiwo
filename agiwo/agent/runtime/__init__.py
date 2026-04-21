"""Runtime context and helpers for agent execution."""

from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.hook_dispatcher import HookDispatcher
from agiwo.agent.runtime.run_engine import RunEngine
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.runtime.state_ops import (
    append_message,
    record_compaction_metadata,
    replace_messages,
    set_termination_reason,
    set_tool_schemas,
    track_step_state,
)
from agiwo.agent.runtime.state_writer import (
    build_compaction_applied_entry,
    build_context_assembled_entry,
    build_hook_failed_entry,
    build_llm_call_completed_entry,
    build_llm_call_started_entry,
    build_messages_rebuilt_entry,
    build_retrospect_applied_entry,
    build_run_failed_entry,
    build_run_finished_entry,
    build_run_started_entry,
    build_step_log_entry,
    build_termination_decided_entry,
)
from agiwo.agent.runtime.step_committer import commit_step

__all__ = [
    "HookDispatcher",
    "RunEngine",
    "RunContext",
    "SessionRuntime",
    "append_message",
    "build_compaction_applied_entry",
    "build_context_assembled_entry",
    "build_hook_failed_entry",
    "build_llm_call_completed_entry",
    "build_llm_call_started_entry",
    "build_messages_rebuilt_entry",
    "build_retrospect_applied_entry",
    "build_run_failed_entry",
    "build_run_finished_entry",
    "build_run_started_entry",
    "build_step_log_entry",
    "build_termination_decided_entry",
    "commit_step",
    "record_compaction_metadata",
    "replace_messages",
    "set_termination_reason",
    "set_tool_schemas",
    "track_step_state",
]
