"""Shared run-log serialization and read-view builders."""

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from agiwo.agent.models.input import UserMessage
from agiwo.agent.models.log import (
    AssistantStepCommitted,
    CommittedStep,
    CompactionApplied,
    ContextAssembled,
    HookFailed,
    LLMCallCompleted,
    LLMCallStarted,
    MessagesRebuilt,
    RetrospectApplied,
    RunFailed,
    RunFinished,
    RunLogEntry,
    RunLogEntryKind,
    RunStarted,
    TerminationDecided,
    ToolStepCommitted,
    UserStepCommitted,
)
from agiwo.agent.models.run import RunMetrics, RunView
from agiwo.agent.models.step import StepView

_RUN_LOG_TYPES: dict[RunLogEntryKind, type[RunLogEntry]] = {
    RunLogEntryKind.RUN_STARTED: RunStarted,
    RunLogEntryKind.RUN_FINISHED: RunFinished,
    RunLogEntryKind.RUN_FAILED: RunFailed,
    RunLogEntryKind.CONTEXT_ASSEMBLED: ContextAssembled,
    RunLogEntryKind.MESSAGES_REBUILT: MessagesRebuilt,
    RunLogEntryKind.LLM_CALL_STARTED: LLMCallStarted,
    RunLogEntryKind.LLM_CALL_COMPLETED: LLMCallCompleted,
    RunLogEntryKind.USER_STEP_COMMITTED: UserStepCommitted,
    RunLogEntryKind.ASSISTANT_STEP_COMMITTED: AssistantStepCommitted,
    RunLogEntryKind.TOOL_STEP_COMMITTED: ToolStepCommitted,
    RunLogEntryKind.COMPACTION_APPLIED: CompactionApplied,
    RunLogEntryKind.RETROSPECT_APPLIED: RetrospectApplied,
    RunLogEntryKind.TERMINATION_DECIDED: TerminationDecided,
    RunLogEntryKind.HOOK_FAILED: HookFailed,
}


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _drop_none(item) for key, item in value.items() if item is not None
        }
    if isinstance(value, list):
        return [_drop_none(item) if item is not None else None for item in value]
    return value


def _normalize_storage_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {
            key: _normalize_storage_value(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, (list, tuple)):
        return [
            _normalize_storage_value(item) if item is not None else None
            for item in value
        ]
    return value


def _as_storage_dict(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("Expected storage payload dict")
    return value


def serialize_run_log_entry_for_storage(entry: RunLogEntry) -> dict[str, Any]:
    data = _as_storage_dict(_normalize_storage_value(asdict(entry)))
    data["kind"] = entry.kind.value
    user_input = getattr(entry, "user_input", None)
    if user_input is not None:
        data["user_input"] = UserMessage.to_storage_value(user_input)
    return _as_storage_dict(_drop_none(data))


def deserialize_run_log_entry_from_storage(data: dict[str, Any]) -> RunLogEntry:
    normalized = dict(data)
    kind = RunLogEntryKind(normalized["kind"])
    entry_type = _RUN_LOG_TYPES[kind]
    if "user_input" in normalized:
        normalized["user_input"] = UserMessage.from_storage_value(
            normalized.get("user_input")
        )
    normalized.pop("kind", None)
    return entry_type(**normalized)


def build_run_view_from_entries(entries: list[RunLogEntry]) -> RunView | None:
    if not entries:
        return None

    started = next(
        (entry for entry in entries if isinstance(entry, RunStarted)),
        None,
    )
    if started is None:
        return None

    finished = next(
        (entry for entry in reversed(entries) if isinstance(entry, RunFinished)),
        None,
    )
    failed = next(
        (entry for entry in reversed(entries) if isinstance(entry, RunFailed)),
        None,
    )
    last_assistant = next(
        (
            entry
            for entry in reversed(entries)
            if isinstance(entry, AssistantStepCommitted)
        ),
        None,
    )

    status = "running"
    updated_at = entries[-1].created_at
    response: str | None = None
    termination_reason = None
    metrics: RunMetrics | None = None

    if finished is not None:
        status = "completed"
        response = finished.response
        termination_reason = finished.termination_reason
        metrics = RunMetrics(**finished.metrics) if finished.metrics else None
        updated_at = finished.created_at
    elif failed is not None:
        status = "failed"
        updated_at = failed.created_at

    if (
        response is None
        and last_assistant is not None
        and isinstance(last_assistant.content, str)
    ):
        response = last_assistant.content

    return RunView(
        run_id=started.run_id,
        session_id=started.session_id,
        agent_id=started.agent_id,
        status=status,
        user_id=started.user_id,
        response=response,
        termination_reason=termination_reason,
        metrics=metrics,
        last_user_input=started.user_input,
        created_at=started.created_at,
        updated_at=updated_at,
        parent_run_id=started.parent_run_id,
    )


def build_run_views_from_entries(entries: list[RunLogEntry]) -> list[RunView]:
    grouped: dict[str, list[RunLogEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.run_id, []).append(entry)
    views = [
        view
        for view in (
            build_run_view_from_entries(run_entries) for run_entries in grouped.values()
        )
        if view is not None
    ]
    earliest = datetime.min.replace(tzinfo=timezone.utc)
    views.sort(key=lambda view: view.created_at or earliest, reverse=True)
    return views


def build_step_view_from_entry(entry: CommittedStep) -> StepView:
    return StepView(
        id=entry.step_id,
        sequence=entry.sequence,
        session_id=entry.session_id,
        run_id=entry.run_id,
        agent_id=entry.agent_id,
        role=entry.role,
        content=entry.content,
        content_for_user=entry.content_for_user,
        reasoning_content=entry.reasoning_content,
        user_input=entry.user_input,
        tool_calls=entry.tool_calls,
        tool_call_id=entry.tool_call_id,
        name=entry.name,
        is_error=getattr(entry, "is_error", False),
        condensed_content=entry.condensed_content,
        metrics=entry.metrics,
        created_at=entry.created_at,
        parent_run_id=entry.parent_run_id,
        depth=entry.depth,
    )


def build_step_views_from_entries(entries: list[RunLogEntry]) -> list[StepView]:
    return [
        build_step_view_from_entry(entry)
        for entry in entries
        if isinstance(
            entry,
            (UserStepCommitted, AssistantStepCommitted, ToolStepCommitted),
        )
    ]


__all__ = [
    "build_run_view_from_entries",
    "build_run_views_from_entries",
    "build_step_view_from_entry",
    "build_step_views_from_entries",
    "deserialize_run_log_entry_from_storage",
    "serialize_run_log_entry_for_storage",
]
