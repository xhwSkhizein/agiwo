"""Shared Run/Step/RunLog storage serialization helpers."""

from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import TypeAdapter

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
from agiwo.agent.models.run import Run, RunMetrics, RunView
from agiwo.agent.models.step import StepRecord, StepView

_RUN_FIELD_NAMES = {field.name for field in fields(Run)}
_STEP_FIELD_NAMES = {field.name for field in fields(StepRecord)}
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
_RUN_ADAPTER = TypeAdapter(Run)
_STEP_ADAPTER = TypeAdapter(StepRecord)


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
    if isinstance(value, list):
        return [
            _normalize_storage_value(item) if item is not None else None
            for item in value
        ]
    if isinstance(value, tuple):
        return [
            _normalize_storage_value(item) if item is not None else None
            for item in value
        ]
    return value


def _as_storage_dict(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("Expected storage payload dict")
    return value


def _filter_storage_fields(
    data: dict[str, Any],
    field_names: set[str],
) -> dict[str, Any]:
    return {key: value for key, value in data.items() if key in field_names}


def serialize_run_for_storage(run: Run) -> dict[str, Any]:
    data = _as_storage_dict(_normalize_storage_value(asdict(run)))
    data["status"] = run.status.value
    data["user_input"] = UserMessage.to_storage_value(run.user_input)
    if run.metrics is not None:
        data["metrics"] = _normalize_storage_value(run.metrics)
    return _as_storage_dict(_drop_none(data))


def serialize_step_for_storage(step: StepRecord) -> dict[str, Any]:
    data = _as_storage_dict(_normalize_storage_value(asdict(step)))
    data["role"] = step.role.value
    if step.user_input is not None:
        data["user_input"] = UserMessage.to_storage_value(step.user_input)
    if step.metrics is not None:
        data["metrics"] = _normalize_storage_value(step.metrics)
    if step.is_user_step() and step.user_input is not None:
        data.pop("content", None)
    return _as_storage_dict(_drop_none(data))


def deserialize_run_from_storage(data: dict[str, Any]) -> Run:
    run_data = _filter_storage_fields(data, _RUN_FIELD_NAMES)
    run_data["user_input"] = UserMessage.from_storage_value(run_data.get("user_input"))
    return _RUN_ADAPTER.validate_python(run_data)


def deserialize_step_from_storage(data: dict[str, Any]) -> StepRecord:
    step_data = _filter_storage_fields(data, _STEP_FIELD_NAMES)
    step_data["user_input"] = UserMessage.from_storage_value(
        step_data.get("user_input")
    )

    return _STEP_ADAPTER.validate_python(step_data)


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


def build_run_view_from_run(run: Run) -> RunView:
    status = run.status.value if hasattr(run.status, "value") else str(run.status)
    return RunView(
        run_id=run.id,
        session_id=run.session_id,
        agent_id=run.agent_id,
        status=status,
        user_id=run.user_id,
        response=run.response_content,
        termination_reason=None,
        metrics=run.metrics,
        last_user_input=run.user_input,
        created_at=run.created_at,
        updated_at=run.updated_at,
        parent_run_id=run.parent_run_id,
    )


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
        id=f"{entry.run_id}:{entry.sequence}",
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
        condensed_content=entry.condensed_content,
        metrics=entry.metrics,
        created_at=entry.created_at,
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
    "build_run_view_from_run",
    "build_run_view_from_entries",
    "build_run_views_from_entries",
    "build_step_view_from_entry",
    "build_step_views_from_entries",
    "deserialize_run_log_entry_from_storage",
    "deserialize_run_from_storage",
    "deserialize_step_from_storage",
    "serialize_run_log_entry_for_storage",
    "serialize_run_for_storage",
    "serialize_step_for_storage",
]
