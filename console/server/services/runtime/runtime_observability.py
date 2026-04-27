"""Read-model builders for session and trace runtime observability."""

import json
import re
from datetime import datetime, timezone
from typing import Any, Callable

from agiwo.agent import StepView
from agiwo.agent.models.log import (
    CompactionApplied,
    CompactionFailed,
    ContextRepairApplied,
    HookFailed,
    RunLogEntry,
    RunRolledBack,
    StepBackApplied,
    TerminationDecided,
)
from agiwo.observability.trace import Span, SpanKind, SpanStatus, Trace

from server.models.session import (
    ConversationEventRecord,
    MilestoneRecord,
    ReviewCycleRecord,
    ReviewCheckpointRecord,
    ReviewOutcomeRecord,
    RuntimeDecisionRecord,
    SessionMilestoneBoardRecord,
    TraceLlmCallRecord,
    TraceMainlineEventRecord,
    TraceTimelineEventRecord,
)

_SYSTEM_REVIEW_BLOCK_RE = re.compile(
    r"<system-review>\s*(?P<body>.*?)\s*</system-review>",
    re.IGNORECASE | re.DOTALL,
)
_TRIGGER_RE = re.compile(r"^Trigger:\s*(?P<value>.+)$", re.MULTILINE)
_STEPS_RE = re.compile(r"^Steps since last review:\s*(?P<value>\d+)$", re.MULTILINE)
_MILESTONE_RE = re.compile(
    r'^Active milestone:\s*"(?P<value>.+)"\s*$',
    re.MULTILINE,
)
_HOOK_ADVICE_RE = re.compile(r"^Hook advice:\s*(?P<value>.+)$", re.MULTILINE)
_ALIGNED_RE = re.compile(r"aligned\s*=\s*(?P<value>true|false)", re.IGNORECASE)

_TIMELINE_KIND_ORDER = {
    "run_started": 0,
    "llm_call": 1,
    "tool_call": 2,
    "review_checkpoint": 3,
    "review_result": 4,
    "milestone_update": 5,
    "runtime_decision": 6,
    "hook_failed": 7,
    "run_finished": 8,
    "run_failed": 9,
}

_MAINLINE_EVENT_KINDS = {
    "run_started",
    "review_checkpoint",
    "review_result",
    "milestone_update",
    "runtime_decision",
    "hook_failed",
    "run_finished",
    "run_failed",
}


def _enum_value(value: object) -> object:
    if hasattr(value, "value"):
        return getattr(value, "value")
    return value


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if isinstance(value, int):
        return value != 0
    return False


def _as_optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes"}:
            return True
        if normalized in {"0", "false", "no"}:
            return False
    return None


def _json_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, default=str)


def _runtime_agent_id(trace: Trace, span: Span) -> str:
    agent_id = span.attributes.get("agent_id")
    if isinstance(agent_id, str) and agent_id:
        return agent_id
    if span.name:
        return span.name
    return trace.agent_id or ""


def _span_sequence(span: Span) -> int:
    return _as_int(span.attributes.get("sequence")) or 0


def _span_sort_key(span: Span) -> tuple[datetime, int, int, str]:
    kind_order = {
        SpanKind.AGENT: 0,
        SpanKind.LLM_CALL: 1,
        SpanKind.TOOL_CALL: 2,
        SpanKind.RUNTIME: 3,
    }
    return (
        span.start_time or datetime.min.replace(tzinfo=timezone.utc),
        _span_sequence(span),
        kind_order.get(span.kind, 999),
        span.span_id,
    )


def _cycle_sort_key(cycle: ReviewCycleRecord) -> tuple[datetime, int, str]:
    return (
        cycle.started_at or datetime.min.replace(tzinfo=timezone.utc),
        _cycle_sequence(cycle) or 0,
        cycle.cycle_id,
    )


def _string_value(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _normalize_tool_details(span: Span) -> dict[str, Any]:
    return dict(span.tool_details or {})


def _review_tool_name(span: Span) -> str:
    tool_details = _normalize_tool_details(span)
    tool_name = tool_details.get("tool_name")
    if isinstance(tool_name, str) and tool_name:
        return tool_name
    return span.name


def _parse_review_result(
    tool_details: dict[str, Any],
    raw_output: str,
) -> tuple[bool | None, str | None]:
    input_args = tool_details.get("input_args")
    if isinstance(input_args, dict):
        aligned = _as_optional_bool(input_args.get("aligned"))
        experience = _string_value(input_args.get("experience"))
        if aligned is not None:
            return aligned, experience
    aligned_match = _ALIGNED_RE.search(raw_output)
    aligned = (
        aligned_match.group("value").strip().lower() == "true"
        if aligned_match is not None
        else None
    )
    experience = None
    if raw_output:
        _, _, tail = raw_output.partition(".")
        experience = tail.strip() or None
    return aligned, experience


def _new_review_cycle_from_notice(
    span: Span,
    notice: dict[str, Any],
    *,
    trace: Trace,
) -> ReviewCycleRecord:
    sequence = _span_sequence(span)
    return ReviewCycleRecord(
        cycle_id=f"{span.run_id or 'run'}:{sequence}:{span.span_id}",
        run_id=span.run_id or "",
        agent_id=_runtime_agent_id(trace, span),
        trigger_reason=str(notice.get("trigger_reason") or "unknown"),
        steps_since_last_review=_as_int(notice.get("steps_since_last_review")),
        active_milestone=_string_value(notice.get("active_milestone")),
        hook_advice=_string_value(notice.get("hook_advice")),
        started_at=span.start_time,
        raw_notice=_string_value(notice.get("raw_notice")),
    )


def _milestone_description_by_id(
    milestones: list[MilestoneRecord],
) -> dict[str, str]:
    return {milestone.id: milestone.description for milestone in milestones}


def _new_review_cycle_from_trigger_span(
    span: Span,
    *,
    trace: Trace,
    milestone_descriptions: dict[str, str],
) -> ReviewCycleRecord:
    sequence = _span_sequence(span)
    active_milestone_id = _string_value(span.attributes.get("active_milestone_id"))
    return ReviewCycleRecord(
        cycle_id=f"{span.run_id or 'run'}:{sequence}:{span.span_id}",
        run_id=span.run_id or "",
        agent_id=_runtime_agent_id(trace, span),
        trigger_reason=_json_text(span.attributes.get("trigger_reason")) or "unknown",
        steps_since_last_review=_as_int(
            span.attributes.get("review_count_since_checkpoint")
        ),
        active_milestone=milestone_descriptions.get(active_milestone_id or ""),
        active_milestone_id=active_milestone_id,
        started_at=span.start_time,
    )


def _find_latest_review_cycle(
    cycles: list[ReviewCycleRecord],
    *,
    run_id: str,
    predicate: Callable[[ReviewCycleRecord], bool],
) -> ReviewCycleRecord | None:
    for cycle in reversed(cycles):
        if cycle.run_id != run_id:
            continue
        if predicate(cycle):
            return cycle
    return None


def _fallback_review_cycle(trace: Trace, span: Span) -> ReviewCycleRecord:
    sequence = _span_sequence(span)
    cycle = ReviewCycleRecord(
        cycle_id=f"{span.run_id or 'run'}:{sequence}:{span.span_id}",
        run_id=span.run_id or "",
        agent_id=_runtime_agent_id(trace, span),
        trigger_reason="unknown",
        started_at=span.start_time,
    )
    return cycle


def _cycle_sequence(cycle: ReviewCycleRecord) -> int | None:
    parts = cycle.cycle_id.rsplit(":", 2)
    if len(parts) == 3:
        sequence = _as_int(parts[1])
        if sequence is not None:
            return sequence
    return _as_int(parts[-1]) if parts else None


def _summarize_text(text: str | None, *, limit: int = 120) -> str:
    if not text:
        return ""
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "..."


def _extract_declared_milestones(span: Span) -> list[dict[str, Any]]:
    tool_details = _normalize_tool_details(span)
    input_args = tool_details.get("input_args")
    output = tool_details.get("output")
    candidates: list[object] = []
    if isinstance(input_args, dict):
        candidates.append(input_args.get("milestones"))
    if isinstance(output, dict):
        candidates.append(output.get("milestones"))
    for candidate in candidates:
        if not isinstance(candidate, list):
            continue
        milestones: list[dict[str, Any]] = []
        for item in candidate:
            if not isinstance(item, dict):
                continue
            milestone_id = _string_value(item.get("id"))
            description = _string_value(item.get("description"))
            if milestone_id is None or description is None:
                continue
            status = _string_value(item.get("status")) or "pending"
            milestones.append(
                {
                    "id": milestone_id,
                    "description": description,
                    "status": status,
                }
            )
        if milestones:
            return milestones
    return []


def _extract_review_fact_milestones(span: Span) -> list[dict[str, Any]]:
    raw_milestones = span.attributes.get("milestones")
    if not isinstance(raw_milestones, list):
        return []
    milestones: list[dict[str, Any]] = []
    for raw in raw_milestones:
        if not isinstance(raw, dict):
            continue
        milestone_id = _string_value(raw.get("id"))
        description = _string_value(raw.get("description"))
        if milestone_id is None or description is None:
            continue
        milestones.append(
            {
                "id": milestone_id,
                "description": description,
                "status": _string_value(raw.get("status")) or "pending",
                "declared_at_seq": _as_int(raw.get("declared_at_seq")),
                "completed_at_seq": _as_int(raw.get("completed_at_seq")),
            }
        )
    return milestones


def _milestone_id_from_description(
    milestones: list[MilestoneRecord],
    description: str | None,
) -> str | None:
    if description is None:
        return None
    for milestone in milestones:
        if milestone.description == description:
            return milestone.id
    return None


def _attach_cycle_milestone_ids(
    cycles: list[ReviewCycleRecord],
    milestones: list[MilestoneRecord],
) -> None:
    for cycle in cycles:
        cycle.active_milestone_id = _milestone_id_from_description(
            milestones,
            cycle.active_milestone,
        )


def _summary_from_step(step: StepView) -> str:
    preferred = step.content_for_user or step.condensed_content
    if preferred:
        return _summarize_text(preferred)
    return _summarize_text(step.get_display_text())


def _step_event_details(step: StepView, **extra: Any) -> dict[str, Any]:
    stored_step = step.to_dict()
    details = {
        "step_id": step.id,
        "role": step.role.value,
        "content": step.content,
        "content_for_user": step.content_for_user,
        "condensed_content": step.condensed_content,
        "user_input": stored_step.get("user_input"),
        "tool_calls": step.tool_calls,
        "tool_call_id": step.tool_call_id,
        "tool_name": step.name,
        "is_error": True if step.is_error else None,
        **extra,
    }
    return {key: value for key, value in details.items() if value is not None}


def _review_cycle_summary(cycle: ReviewCycleRecord) -> str:
    if cycle.aligned is True:
        return "Review aligned with milestone"
    if cycle.aligned is False and cycle.step_back_applied:
        count = cycle.affected_count or 0
        return f"Review misaligned; {count} steps condensed"
    if cycle.aligned is False:
        return "Review flagged trajectory drift"
    return "Review checkpoint recorded"


def _build_runtime_decision_record(
    *,
    kind: str,
    sequence: int,
    run_id: str,
    agent_id: str,
    created_at,
    payload: dict[str, Any],
) -> RuntimeDecisionRecord:
    details = {key: _enum_value(value) for key, value in payload.items()}
    if kind == "compaction":
        summary = (
            f"seq {details['start_sequence']}-{details['end_sequence']} "
            f"{details['before_token_estimate']} -> {details['after_token_estimate']} tokens"
        )
    elif kind == "compaction_failed":
        summary = f"compaction failed (attempt {details['attempt']}/{details['max_attempts']})"
    elif kind == "step_back":
        summary = (
            f"{details['affected_count']} results condensed after checkpoint seq "
            f"{details['checkpoint_seq']}"
        )
    elif kind == "rollback":
        summary = f"seq {details['start_sequence']}-{details['end_sequence']} hidden"
    elif kind == "termination":
        summary = f"{details['reason']} via {details['source']}"
    elif kind == "hook_failed":
        summary = f"{details['phase']}: {details['handler_name']} failed"
    else:
        raise ValueError(f"Unsupported runtime decision kind: {kind}")
    return RuntimeDecisionRecord(
        kind=kind,
        sequence=sequence,
        run_id=run_id,
        agent_id=agent_id,
        created_at=created_at,
        summary=summary,
        details=details,
    )


def build_runtime_decision_record_from_entry(
    entry: RunLogEntry,
) -> RuntimeDecisionRecord:
    if isinstance(entry, CompactionApplied):
        return _build_runtime_decision_record(
            kind="compaction",
            sequence=entry.sequence,
            run_id=entry.run_id,
            agent_id=entry.agent_id,
            created_at=entry.created_at,
            payload={
                "start_sequence": entry.start_sequence,
                "end_sequence": entry.end_sequence,
                "before_token_estimate": entry.before_token_estimate,
                "after_token_estimate": entry.after_token_estimate,
                "message_count": entry.message_count,
                "summary": entry.summary,
                "transcript_path": entry.transcript_path,
            },
        )
    if isinstance(entry, CompactionFailed):
        return _build_runtime_decision_record(
            kind="compaction_failed",
            sequence=entry.sequence,
            run_id=entry.run_id,
            agent_id=entry.agent_id,
            created_at=entry.created_at,
            payload={
                "error": entry.error,
                "attempt": entry.attempt,
                "max_attempts": entry.max_attempts,
                "terminal": entry.terminal,
            },
        )
    if isinstance(entry, StepBackApplied):
        return _build_runtime_decision_record(
            kind="step_back",
            sequence=entry.sequence,
            run_id=entry.run_id,
            agent_id=entry.agent_id,
            created_at=entry.created_at,
            payload={
                "affected_count": entry.affected_count,
                "checkpoint_seq": entry.checkpoint_seq,
                "experience": entry.experience,
            },
        )
    if isinstance(entry, ContextRepairApplied):
        return _build_runtime_decision_record(
            kind="step_back",
            sequence=entry.sequence,
            run_id=entry.run_id,
            agent_id=entry.agent_id,
            created_at=entry.created_at,
            payload={
                "affected_count": entry.affected_count,
                "checkpoint_seq": entry.start_seq - 1,
                "start_sequence": entry.start_seq,
                "end_sequence": entry.end_seq,
                "experience": entry.experience,
            },
        )
    if isinstance(entry, RunRolledBack):
        return _build_runtime_decision_record(
            kind="rollback",
            sequence=entry.sequence,
            run_id=entry.run_id,
            agent_id=entry.agent_id,
            created_at=entry.created_at,
            payload={
                "start_sequence": entry.start_sequence,
                "end_sequence": entry.end_sequence,
                "reason": entry.reason,
            },
        )
    if isinstance(entry, TerminationDecided):
        return _build_runtime_decision_record(
            kind="termination",
            sequence=entry.sequence,
            run_id=entry.run_id,
            agent_id=entry.agent_id,
            created_at=entry.created_at,
            payload={
                "reason": entry.termination_reason.value,
                "phase": entry.phase,
                "source": entry.source,
            },
        )
    if isinstance(entry, HookFailed):
        return _build_runtime_decision_record(
            kind="hook_failed",
            sequence=entry.sequence,
            run_id=entry.run_id,
            agent_id=entry.agent_id,
            created_at=entry.created_at,
            payload={
                "phase": entry.phase,
                "handler_name": entry.handler_name,
                "critical": entry.critical,
                "error": entry.error,
            },
        )
    raise TypeError(f"Unsupported runtime decision entry: {type(entry).__name__}")


def _build_compaction_decision_from_span(
    trace: Trace,
    span: Span,
    *,
    sequence: int,
) -> RuntimeDecisionRecord:
    return _build_runtime_decision_record(
        kind="compaction",
        sequence=sequence,
        run_id=span.run_id or "",
        agent_id=_runtime_agent_id(trace, span),
        created_at=span.start_time,
        payload={
            "start_sequence": _as_int(span.attributes.get("start_sequence")) or 0,
            "end_sequence": _as_int(span.attributes.get("end_sequence")) or 0,
            "before_token_estimate": _as_int(
                span.attributes.get("before_token_estimate")
            )
            or 0,
            "after_token_estimate": _as_int(span.attributes.get("after_token_estimate"))
            or 0,
            "message_count": _as_int(span.attributes.get("message_count")) or 0,
            "summary": span.attributes.get("summary"),
            "transcript_path": span.attributes.get("transcript_path"),
        },
    )


def _build_compaction_failed_decision_from_span(
    trace: Trace,
    span: Span,
    *,
    sequence: int,
) -> RuntimeDecisionRecord:
    return _build_runtime_decision_record(
        kind="compaction_failed",
        sequence=sequence,
        run_id=span.run_id or "",
        agent_id=_runtime_agent_id(trace, span),
        created_at=span.start_time,
        payload={
            "error": _json_text(span.attributes.get("error")),
            "attempt": _as_int(span.attributes.get("attempt")) or 0,
            "max_attempts": _as_int(span.attributes.get("max_attempts")) or 0,
            "terminal": _as_bool(span.attributes.get("terminal")),
        },
    )


def _build_step_back_decision_from_span(
    trace: Trace,
    span: Span,
    *,
    sequence: int,
) -> RuntimeDecisionRecord:
    return _build_runtime_decision_record(
        kind="step_back",
        sequence=sequence,
        run_id=span.run_id or "",
        agent_id=_runtime_agent_id(trace, span),
        created_at=span.start_time,
        payload={
            "affected_count": _as_int(span.attributes.get("affected_count")) or 0,
            "checkpoint_seq": _as_int(span.attributes.get("checkpoint_seq")) or 0,
            "experience": _json_text(span.attributes.get("experience")),
        },
    )


def _build_rollback_decision_from_span(
    trace: Trace,
    span: Span,
    *,
    sequence: int,
) -> RuntimeDecisionRecord:
    return _build_runtime_decision_record(
        kind="rollback",
        sequence=sequence,
        run_id=span.run_id or "",
        agent_id=_runtime_agent_id(trace, span),
        created_at=span.start_time,
        payload={
            "start_sequence": _as_int(span.attributes.get("start_sequence")) or 0,
            "end_sequence": _as_int(span.attributes.get("end_sequence")) or 0,
            "reason": _json_text(span.attributes.get("reason")),
        },
    )


def _build_termination_decision_from_span(
    trace: Trace,
    span: Span,
    *,
    sequence: int,
) -> RuntimeDecisionRecord:
    return _build_runtime_decision_record(
        kind="termination",
        sequence=sequence,
        run_id=span.run_id or "",
        agent_id=_runtime_agent_id(trace, span),
        created_at=span.start_time,
        payload={
            "reason": _json_text(span.attributes.get("termination_reason")),
            "phase": _json_text(span.attributes.get("phase")),
            "source": _json_text(span.attributes.get("source")),
        },
    )


def _build_hook_failed_decision_from_span(
    trace: Trace,
    span: Span,
    *,
    sequence: int,
) -> RuntimeDecisionRecord:
    return _build_runtime_decision_record(
        kind="hook_failed",
        sequence=sequence,
        run_id=span.run_id or "",
        agent_id=_runtime_agent_id(trace, span),
        created_at=span.start_time,
        payload={
            "phase": _json_text(span.attributes.get("phase")),
            "handler_name": _json_text(span.attributes.get("handler_name")),
            "critical": _as_bool(span.attributes.get("critical")),
            "error": _json_text(span.attributes.get("error")),
        },
    )


_RUNTIME_DECISION_SPAN_BUILDERS: dict[
    str,
    Callable[[Trace, Span], RuntimeDecisionRecord],
] = {
    "compaction": lambda trace, span: _build_compaction_decision_from_span(
        trace,
        span,
        sequence=_span_sequence(span),
    ),
    "compaction_failed": lambda trace, span: (
        _build_compaction_failed_decision_from_span(
            trace,
            span,
            sequence=_span_sequence(span),
        )
    ),
    "step_back": lambda trace, span: _build_step_back_decision_from_span(
        trace,
        span,
        sequence=_span_sequence(span),
    ),
    "rollback": lambda trace, span: _build_rollback_decision_from_span(
        trace,
        span,
        sequence=_span_sequence(span),
    ),
    "termination": lambda trace, span: _build_termination_decision_from_span(
        trace,
        span,
        sequence=_span_sequence(span),
    ),
    "hook_failed": lambda trace, span: _build_hook_failed_decision_from_span(
        trace,
        span,
        sequence=_span_sequence(span),
    ),
}


def build_runtime_decision_record_from_span(
    trace: Trace,
    span: Span,
) -> RuntimeDecisionRecord | None:
    if span.kind != SpanKind.RUNTIME:
        return None
    builder = _RUNTIME_DECISION_SPAN_BUILDERS.get(span.name)
    return builder(trace, span) if builder is not None else None


def parse_system_review_notice(text: str) -> dict[str, Any] | None:
    match = _SYSTEM_REVIEW_BLOCK_RE.search(text)
    if match is None:
        return None
    body = match.group("body").strip()
    trigger_match = _TRIGGER_RE.search(body)
    step_match = _STEPS_RE.search(body)
    milestone_match = _MILESTONE_RE.search(body)
    advice_match = _HOOK_ADVICE_RE.search(body)
    return {
        "raw_notice": body,
        "trigger_reason": (
            trigger_match.group("value").strip()
            if trigger_match is not None
            else "unknown"
        ),
        "steps_since_last_review": (
            int(step_match.group("value")) if step_match is not None else None
        ),
        "active_milestone": (
            milestone_match.group("value").strip()
            if milestone_match is not None
            else None
        ),
        "hook_advice": (
            advice_match.group("value").strip() if advice_match is not None else None
        ),
    }


def build_trace_runtime_decisions(trace: Trace) -> list[RuntimeDecisionRecord]:
    decisions: list[RuntimeDecisionRecord] = []
    for span in trace.spans:
        record = build_runtime_decision_record_from_span(trace, span)
        if record is None or record.kind == "hook_failed":
            continue
        decisions.append(record)
    decisions.sort(key=lambda item: (item.created_at, item.sequence))
    return decisions


def _timeline_sort_key(
    event: TraceTimelineEventRecord,
) -> tuple[datetime, int, int, str]:
    return (
        event.timestamp or datetime.min.replace(tzinfo=timezone.utc),
        event.sequence if event.sequence is not None else 1_000_000_000,
        _TIMELINE_KIND_ORDER.get(event.kind, 999),
        event.title,
    )


def _timeline_events_from_agent_span(
    trace: Trace,
    span: Span,
    *,
    status: str,
) -> list[TraceTimelineEventRecord]:
    agent_id = _runtime_agent_id(trace, span)
    events = [
        TraceTimelineEventRecord(
            kind="run_started",
            timestamp=span.start_time,
            sequence=_as_int(span.attributes.get("start_sequence")),
            run_id=span.run_id,
            agent_id=agent_id,
            span_id=span.span_id,
            title="Run Started",
            summary=f"{span.name} started",
            status="ok",
            details={
                "agent_id": agent_id,
                "nested": span.attributes.get("nested"),
                "parent_run_id": span.attributes.get("parent_run_id"),
            },
        )
    ]
    if span.end_time is None:
        return events
    completed_kind = "run_failed" if span.status == SpanStatus.ERROR else "run_finished"
    events.append(
        TraceTimelineEventRecord(
            kind=completed_kind,
            timestamp=span.end_time,
            sequence=_as_int(span.attributes.get("end_sequence")),
            run_id=span.run_id,
            agent_id=agent_id,
            span_id=span.span_id,
            title="Run Failed" if completed_kind == "run_failed" else "Run Finished",
            summary=(
                span.error_message
                if completed_kind == "run_failed"
                else (span.output_preview or "run completed")
            ),
            status=status,
            details={
                "agent_id": agent_id,
                "duration_ms": span.duration_ms,
                "termination_reason": span.attributes.get("termination_reason"),
                "error": span.error_message,
            },
        )
    )
    return events


def _timeline_event_from_llm_span(
    trace: Trace,
    span: Span,
    *,
    sequence: int | None,
    status: str,
) -> TraceTimelineEventRecord:
    details = dict(span.llm_details or {})
    return TraceTimelineEventRecord(
        kind="llm_call",
        timestamp=span.start_time,
        sequence=sequence,
        run_id=span.run_id,
        agent_id=_runtime_agent_id(trace, span),
        span_id=span.span_id,
        title="LLM Call",
        summary=f"{span.name} · {details.get('finish_reason') or 'completed'}",
        status=status,
        details=details,
    )


def _review_checkpoint_event_from_span(
    trace: Trace,
    span: Span,
    *,
    sequence: int | None,
    review_notice: dict[str, Any],
) -> TraceTimelineEventRecord:
    trigger_reason = review_notice.get("trigger_reason") or "unknown"
    step_count = review_notice.get("steps_since_last_review")
    step_summary = f"{step_count} steps" if step_count is not None else "unknown steps"
    return TraceTimelineEventRecord(
        kind="review_checkpoint",
        timestamp=span.start_time,
        sequence=sequence,
        run_id=span.run_id,
        agent_id=_runtime_agent_id(trace, span),
        span_id=span.span_id,
        step_id=span.step_id,
        title="Review Checkpoint",
        summary=f"triggered by {trigger_reason} after {step_summary}",
        status="ok",
        details=review_notice,
    )


def _timeline_events_from_tool_span(
    trace: Trace,
    span: Span,
    *,
    sequence: int | None,
    status: str,
) -> list[TraceTimelineEventRecord]:
    tool_details = dict(span.tool_details or {})
    tool_name = str(tool_details.get("tool_name") or span.name)
    events: list[TraceTimelineEventRecord] = []
    events.append(
        TraceTimelineEventRecord(
            kind="tool_call",
            timestamp=span.start_time,
            sequence=sequence,
            run_id=span.run_id,
            agent_id=_runtime_agent_id(trace, span),
            span_id=span.span_id,
            step_id=span.step_id,
            title=f"Tool Call: {tool_name}",
            summary=str(tool_details.get("status") or "completed"),
            status=status,
            details=tool_details,
        )
    )
    return events


def _timeline_event_from_runtime_span(
    trace: Trace,
    span: Span,
    *,
    status: str,
) -> TraceTimelineEventRecord | None:
    sequence = _span_sequence(span)
    if span.name == "review_trigger":
        trigger_reason = span.attributes.get("trigger_reason") or "unknown"
        step_count = _as_int(span.attributes.get("review_count_since_checkpoint"))
        step_summary = (
            f"{step_count} steps" if step_count is not None else "unknown steps"
        )
        return TraceTimelineEventRecord(
            kind="review_checkpoint",
            timestamp=span.start_time,
            sequence=sequence,
            run_id=span.run_id,
            agent_id=_runtime_agent_id(trace, span),
            span_id=span.span_id,
            step_id=_string_value(span.attributes.get("notice_step_id")),
            title="Review Checkpoint",
            summary=f"triggered by {trigger_reason} after {step_summary}",
            status=status,
            details=dict(span.attributes),
        )
    if span.name == "review_outcome":
        aligned = _as_optional_bool(span.attributes.get("aligned"))
        return TraceTimelineEventRecord(
            kind="review_result",
            timestamp=span.start_time,
            sequence=sequence,
            run_id=span.run_id,
            agent_id=_runtime_agent_id(trace, span),
            span_id=span.span_id,
            step_id=_string_value(span.attributes.get("review_step_id")),
            title="Review Result",
            summary=(
                "trajectory aligned"
                if aligned is True
                else "trajectory misaligned"
                if aligned is False
                else "trajectory reviewed"
            ),
            status=status,
            details=dict(span.attributes),
        )
    if span.name == "review_milestones":
        milestones = _extract_review_fact_milestones(span)
        return TraceTimelineEventRecord(
            kind="milestone_update",
            timestamp=span.start_time,
            sequence=sequence,
            run_id=span.run_id,
            agent_id=_runtime_agent_id(trace, span),
            span_id=span.span_id,
            step_id=_string_value(span.attributes.get("source_step_id")),
            title="Milestone Update",
            summary=f"{len(milestones)} milestones declared/updated",
            status=status,
            details={**dict(span.attributes), "milestones": milestones},
        )
    record = build_runtime_decision_record_from_span(trace, span)
    if record is None:
        return None
    timeline_kind = (
        "hook_failed" if record.kind == "hook_failed" else "runtime_decision"
    )
    details = dict(record.details)
    if timeline_kind == "runtime_decision":
        details = {"kind": record.kind, **details}
    return TraceTimelineEventRecord(
        kind=timeline_kind,
        timestamp=record.created_at,
        sequence=record.sequence,
        run_id=record.run_id,
        agent_id=record.agent_id,
        span_id=span.span_id,
        step_id=span.step_id,
        title=span.name.replace("_", " ").title(),
        summary=record.summary,
        status=status,
        details=details,
    )


def build_trace_timeline_events(trace: Trace) -> list[TraceTimelineEventRecord]:
    events: list[TraceTimelineEventRecord] = []

    for span in trace.spans:
        sequence = _as_int(span.attributes.get("sequence"))
        status = str(_enum_value(span.status))

        if span.kind == SpanKind.AGENT:
            events.extend(_timeline_events_from_agent_span(trace, span, status=status))
        elif span.kind == SpanKind.LLM_CALL:
            events.append(
                _timeline_event_from_llm_span(
                    trace,
                    span,
                    sequence=sequence,
                    status=status,
                )
            )
        elif span.kind == SpanKind.TOOL_CALL:
            events.extend(
                _timeline_events_from_tool_span(
                    trace,
                    span,
                    sequence=sequence,
                    status=status,
                )
            )
        elif span.kind == SpanKind.RUNTIME:
            event = _timeline_event_from_runtime_span(trace, span, status=status)
            if event is not None:
                events.append(event)

    events.sort(key=_timeline_sort_key)
    return events


def build_trace_mainline_events(trace: Trace) -> list[TraceMainlineEventRecord]:
    return [
        TraceMainlineEventRecord(
            id=(
                f"{event.kind}:{event.run_id or 'run'}:"
                f"{event.sequence if event.sequence is not None else 'na'}:"
                f"{event.span_id or event.title}"
            ),
            kind=event.kind,
            title=event.title,
            summary=event.summary,
            status=event.status,
            sequence=event.sequence,
            timestamp=event.timestamp,
            run_id=event.run_id,
            agent_id=event.agent_id,
            details=dict(event.details),
        )
        for event in build_trace_timeline_events(trace)
        if event.kind in _MAINLINE_EVENT_KINDS
    ]


def _apply_review_result_to_cycles(
    cycles: list[ReviewCycleRecord],
    *,
    trace: Trace,
    span: Span,
    tool_details: dict[str, Any],
    raw_output: str,
) -> None:
    cycle = _find_latest_review_cycle(
        cycles,
        run_id=span.run_id or "",
        predicate=lambda item: item.aligned is None,
    )
    if cycle is None:
        cycle = _fallback_review_cycle(trace, span)
        cycles.append(cycle)

    aligned, experience = _parse_review_result(tool_details, raw_output)
    cycle.aligned = aligned
    if experience:
        cycle.experience = experience
    cycle.resolved_at = span.start_time


def _apply_runtime_review_outcome_to_cycles(
    cycles: list[ReviewCycleRecord],
    *,
    trace: Trace,
    span: Span,
    decision: RuntimeDecisionRecord,
) -> None:
    if decision.kind == "step_back":
        cycle = _find_latest_review_cycle(
            cycles,
            run_id=decision.run_id,
            predicate=lambda item: item.aligned is False,
        )
        if cycle is None:
            cycle = _fallback_review_cycle(trace, span)
            cycles.append(cycle)
        cycle.step_back_applied = True
        cycle.affected_count = _as_int(decision.details.get("affected_count"))
        experience = _string_value(decision.details.get("experience"))
        if experience:
            cycle.experience = experience
        cycle.resolved_at = decision.created_at
        return

    if decision.kind != "rollback":
        return

    cycle = _find_latest_review_cycle(
        cycles,
        run_id=decision.run_id,
        predicate=lambda item: item.step_back_applied and item.rollback_range is None,
    )
    if cycle is None:
        cycle = _fallback_review_cycle(trace, span)
        cycles.append(cycle)
    start_sequence = _as_int(decision.details.get("start_sequence"))
    end_sequence = _as_int(decision.details.get("end_sequence"))
    if start_sequence is not None and end_sequence is not None:
        cycle.rollback_range = (start_sequence, end_sequence)
    cycle.resolved_at = decision.created_at


def _apply_review_outcome_fact_to_cycles(
    cycles: list[ReviewCycleRecord],
    *,
    trace: Trace,
    span: Span,
    milestone_descriptions: dict[str, str],
) -> None:
    active_milestone_id = _string_value(span.attributes.get("active_milestone_id"))
    cycle = _find_latest_review_cycle(
        cycles,
        run_id=span.run_id or "",
        predicate=lambda item: item.aligned is None,
    )
    if cycle is None:
        cycle = _fallback_review_cycle(trace, span)
        cycle.active_milestone_id = active_milestone_id
        cycle.active_milestone = milestone_descriptions.get(active_milestone_id or "")
        cycles.append(cycle)

    aligned = _as_optional_bool(span.attributes.get("aligned"))
    cycle.aligned = aligned
    cycle.active_milestone_id = active_milestone_id or cycle.active_milestone_id
    if cycle.active_milestone_id is not None:
        cycle.active_milestone = milestone_descriptions.get(cycle.active_milestone_id)
    experience = _string_value(span.attributes.get("experience"))
    if experience:
        cycle.experience = experience
    mode = _string_value(span.attributes.get("mode"))
    condensed_step_ids = span.attributes.get("condensed_step_ids")
    if mode == "step_back":
        cycle.step_back_applied = True
        if isinstance(condensed_step_ids, list):
            cycle.affected_count = len(condensed_step_ids)
    cycle.resolved_at = span.start_time


def _update_review_cycles_from_review_fact_span(
    cycles: list[ReviewCycleRecord],
    *,
    trace: Trace,
    span: Span,
    milestone_descriptions: dict[str, str],
) -> bool:
    if span.name == "review_trigger":
        cycles.append(
            _new_review_cycle_from_trigger_span(
                span,
                trace=trace,
                milestone_descriptions=milestone_descriptions,
            )
        )
        return True
    if span.name == "review_outcome":
        _apply_review_outcome_fact_to_cycles(
            cycles,
            trace=trace,
            span=span,
            milestone_descriptions=milestone_descriptions,
        )
        return True
    return span.name in {"review_milestones", "review_checkpoint"}


def _update_review_cycles_from_tool_span(
    cycles: list[ReviewCycleRecord],
    *,
    trace: Trace,
    span: Span,
) -> None:
    tool_details = _normalize_tool_details(span)
    raw_output = _json_text(tool_details.get("output"))
    review_notice = parse_system_review_notice(raw_output)
    if review_notice is not None:
        cycles.append(_new_review_cycle_from_notice(span, review_notice, trace=trace))
        return
    if _review_tool_name(span) != "review_trajectory":
        return
    _apply_review_result_to_cycles(
        cycles,
        trace=trace,
        span=span,
        tool_details=tool_details,
        raw_output=raw_output,
    )


def _update_review_cycles_from_runtime_span(
    cycles: list[ReviewCycleRecord],
    *,
    trace: Trace,
    span: Span,
) -> None:
    decision = build_runtime_decision_record_from_span(trace, span)
    if decision is None:
        return
    _apply_runtime_review_outcome_to_cycles(
        cycles,
        trace=trace,
        span=span,
        decision=decision,
    )


def build_trace_review_cycles(trace: Trace) -> list[ReviewCycleRecord]:
    cycles: list[ReviewCycleRecord] = []
    milestones = _collect_milestones_from_trace(trace)
    milestone_descriptions = _milestone_description_by_id(milestones)

    for span in sorted(trace.spans, key=_span_sort_key):
        if span.kind == SpanKind.RUNTIME:
            if _update_review_cycles_from_review_fact_span(
                cycles,
                trace=trace,
                span=span,
                milestone_descriptions=milestone_descriptions,
            ):
                continue
            _update_review_cycles_from_runtime_span(cycles, trace=trace, span=span)

    _attach_cycle_milestone_ids(cycles, milestones)
    cycles.sort(key=_cycle_sort_key)
    return cycles


def build_trace_llm_call_records(trace: Trace) -> list[TraceLlmCallRecord]:
    records: list[TraceLlmCallRecord] = []
    for span in sorted(trace.spans, key=_span_sort_key):
        if span.kind != SpanKind.LLM_CALL:
            continue
        details = dict(span.llm_details or {})
        detail_metrics = details.get("metrics")
        metrics = detail_metrics if isinstance(detail_metrics, dict) else {}
        span_metrics = dict(span.metrics or {})
        messages = details.get("messages")
        tools = details.get("tools")
        response_tool_calls = details.get("response_tool_calls")
        response_content = details.get("response_content")
        records.append(
            TraceLlmCallRecord(
                span_id=span.span_id,
                run_id=span.run_id or "",
                agent_id=_runtime_agent_id(trace, span),
                model=(
                    _string_value(span.attributes.get("model_name"))
                    or _string_value(span.attributes.get("model_id"))
                    or span.name
                ),
                provider=(
                    _string_value(span.attributes.get("provider"))
                    or _string_value(metrics.get("provider"))
                    or _string_value(span_metrics.get("provider"))
                ),
                finish_reason=_string_value(details.get("finish_reason")),
                duration_ms=(
                    span.duration_ms
                    or (
                        metrics.get("duration_ms")
                        if isinstance(metrics.get("duration_ms"), (int, float))
                        else None
                    )
                    or (
                        span_metrics.get("duration_ms")
                        if isinstance(span_metrics.get("duration_ms"), (int, float))
                        else None
                    )
                ),
                first_token_latency_ms=(
                    (
                        metrics.get("first_token_ms")
                        if isinstance(metrics.get("first_token_ms"), (int, float))
                        else None
                    )
                    or (
                        span_metrics.get("first_token_ms")
                        if isinstance(span_metrics.get("first_token_ms"), (int, float))
                        else None
                    )
                ),
                input_tokens=_as_int(metrics.get("input_tokens"))
                or _as_int(span_metrics.get("tokens.input")),
                output_tokens=_as_int(metrics.get("output_tokens"))
                or _as_int(span_metrics.get("tokens.output")),
                total_tokens=_as_int(metrics.get("total_tokens"))
                or _as_int(span_metrics.get("tokens.total")),
                message_count=len(messages) if isinstance(messages, list) else 0,
                tool_schema_count=len(tools) if isinstance(tools, list) else 0,
                response_tool_call_count=(
                    len(response_tool_calls)
                    if isinstance(response_tool_calls, list)
                    else 0
                ),
                output_preview=(
                    span.output_preview
                    or (
                        _json_text(response_content)[:280]
                        if response_content is not None
                        else None
                    )
                ),
            )
        )
    return records


def _collect_milestones_from_trace(trace: Trace) -> list[MilestoneRecord]:
    milestones: list[MilestoneRecord] = []
    milestone_index_by_id: dict[str, int] = {}
    for span in sorted(trace.spans, key=_span_sort_key):
        if span.kind != SpanKind.RUNTIME or span.name != "review_milestones":
            continue
        sequence = _span_sequence(span)
        for raw in _extract_review_fact_milestones(span):
            milestone_id = str(raw["id"])
            declared_at_seq = _as_int(raw.get("declared_at_seq"))
            record = MilestoneRecord(
                id=milestone_id,
                description=str(raw["description"]),
                status=str(raw.get("status") or "pending"),
                declared_at_seq=(
                    declared_at_seq if declared_at_seq is not None else sequence
                ),
                completed_at_seq=(
                    _as_int(raw.get("completed_at_seq"))
                    if raw.get("completed_at_seq") is not None
                    else sequence
                    if raw.get("status") == "completed"
                    else None
                ),
            )
            existing_index = milestone_index_by_id.get(milestone_id)
            if existing_index is None:
                milestone_index_by_id[milestone_id] = len(milestones)
                milestones.append(record)
                continue
            existing = milestones[existing_index]
            milestones[existing_index] = MilestoneRecord(
                id=existing.id,
                description=record.description,
                status=record.status,
                declared_at_seq=existing.declared_at_seq,
                completed_at_seq=(
                    record.completed_at_seq
                    if record.completed_at_seq is not None
                    else existing.completed_at_seq
                ),
            )
    return milestones


def _active_milestone_id_for_board(
    milestones: list[MilestoneRecord],
    latest_cycle: ReviewCycleRecord | None,
) -> str | None:
    active_milestone_id = next(
        (milestone.id for milestone in milestones if milestone.status == "active"),
        None,
    )
    if active_milestone_id is not None or latest_cycle is None:
        return active_milestone_id
    return _milestone_id_from_description(
        milestones,
        latest_cycle.active_milestone,
    )


def _latest_checkpoint_for_board(
    trace: Trace,
    *,
    milestones: list[MilestoneRecord],
    latest_cycle: ReviewCycleRecord | None,
    active_milestone_id: str | None,
) -> ReviewCheckpointRecord | None:
    for span in reversed(sorted(trace.spans, key=_span_sort_key)):
        if span.kind != SpanKind.RUNTIME or span.name != "review_checkpoint":
            continue
        milestone_id = _string_value(span.attributes.get("milestone_id"))
        if milestone_id is None:
            continue
        return ReviewCheckpointRecord(
            seq=_as_int(span.attributes.get("checkpoint_seq")) or _span_sequence(span),
            milestone_id=milestone_id,
            confirmed_at=span.start_time or trace.start_time,
        )
    if latest_cycle is None:
        return None
    milestone_id = (
        _milestone_id_from_description(milestones, latest_cycle.active_milestone)
        or active_milestone_id
        or latest_cycle.active_milestone
    )
    if milestone_id is None:
        return None
    return ReviewCheckpointRecord(
        seq=_cycle_sequence(latest_cycle) or 0,
        milestone_id=milestone_id,
        confirmed_at=latest_cycle.started_at
        or latest_cycle.resolved_at
        or trace.start_time,
    )


def _latest_review_outcome_for_board(
    latest_cycle: ReviewCycleRecord | None,
) -> ReviewOutcomeRecord | None:
    if latest_cycle is None:
        return None
    return ReviewOutcomeRecord(
        aligned=latest_cycle.aligned,
        experience=latest_cycle.experience,
        step_back_applied=latest_cycle.step_back_applied,
        affected_count=latest_cycle.affected_count,
        trigger_reason=latest_cycle.trigger_reason,
        active_milestone=latest_cycle.active_milestone,
        resolved_at=latest_cycle.resolved_at,
    )


def build_session_milestone_board(
    *,
    session_id: str,
    trace: Trace | None,
    review_cycles: list[ReviewCycleRecord],
) -> SessionMilestoneBoardRecord | None:
    if trace is None:
        return None

    milestones = _collect_milestones_from_trace(trace)
    latest_cycle = review_cycles[-1] if review_cycles else None
    active_milestone_id = _active_milestone_id_for_board(milestones, latest_cycle)
    latest_checkpoint = _latest_checkpoint_for_board(
        trace,
        milestones=milestones,
        latest_cycle=latest_cycle,
        active_milestone_id=active_milestone_id,
    )
    latest_review_outcome = _latest_review_outcome_for_board(latest_cycle)

    pending_review_reason = next(
        (
            cycle.trigger_reason
            for cycle in reversed(review_cycles)
            if cycle.resolved_at is None
        ),
        None,
    )

    if (
        not milestones
        and latest_checkpoint is None
        and latest_review_outcome is None
        and pending_review_reason is None
    ):
        return None

    return SessionMilestoneBoardRecord(
        session_id=session_id,
        run_id=(trace.spans[0].run_id if trace.spans else None),
        milestones=milestones,
        active_milestone_id=active_milestone_id,
        latest_checkpoint=latest_checkpoint,
        latest_review_outcome=latest_review_outcome,
        pending_review_reason=pending_review_reason,
    )


def build_conversation_events(
    *,
    session_id: str,
    steps: list[StepView],
    review_cycles: list[ReviewCycleRecord],
) -> list[ConversationEventRecord]:
    events: list[ConversationEventRecord] = []

    for step in sorted(steps, key=lambda item: item.sequence):
        if step.is_user_step():
            events.append(
                ConversationEventRecord(
                    id=step.id,
                    session_id=session_id,
                    run_id=step.run_id,
                    sequence=step.sequence,
                    kind="user_message",
                    priority="primary",
                    title="User",
                    summary=_summary_from_step(step),
                    details=_step_event_details(step),
                )
            )
            continue

        if step.is_assistant_step():
            events.append(
                ConversationEventRecord(
                    id=step.id,
                    session_id=session_id,
                    run_id=step.run_id,
                    sequence=step.sequence,
                    kind="assistant_message",
                    priority="primary",
                    title="Assistant",
                    summary=_summary_from_step(step),
                    details=_step_event_details(step),
                )
            )
            continue

        tool_name = step.name or "tool"
        if step.condensed_content:
            events.append(
                ConversationEventRecord(
                    id=step.id,
                    session_id=session_id,
                    run_id=step.run_id,
                    sequence=step.sequence,
                    kind="compressed_history_event",
                    priority="muted",
                    title="Compressed History",
                    summary=_summarize_text(step.condensed_content),
                    details=_step_event_details(step, tool_name=tool_name),
                )
            )
            continue

        if tool_name == "declare_milestones":
            summary = _summary_from_step(step) or "Milestones updated"
            events.append(
                ConversationEventRecord(
                    id=step.id,
                    session_id=session_id,
                    run_id=step.run_id,
                    sequence=step.sequence,
                    kind="milestone_event",
                    priority="secondary",
                    title="Milestones Updated",
                    summary=summary,
                    details=_step_event_details(step, tool_name=tool_name),
                )
            )
            continue

        if tool_name == "review_trajectory":
            events.append(
                ConversationEventRecord(
                    id=step.id,
                    session_id=session_id,
                    run_id=step.run_id,
                    sequence=step.sequence,
                    kind="review_event",
                    priority="muted",
                    title="Review",
                    summary=_summary_from_step(step),
                    details=_step_event_details(step, tool_name=tool_name),
                )
            )
            continue

        events.append(
            ConversationEventRecord(
                id=step.id,
                session_id=session_id,
                run_id=step.run_id,
                sequence=step.sequence,
                kind="tool_event",
                priority="secondary",
                title=f"Tool: {tool_name}",
                summary=_summary_from_step(step),
                details=_step_event_details(step, tool_name=tool_name),
            )
        )

    existing_review_sequences = {
        event.sequence for event in events if event.kind == "review_event"
    }
    for cycle in review_cycles:
        sequence = _cycle_sequence(cycle)
        if sequence in existing_review_sequences:
            continue
        events.append(
            ConversationEventRecord(
                id=f"review:{cycle.cycle_id}",
                session_id=session_id,
                run_id=cycle.run_id,
                sequence=sequence,
                kind="review_event",
                priority="muted",
                title="Review",
                summary=_review_cycle_summary(cycle),
                details={
                    "cycle_id": cycle.cycle_id,
                    "trigger_reason": cycle.trigger_reason,
                    "aligned": cycle.aligned,
                    "step_back_applied": cycle.step_back_applied,
                    "experience": cycle.experience,
                    "active_milestone": cycle.active_milestone,
                },
            )
        )

    events.sort(
        key=lambda event: (
            event.sequence if event.sequence is not None else 1_000_000_000,
            {"primary": 0, "secondary": 1, "muted": 2}.get(event.priority, 9),
            event.id,
        )
    )
    return events


__all__ = [
    "build_conversation_events",
    "build_runtime_decision_record_from_entry",
    "build_runtime_decision_record_from_span",
    "build_session_milestone_board",
    "build_trace_llm_call_records",
    "build_trace_mainline_events",
    "build_trace_review_cycles",
    "build_trace_runtime_decisions",
    "build_trace_timeline_events",
    "parse_system_review_notice",
]
