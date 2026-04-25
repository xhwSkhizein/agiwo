"""Read-model builders for session and trace runtime observability."""

import json
import re
from typing import Any

from agiwo.agent.models.log import (
    CompactionApplied,
    CompactionFailed,
    HookFailed,
    RunLogEntry,
    RunRolledBack,
    StepBackApplied,
    TerminationDecided,
)
from agiwo.observability.trace import Span, SpanKind, SpanStatus, Trace

from server.models.session import RuntimeDecisionRecord, TraceTimelineEventRecord

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


def _iso(value: object) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


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


def build_runtime_decision_record_from_span(
    trace: Trace,
    span: Span,
) -> RuntimeDecisionRecord | None:
    if span.kind != SpanKind.RUNTIME:
        return None
    sequence = _as_int(span.attributes.get("sequence")) or 0
    if span.name == "compaction":
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
                "after_token_estimate": _as_int(
                    span.attributes.get("after_token_estimate")
                )
                or 0,
                "message_count": _as_int(span.attributes.get("message_count")) or 0,
                "summary": span.attributes.get("summary"),
                "transcript_path": span.attributes.get("transcript_path"),
            },
        )
    if span.name == "compaction_failed":
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
    if span.name == "step_back":
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
    if span.name == "rollback":
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
    if span.name == "termination":
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
    if span.name == "hook_failed":
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
    return None


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
    decisions.sort(key=lambda item: (_iso(item.created_at) or "", item.sequence))
    return decisions


def _timeline_sort_key(event: TraceTimelineEventRecord) -> tuple[str, int, int, str]:
    return (
        event.timestamp or "",
        event.sequence if event.sequence is not None else 1_000_000_000,
        _TIMELINE_KIND_ORDER.get(event.kind, 999),
        event.title,
    )


def build_trace_timeline_events(trace: Trace) -> list[TraceTimelineEventRecord]:
    events: list[TraceTimelineEventRecord] = []

    for span in trace.spans:
        sequence = _as_int(span.attributes.get("sequence"))
        agent_id = _runtime_agent_id(trace, span)
        status = str(_enum_value(span.status))

        if span.kind == SpanKind.AGENT:
            events.append(
                TraceTimelineEventRecord(
                    kind="run_started",
                    timestamp=_iso(span.start_time),
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
            )
            if span.end_time is not None:
                completed_kind = (
                    "run_failed" if span.status == SpanStatus.ERROR else "run_finished"
                )
                events.append(
                    TraceTimelineEventRecord(
                        kind=completed_kind,
                        timestamp=_iso(span.end_time),
                        sequence=_as_int(span.attributes.get("end_sequence")),
                        run_id=span.run_id,
                        agent_id=agent_id,
                        span_id=span.span_id,
                        title="Run Failed"
                        if completed_kind == "run_failed"
                        else "Run Finished",
                        summary=(
                            span.error_message
                            if completed_kind == "run_failed"
                            else (span.output_preview or "run completed")
                        ),
                        status=status,
                        details={
                            "agent_id": agent_id,
                            "duration_ms": span.duration_ms,
                            "termination_reason": span.attributes.get(
                                "termination_reason"
                            ),
                            "error": span.error_message,
                        },
                    )
                )
            continue

        if span.kind == SpanKind.LLM_CALL:
            details = dict(span.llm_details or {})
            events.append(
                TraceTimelineEventRecord(
                    kind="llm_call",
                    timestamp=_iso(span.start_time),
                    sequence=sequence,
                    run_id=span.run_id,
                    agent_id=agent_id,
                    span_id=span.span_id,
                    title="LLM Call",
                    summary=(
                        f"{span.name} · {details.get('finish_reason') or 'completed'}"
                    ),
                    status=status,
                    details=details,
                )
            )
            continue

        if span.kind == SpanKind.TOOL_CALL:
            tool_details = dict(span.tool_details or {})
            tool_name = str(tool_details.get("tool_name") or span.name)
            raw_output = _json_text(tool_details.get("output"))
            review_notice = parse_system_review_notice(raw_output)
            if review_notice is not None:
                trigger_reason = review_notice.get("trigger_reason") or "unknown"
                step_count = review_notice.get("steps_since_last_review")
                step_summary = (
                    f"{step_count} steps" if step_count is not None else "unknown steps"
                )
                events.append(
                    TraceTimelineEventRecord(
                        kind="review_checkpoint",
                        timestamp=_iso(span.start_time),
                        sequence=sequence,
                        run_id=span.run_id,
                        agent_id=agent_id,
                        span_id=span.span_id,
                        step_id=span.step_id,
                        title="Review Checkpoint",
                        summary=f"triggered by {trigger_reason} after {step_summary}",
                        status="ok",
                        details=review_notice,
                    )
                )
            if tool_name == "review_trajectory":
                aligned_match = _ALIGNED_RE.search(raw_output)
                aligned = (
                    aligned_match.group("value").lower()
                    if aligned_match is not None
                    else "unknown"
                )
                events.append(
                    TraceTimelineEventRecord(
                        kind="review_result",
                        timestamp=_iso(span.start_time),
                        sequence=sequence,
                        run_id=span.run_id,
                        agent_id=agent_id,
                        span_id=span.span_id,
                        step_id=span.step_id,
                        title="Review Result",
                        summary=(
                            "trajectory aligned"
                            if aligned == "true"
                            else "trajectory misaligned"
                            if aligned == "false"
                            else "trajectory reviewed"
                        ),
                        status=status,
                        details={
                            "tool_name": tool_name,
                            "tool_call_id": tool_details.get("tool_call_id"),
                            "aligned": aligned,
                            "raw_output": raw_output,
                        },
                    )
                )
                continue
            if tool_name == "declare_milestones":
                input_args = tool_details.get("input_args")
                milestones = (
                    input_args.get("milestones", [])
                    if isinstance(input_args, dict)
                    else []
                )
                events.append(
                    TraceTimelineEventRecord(
                        kind="milestone_update",
                        timestamp=_iso(span.start_time),
                        sequence=sequence,
                        run_id=span.run_id,
                        agent_id=agent_id,
                        span_id=span.span_id,
                        step_id=span.step_id,
                        title="Milestone Update",
                        summary=f"{len(milestones)} milestones declared/updated",
                        status=status,
                        details={"milestones": milestones},
                    )
                )
                continue
            events.append(
                TraceTimelineEventRecord(
                    kind="tool_call",
                    timestamp=_iso(span.start_time),
                    sequence=sequence,
                    run_id=span.run_id,
                    agent_id=agent_id,
                    span_id=span.span_id,
                    step_id=span.step_id,
                    title=f"Tool Call: {tool_name}",
                    summary=str(tool_details.get("status") or "completed"),
                    status=status,
                    details=tool_details,
                )
            )
            continue

        if span.kind == SpanKind.RUNTIME:
            record = build_runtime_decision_record_from_span(trace, span)
            if record is None:
                continue
            timeline_kind = (
                "hook_failed" if record.kind == "hook_failed" else "runtime_decision"
            )
            details = dict(record.details)
            if timeline_kind == "runtime_decision":
                details = {"kind": record.kind, **details}
            events.append(
                TraceTimelineEventRecord(
                    kind=timeline_kind,
                    timestamp=_iso(record.created_at),
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
            )

    events.sort(key=_timeline_sort_key)
    return events


__all__ = [
    "build_runtime_decision_record_from_entry",
    "build_runtime_decision_record_from_span",
    "build_trace_runtime_decisions",
    "build_trace_timeline_events",
    "parse_system_review_notice",
]
