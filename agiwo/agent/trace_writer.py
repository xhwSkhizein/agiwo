"""Agent trace collector — build Trace/Span trees from committed run-log entries."""

import json
from collections import OrderedDict
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from agiwo.agent.models.log import (
    AssistantStepCommitted,
    CompactionApplied,
    CompactionFailed,
    HookFailed,
    LLMCallCompleted,
    LLMCallStarted,
    ReviewCheckpointRecorded,
    ReviewMilestonesUpdated,
    ReviewOutcomeRecorded,
    ReviewTriggerDecided,
    RunFailed as RunFailedEntry,
    RunFinished,
    RunLogEntry,
    RunRolledBack,
    RunStarted,
    StepBackApplied,
    TerminationDecided,
    ToolStepCommitted,
)
from agiwo.observability.base import BaseTraceStorage
from agiwo.observability.trace import Span, SpanKind, SpanStatus, Trace

from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


def _utc(value: datetime | None) -> datetime | None:
    return (
        value.replace(tzinfo=timezone.utc) if value and value.tzinfo is None else value
    )


def _build_assistant_span_from_entries(
    trace_id: str,
    run_span: Span | None,
    started: LLMCallStarted | None,
    completed: LLMCallCompleted,
    preview_length: int,
) -> Span:
    m = completed.metrics
    start = _utc((m.start_at if m and m.start_at else None) or completed.created_at)
    end = _utc(m.end_at if m else None)
    dur = m.duration_ms if m else None
    parent_id = run_span.span_id if run_span else None
    parent_depth = run_span.depth if run_span else 0
    name = m.model_name if m and m.model_name else "llm"

    llm_details: dict[str, Any] = {
        "request": {},
        "messages": started.messages if started is not None else [],
        "tools": started.tools if started is not None else None,
        "response_content": completed.content,
        "response_tool_calls": completed.tool_calls,
        "finish_reason": completed.finish_reason,
        "status": "completed",
    }
    if m:
        llm_details["metrics"] = {
            "duration_ms": dur,
            "first_token_ms": m.first_token_latency_ms,
            "input_tokens": m.input_tokens,
            "output_tokens": m.output_tokens,
            "total_tokens": m.total_tokens,
            "cache_read_tokens": m.cache_read_tokens,
            "cache_creation_tokens": m.cache_creation_tokens,
            "usage_source": m.usage_source,
        }

    span = Span(
        trace_id=trace_id,
        parent_span_id=parent_id,
        kind=SpanKind.LLM_CALL,
        name=name,
        depth=parent_depth + 1,
        attributes={
            "model_name": m.model_name if m else None,
            "provider": m.provider if m else None,
            "has_tool_calls": bool(completed.tool_calls),
            "sequence": completed.sequence,
            "agent_id": completed.agent_id,
        },
        run_id=completed.run_id,
        start_time=start or completed.created_at,
        end_time=end,
        duration_ms=dur,
        status=SpanStatus.OK,
        llm_details=llm_details,
        output_preview=(
            completed.content[:preview_length]
            if isinstance(completed.content, str) and completed.content
            else None
        ),
    )
    if m:
        span.metrics = {
            "tokens.input": m.input_tokens,
            "tokens.output": m.output_tokens,
            "tokens.total": m.total_tokens,
            "tokens.cache_read": m.cache_read_tokens,
            "tokens.cache_creation": m.cache_creation_tokens,
            "token_cost": m.token_cost,
            "first_token_ms": m.first_token_latency_ms,
            "duration_ms": dur,
            "model": m.model_name,
            "provider": m.provider,
            "usage_source": m.usage_source,
        }
    return span


def _extract_tool_args_from_committed(
    step: ToolStepCommitted,
    cache: dict[str, AssistantStepCommitted],
) -> dict[str, Any]:
    if not step.tool_call_id:
        return {}
    assistant = cache.get(step.tool_call_id)
    if not assistant or not assistant.tool_calls:
        return {}
    for tc in assistant.tool_calls:
        if tc.get("id") != step.tool_call_id:
            continue
        raw = tc.get("function", {}).get("arguments", "{}")
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}
        return raw if isinstance(raw, dict) else {}
    return {}


def _remove_tool_call_index(
    run_to_tool_calls: dict[str, set[str]] | None,
    *,
    run_id: str,
    tool_call_id: str,
) -> None:
    if run_to_tool_calls is None:
        return
    tool_call_ids = run_to_tool_calls.get(run_id)
    if not tool_call_ids:
        return
    tool_call_ids.discard(tool_call_id)
    if not tool_call_ids:
        run_to_tool_calls.pop(run_id, None)


def _cache_assistant_tool_call(
    assistant_cache: OrderedDict[str, AssistantStepCommitted],
    *,
    assistant: AssistantStepCommitted,
    tool_call_id: str,
    run_to_tool_calls: dict[str, set[str]] | None = None,
) -> None:
    previous = assistant_cache.get(tool_call_id)
    if previous is not None and previous.run_id != assistant.run_id:
        _remove_tool_call_index(
            run_to_tool_calls,
            run_id=previous.run_id,
            tool_call_id=tool_call_id,
        )
    assistant_cache[tool_call_id] = assistant
    if run_to_tool_calls is not None:
        run_to_tool_calls.setdefault(assistant.run_id, set()).add(tool_call_id)


def _build_tool_span_from_entry(
    trace_id: str,
    step: ToolStepCommitted,
    cache: dict[str, AssistantStepCommitted],
    run_span: Span | None,
) -> Span:
    m = step.metrics
    start = _utc((m.start_at if m and m.start_at else None) or step.created_at)
    end = _utc(m.end_at if m else None)
    dur = m.duration_ms if m else None
    parent_id = run_span.span_id if run_span else None
    parent_depth = run_span.depth if run_span else 0

    span = Span(
        trace_id=trace_id,
        parent_span_id=parent_id,
        kind=SpanKind.TOOL_CALL,
        name=step.name or "tool",
        depth=parent_depth + 1,
        attributes={
            "tool_name": step.name,
            "tool_call_id": step.tool_call_id,
            "sequence": step.sequence,
            "agent_id": step.agent_id,
        },
        tool_details={
            "tool_name": step.name,
            "tool_call_id": step.tool_call_id,
            "input_args": _extract_tool_args_from_committed(step, cache),
            "output": step.content,
            "content_for_user": step.content_for_user,
            "error": step.content if step.is_error else None,
            "status": "error" if step.is_error else "completed",
            **({"metrics": {"duration_ms": dur}} if m else {}),
        },
        run_id=step.run_id,
        step_id=step.step_id,
        start_time=start or step.created_at,
        end_time=end,
        duration_ms=dur,
        status=SpanStatus.ERROR if step.is_error else SpanStatus.OK,
        error_message=step.content if step.is_error else None,
    )
    if m:
        span.metrics = {"tool.exec_time_ms": dur, "duration_ms": span.duration_ms}
    return span


def _build_runtime_span_from_entry(
    trace_id: str,
    entry: (
        CompactionApplied
        | CompactionFailed
        | HookFailed
        | ReviewCheckpointRecorded
        | ReviewMilestonesUpdated
        | ReviewOutcomeRecorded
        | ReviewTriggerDecided
        | RunRolledBack
        | StepBackApplied
        | TerminationDecided
    ),
    run_span: Span | None,
) -> Span:
    parent_id = run_span.span_id if run_span else None
    parent_depth = run_span.depth if run_span else 0
    attributes: dict[str, Any] = {
        "sequence": entry.sequence,
        "agent_id": entry.agent_id,
    }
    name = "runtime"
    status = SpanStatus.OK
    error_message: str | None = None
    if isinstance(entry, CompactionApplied):
        name = "compaction"
        attributes.update(
            {
                "start_sequence": entry.start_sequence,
                "end_sequence": entry.end_sequence,
                "before_token_estimate": entry.before_token_estimate,
                "after_token_estimate": entry.after_token_estimate,
                "message_count": entry.message_count,
                "transcript_path": entry.transcript_path,
                "summary": entry.summary,
            }
        )
    elif isinstance(entry, CompactionFailed):
        name = "compaction_failed"
        status = SpanStatus.ERROR
        error_message = entry.error
        attributes.update(
            {
                "error": entry.error,
                "attempt": entry.attempt,
                "max_attempts": entry.max_attempts,
                "terminal": entry.terminal,
            }
        )
    elif isinstance(entry, StepBackApplied):
        name = "step_back"
        attributes.update(
            {
                "affected_count": entry.affected_count,
                "checkpoint_seq": entry.checkpoint_seq,
                "experience": entry.experience,
            }
        )
    elif isinstance(entry, RunRolledBack):
        name = "rollback"
        attributes.update(
            {
                "start_sequence": entry.start_sequence,
                "end_sequence": entry.end_sequence,
                "reason": entry.reason,
            }
        )
    elif isinstance(entry, TerminationDecided):
        name = "termination"
        attributes.update(
            {
                "termination_reason": entry.termination_reason.value,
                "phase": entry.phase,
                "source": entry.source,
            }
        )
    elif isinstance(entry, HookFailed):
        name = "hook_failed"
        status = SpanStatus.ERROR
        error_message = entry.error
        attributes.update(
            {
                "phase": entry.phase,
                "handler_name": entry.handler_name,
                "critical": entry.critical,
                "error": entry.error,
            }
        )
    elif isinstance(entry, ReviewMilestonesUpdated):
        name = "review_milestones"
        attributes.update(
            {
                "milestones": [asdict(milestone) for milestone in entry.milestones],
                "active_milestone_id": entry.active_milestone_id,
                "source_tool_call_id": entry.source_tool_call_id,
                "source_step_id": entry.source_step_id,
                "reason": entry.reason,
            }
        )
    elif isinstance(entry, ReviewTriggerDecided):
        name = "review_trigger"
        attributes.update(
            {
                "trigger_reason": entry.trigger_reason,
                "active_milestone_id": entry.active_milestone_id,
                "review_count_since_checkpoint": (entry.review_count_since_checkpoint),
                "trigger_tool_call_id": entry.trigger_tool_call_id,
                "trigger_tool_step_id": entry.trigger_tool_step_id,
                "notice_step_id": entry.notice_step_id,
            }
        )
    elif isinstance(entry, ReviewCheckpointRecorded):
        name = "review_checkpoint"
        attributes.update(
            {
                "checkpoint_seq": entry.checkpoint_seq,
                "milestone_id": entry.milestone_id,
                "review_tool_call_id": entry.review_tool_call_id,
                "review_step_id": entry.review_step_id,
            }
        )
    elif isinstance(entry, ReviewOutcomeRecorded):
        name = "review_outcome"
        attributes.update(
            {
                "aligned": entry.aligned,
                "mode": entry.mode,
                "experience": entry.experience,
                "active_milestone_id": entry.active_milestone_id,
                "review_tool_call_id": entry.review_tool_call_id,
                "review_step_id": entry.review_step_id,
                "hidden_step_ids": list(entry.hidden_step_ids),
                "notice_cleaned_step_ids": list(entry.notice_cleaned_step_ids),
                "condensed_step_ids": list(entry.condensed_step_ids),
            }
        )
    span = Span(
        trace_id=trace_id,
        parent_span_id=parent_id,
        kind=SpanKind.RUNTIME,
        name=name,
        depth=parent_depth + 1,
        attributes=attributes,
        run_id=entry.run_id,
        start_time=entry.created_at,
        end_time=entry.created_at,
        duration_ms=0.0,
        status=status,
        error_message=error_message,
    )
    return span


def _append_runtime_entry_to_trace(
    trace: Trace,
    entry: (
        CompactionApplied
        | CompactionFailed
        | HookFailed
        | ReviewCheckpointRecorded
        | ReviewMilestonesUpdated
        | ReviewOutcomeRecorded
        | ReviewTriggerDecided
        | RunRolledBack
        | StepBackApplied
        | TerminationDecided
    ),
    *,
    run_spans: dict[str, Span],
) -> None:
    trace.add_span(
        _build_runtime_span_from_entry(
            trace.trace_id,
            entry,
            run_spans.get(entry.run_id),
        )
    )


def _create_run_span(
    trace: Trace,
    *,
    run_id: str,
    agent_id: str,
    session_id: str,
    parent_run_id: str | None,
    run_spans: dict[str, Span],
    created_at: datetime | None = None,
    depth_hint: int = 0,
    start_sequence: int | None = None,
) -> Span:
    parent = run_spans.get(parent_run_id) if parent_run_id else None
    is_root = trace.root_span_id is None
    span = Span(
        trace_id=trace.trace_id,
        parent_span_id=parent.span_id if parent is not None else None,
        kind=SpanKind.AGENT,
        name=agent_id,
        depth=(parent.depth + 1)
        if parent is not None
        else (0 if is_root else depth_hint),
        attributes={
            "agent_id": agent_id,
            "session_id": session_id,
            "nested": not is_root,
            "parent_run_id": parent_run_id,
            "start_sequence": start_sequence,
        },
        run_id=run_id,
        start_time=created_at or datetime.now(timezone.utc),
    )
    if trace.root_span_id is None:
        trace.root_span_id = span.span_id
        if created_at is not None:
            trace.start_time = created_at
    trace.add_span(span)
    run_spans[run_id] = span
    return span


def _complete_run_span(
    span: Span | None,
    *,
    completed_at: datetime | None = None,
    status: SpanStatus,
    preview_length: int,
    response: str | None = None,
    error_message: str | None = None,
) -> None:
    if span is None:
        return
    if completed_at is None:
        span.complete(
            status=status,
            error_message=error_message,
            output_preview=response[:preview_length] if response else None,
        )
        return

    span.end_time = completed_at
    span.duration_ms = (completed_at - span.start_time).total_seconds() * 1000
    span.status = status
    span.error_message = error_message
    span.output_preview = response[:preview_length] if response else None


def _complete_root_trace(
    trace: Trace,
    span: Span | None,
    *,
    completed_at: datetime | None = None,
    status: SpanStatus,
    response: str | None = None,
) -> None:
    if span is None or span.span_id != trace.root_span_id:
        return
    if completed_at is None:
        trace.complete(status=status, final_output=response)
        return
    trace.end_time = completed_at
    trace.status = status
    trace.final_output = response
    trace.duration_ms = (completed_at - span.start_time).total_seconds() * 1000


def _start_run_span_from_entry(
    trace: Trace,
    entry: RunStarted,
    run_spans: dict[str, Span],
) -> None:
    if trace.agent_id is None:
        trace.agent_id = entry.agent_id
    if trace.session_id is None:
        trace.session_id = entry.session_id
    if trace.input_query is None and entry.user_input is not None:
        trace.input_query = str(entry.user_input)
    _create_run_span(
        trace,
        run_id=entry.run_id,
        agent_id=entry.agent_id,
        session_id=entry.session_id,
        parent_run_id=entry.parent_run_id,
        run_spans=run_spans,
        created_at=entry.created_at,
        depth_hint=entry.depth,
        start_sequence=entry.sequence,
    )


def _complete_trace_from_run_finished(
    trace: Trace,
    run_spans: dict[str, Span],
    entry: RunFinished,
    preview_length: int,
) -> None:
    span = run_spans.get(entry.run_id)
    if span is not None:
        span.attributes["end_sequence"] = entry.sequence
        span.attributes["termination_reason"] = (
            entry.termination_reason.value
            if entry.termination_reason is not None
            else None
        )
    _complete_run_span(
        span,
        completed_at=entry.created_at,
        status=SpanStatus.OK,
        preview_length=preview_length,
        response=entry.response,
    )
    _complete_root_trace(
        trace,
        span,
        completed_at=entry.created_at,
        status=SpanStatus.OK,
        response=entry.response,
    )


def _complete_trace_from_run_failed(
    trace: Trace,
    run_spans: dict[str, Span],
    entry: RunFailedEntry,
) -> None:
    span = run_spans.get(entry.run_id)
    if span is not None:
        span.attributes["end_sequence"] = entry.sequence
    _complete_run_span(
        span,
        completed_at=entry.created_at,
        status=SpanStatus.ERROR,
        preview_length=0,
        error_message=entry.error,
    )
    _complete_root_trace(
        trace,
        span,
        completed_at=entry.created_at,
        status=SpanStatus.ERROR,
    )


def _apply_run_entry_to_trace(
    trace: Trace,
    entry: RunLogEntry,
    *,
    run_spans: dict[str, Span],
    preview_length: int,
) -> bool:
    if isinstance(entry, RunStarted):
        _start_run_span_from_entry(trace, entry, run_spans)
        return True
    if isinstance(entry, RunFinished):
        _complete_trace_from_run_finished(trace, run_spans, entry, preview_length)
        return True
    if isinstance(entry, RunFailedEntry):
        _complete_trace_from_run_failed(trace, run_spans, entry)
        return True
    return False


def _apply_llm_entry_to_trace(
    trace: Trace,
    entry: RunLogEntry,
    *,
    run_spans: dict[str, Span],
    llm_started: dict[str, LLMCallStarted],
    preview_length: int,
) -> bool:
    if isinstance(entry, LLMCallStarted):
        llm_started[entry.run_id] = entry
        return True
    if isinstance(entry, LLMCallCompleted):
        trace.add_span(
            _build_assistant_span_from_entries(
                trace.trace_id,
                run_spans.get(entry.run_id),
                llm_started.get(entry.run_id),
                entry,
                preview_length,
            )
        )
        llm_started.pop(entry.run_id, None)
        return True
    return False


def _apply_tool_entry_to_trace(
    trace: Trace,
    entry: RunLogEntry,
    *,
    run_spans: dict[str, Span],
    assistant_cache: OrderedDict[str, AssistantStepCommitted],
    run_to_tool_calls: dict[str, set[str]] | None = None,
) -> bool:
    if isinstance(entry, AssistantStepCommitted) and entry.tool_calls:
        for tc in entry.tool_calls:
            tool_call_id = tc.get("id")
            if tool_call_id:
                _cache_assistant_tool_call(
                    assistant_cache,
                    assistant=entry,
                    tool_call_id=tool_call_id,
                    run_to_tool_calls=run_to_tool_calls,
                )
        return True
    if isinstance(entry, ToolStepCommitted):
        trace.add_span(
            _build_tool_span_from_entry(
                trace.trace_id,
                entry,
                assistant_cache,
                run_spans.get(entry.run_id),
            )
        )
        return True
    return False


def _apply_runtime_entry_to_trace(
    trace: Trace,
    entry: RunLogEntry,
    *,
    run_spans: dict[str, Span],
) -> bool:
    if not isinstance(
        entry,
        (
            CompactionApplied,
            CompactionFailed,
            HookFailed,
            ReviewCheckpointRecorded,
            ReviewMilestonesUpdated,
            ReviewOutcomeRecorded,
            ReviewTriggerDecided,
            RunRolledBack,
            StepBackApplied,
            TerminationDecided,
        ),
    ):
        return False
    _append_runtime_entry_to_trace(
        trace,
        entry,
        run_spans=run_spans,
    )
    return True


def _apply_entry_to_trace(
    trace: Trace,
    entry: RunLogEntry,
    *,
    run_spans: dict[str, Span],
    llm_started: dict[str, LLMCallStarted],
    assistant_cache: OrderedDict[str, AssistantStepCommitted],
    preview_length: int,
    run_to_tool_calls: dict[str, set[str]] | None = None,
) -> None:
    if _apply_run_entry_to_trace(
        trace,
        entry,
        run_spans=run_spans,
        preview_length=preview_length,
    ):
        return
    if _apply_llm_entry_to_trace(
        trace,
        entry,
        run_spans=run_spans,
        llm_started=llm_started,
        preview_length=preview_length,
    ):
        return
    if _apply_tool_entry_to_trace(
        trace,
        entry,
        run_spans=run_spans,
        assistant_cache=assistant_cache,
        run_to_tool_calls=run_to_tool_calls,
    ):
        return
    _apply_runtime_entry_to_trace(
        trace,
        entry,
        run_spans=run_spans,
    )


class AgentTraceCollector:
    """Construct a Trace from committed run-log entries."""

    PREVIEW_LENGTH = 500
    _CACHE_MAX_SIZE = 10_000

    def __init__(self, store: BaseTraceStorage | None = None) -> None:
        self.store = store
        self._trace: Trace | None = None
        self._run_spans: dict[str, Span] = {}
        self._assistant_committed_cache: OrderedDict[str, AssistantStepCommitted] = (
            OrderedDict()
        )
        self._run_to_tool_calls: dict[str, set[str]] = {}
        self._llm_started: dict[str, LLMCallStarted] = {}

    @property
    def trace_id(self) -> str | None:
        return self._trace.trace_id if self._trace is not None else None

    def start(
        self,
        trace_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        input_query: str | None = None,
    ) -> None:
        self._trace = Trace(
            trace_id=trace_id or str(uuid4()),
            agent_id=agent_id,
            session_id=session_id,
            user_id=user_id,
            input_query=input_query,
        )
        self._run_spans = {}
        self._assistant_committed_cache = OrderedDict()
        self._run_to_tool_calls = {}
        self._llm_started = {}

    async def stop(self) -> None:
        if self._trace is None:
            return
        if self._trace.end_time is None:
            self._trace.complete(status=self._trace.status)
        await self._save_trace()
        self._trace = None
        self._run_spans = {}
        self._assistant_committed_cache = OrderedDict()
        self._run_to_tool_calls = {}
        self._llm_started = {}

    async def on_run_log_entries(self, entries: list[RunLogEntry]) -> None:
        trace = self._trace
        if trace is None:
            return
        for entry in entries:
            _apply_entry_to_trace(
                trace,
                entry,
                run_spans=self._run_spans,
                llm_started=self._llm_started,
                assistant_cache=self._assistant_committed_cache,
                preview_length=self.PREVIEW_LENGTH,
                run_to_tool_calls=self._run_to_tool_calls,
            )
            while len(self._assistant_committed_cache) > self._CACHE_MAX_SIZE:
                tool_call_id, assistant = self._assistant_committed_cache.popitem(
                    last=False
                )
                _remove_tool_call_index(
                    self._run_to_tool_calls,
                    run_id=assistant.run_id,
                    tool_call_id=tool_call_id,
                )
            if isinstance(entry, (RunFinished, RunFailedEntry)):
                self._purge_run_correlation_state(entry.run_id)
        await self._save_trace()

    def build_from_entries(self, entries: list[RunLogEntry]) -> Trace:
        trace = Trace()
        run_spans: dict[str, Span] = {}
        llm_started: dict[str, LLMCallStarted] = {}
        assistant_cache: OrderedDict[str, AssistantStepCommitted] = OrderedDict()

        for entry in entries:
            _apply_entry_to_trace(
                trace,
                entry,
                run_spans=run_spans,
                llm_started=llm_started,
                assistant_cache=assistant_cache,
                preview_length=self.PREVIEW_LENGTH,
            )

        return trace

    def _require_trace(self) -> Trace:
        if self._trace is None:
            raise RuntimeError("trace_not_started")
        return self._trace

    def _purge_run_correlation_state(self, run_id: str) -> None:
        self._run_spans.pop(run_id, None)
        self._llm_started.pop(run_id, None)
        for tool_call_id in self._run_to_tool_calls.pop(run_id, set()):
            self._assistant_committed_cache.pop(tool_call_id, None)

    async def _save_trace(self) -> None:
        if self._trace is None or self.store is None:
            return
        try:
            await self.store.save_trace(self._trace)
        except Exception as error:  # noqa: BLE001 - observability boundary
            logger.warning(
                "trace_save_failed", trace_id=self._trace.trace_id, error=str(error)
            )


__all__ = ["AgentTraceCollector"]
