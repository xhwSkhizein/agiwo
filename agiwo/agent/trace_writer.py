"""Agent trace collector — build Trace/Span trees from run-log and step callbacks."""

import json
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from agiwo.agent.models.log import (
    AssistantStepCommitted,
    CompactionApplied,
    LLMCallCompleted,
    LLMCallStarted,
    RetrospectApplied,
    RunFailed as RunFailedEntry,
    RunFinished,
    RunLogEntry,
    RunStarted,
    TerminationDecided,
    ToolStepCommitted,
)
from agiwo.agent.models.run import RunOutput
from agiwo.agent.models.step import LLMCallContext, StepView
from agiwo.observability.base import BaseTraceStorage
from agiwo.observability.trace import Span, SpanKind, SpanStatus, Trace

from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


def _utc(value: datetime | None) -> datetime | None:
    return (
        value.replace(tzinfo=timezone.utc) if value and value.tzinfo is None else value
    )


def _resolve_parent(
    step: StepView,
    run_spans: dict[str, Span],
    fallback: Span | None,
) -> tuple[str | None, int]:
    parent = run_spans.get(step.run_id)
    if not parent and step.parent_run_id:
        parent = run_spans.get(step.parent_run_id)
    parent = parent or fallback
    return (parent.span_id, parent.depth) if parent else (None, 0)


def _extract_tool_args(step: StepView, cache: dict[str, StepView]) -> dict[str, Any]:
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


def _build_tool_span(
    trace_id: str,
    step: StepView,
    cache: dict[str, StepView],
    run_spans: dict[str, Span],
    fallback: Span | None,
) -> Span:
    m = step.metrics
    start = _utc((m.start_at if m and m.start_at else None) or step.created_at)
    end = _utc(m.end_at if m else None)
    dur = m.duration_ms if m else None
    parent_id, parent_depth = _resolve_parent(step, run_spans, fallback)
    is_error = step.is_error

    span = Span(
        trace_id=trace_id,
        parent_span_id=parent_id,
        kind=SpanKind.TOOL_CALL,
        name=step.name or "tool",
        depth=parent_depth + 1,
        attributes={"tool_name": step.name, "tool_call_id": step.tool_call_id},
        tool_details={
            "tool_name": step.name,
            "tool_call_id": step.tool_call_id,
            "input_args": _extract_tool_args(step, cache),
            "output": step.content,
            "content_for_user": step.content_for_user,
            "error": step.content if is_error else None,
            "status": "error" if is_error else "completed",
            **({"metrics": {"duration_ms": dur}} if m else {}),
        },
        step_id=step.id,
        run_id=step.run_id,
        start_time=start,
        end_time=end,
        duration_ms=dur,
        status=SpanStatus.ERROR if is_error else SpanStatus.OK,
        error_message=step.content if is_error else None,
    )
    if m:
        span.metrics = {"tool.exec_time_ms": dur, "duration_ms": span.duration_ms}
    return span


def _build_assistant_span(
    trace_id: str,
    step: StepView,
    llm: LLMCallContext | None,
    run_spans: dict[str, Span],
    fallback: Span | None,
    preview_length: int,
) -> Span:
    m = step.metrics
    start = _utc((m.start_at if m and m.start_at else None) or step.created_at)
    end = _utc(m.end_at if m else None)
    dur = m.duration_ms if m else None
    parent_id, parent_depth = _resolve_parent(step, run_spans, fallback)
    name = m.model_name if m and m.model_name else "llm"

    llm_details: dict[str, Any] = {}
    if llm:
        llm_details = {
            "request": llm.request_params or {},
            "messages": llm.messages,
            "tools": llm.tools,
            "response_content": step.content,
            "response_tool_calls": step.tool_calls,
            "finish_reason": llm.finish_reason,
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
            "has_tool_calls": bool(step.tool_calls),
        },
        step_id=step.id,
        run_id=step.run_id,
        start_time=start,
        end_time=end,
        duration_ms=dur,
        status=SpanStatus.OK,
        llm_details=llm_details,
        output_preview=(
            step.content[:preview_length]
            if isinstance(step.content, str) and step.content
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
        attributes={"tool_name": step.name, "tool_call_id": step.tool_call_id},
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
    entry: CompactionApplied | RetrospectApplied | TerminationDecided,
    run_span: Span | None,
) -> Span:
    parent_id = run_span.span_id if run_span else None
    parent_depth = run_span.depth if run_span else 0
    attributes: dict[str, Any] = {}
    name = "runtime"
    if isinstance(entry, CompactionApplied):
        name = "compaction"
        attributes = {
            "start_sequence": entry.start_sequence,
            "end_sequence": entry.end_sequence,
            "transcript_path": entry.transcript_path,
            "summary": entry.summary,
        }
    elif isinstance(entry, RetrospectApplied):
        name = "retrospect"
        attributes = {
            "affected_sequences": list(entry.affected_sequences),
            "affected_step_ids": list(entry.affected_step_ids),
            "feedback": entry.feedback,
            "replacement": entry.replacement,
            "trigger": entry.trigger,
        }
    elif isinstance(entry, TerminationDecided):
        name = "termination"
        attributes = {
            "termination_reason": entry.termination_reason.value,
            "phase": entry.phase,
            "source": entry.source,
        }
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
        status=SpanStatus.OK,
    )
    return span


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
    span = Span(
        trace_id=trace.trace_id,
        kind=SpanKind.AGENT,
        name=entry.agent_id,
        depth=0,
        attributes={
            "agent_id": entry.agent_id,
            "session_id": entry.session_id,
            "nested": False,
            "parent_run_id": None,
        },
        run_id=entry.run_id,
        start_time=entry.created_at,
    )
    if trace.root_span_id is None:
        trace.root_span_id = span.span_id
    trace.add_span(span)
    run_spans[entry.run_id] = span


def _complete_trace_from_run_finished(
    trace: Trace,
    run_spans: dict[str, Span],
    entry: RunFinished,
    preview_length: int,
) -> None:
    span = run_spans.get(entry.run_id)
    if span is not None:
        span.end_time = entry.created_at
        span.duration_ms = (entry.created_at - span.start_time).total_seconds() * 1000
        span.status = SpanStatus.OK
        span.output_preview = (
            entry.response[:preview_length] if entry.response else None
        )
    trace.end_time = entry.created_at
    trace.status = SpanStatus.OK
    trace.final_output = entry.response
    trace.duration_ms = (trace.end_time - trace.start_time).total_seconds() * 1000


def _complete_trace_from_run_failed(
    trace: Trace,
    run_spans: dict[str, Span],
    entry: RunFailedEntry,
) -> None:
    span = run_spans.get(entry.run_id)
    if span is not None:
        span.end_time = entry.created_at
        span.duration_ms = (entry.created_at - span.start_time).total_seconds() * 1000
        span.status = SpanStatus.ERROR
        span.error_message = entry.error
    trace.end_time = entry.created_at
    trace.status = SpanStatus.ERROR
    trace.duration_ms = (trace.end_time - trace.start_time).total_seconds() * 1000


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
        return True
    return False


def _apply_tool_entry_to_trace(
    trace: Trace,
    entry: RunLogEntry,
    *,
    run_spans: dict[str, Span],
    assistant_cache: OrderedDict[str, AssistantStepCommitted],
) -> bool:
    if isinstance(entry, AssistantStepCommitted) and entry.tool_calls:
        for tc in entry.tool_calls:
            tool_call_id = tc.get("id")
            if tool_call_id:
                assistant_cache[tool_call_id] = entry
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
        entry, (CompactionApplied, RetrospectApplied, TerminationDecided)
    ):
        return False
    trace.add_span(
        _build_runtime_span_from_entry(
            trace.trace_id,
            entry,
            run_spans.get(entry.run_id),
        )
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
    ):
        return
    _apply_runtime_entry_to_trace(
        trace,
        entry,
        run_spans=run_spans,
    )


class AgentTraceCollector:
    """Construct a Trace from explicit run and step callbacks."""

    PREVIEW_LENGTH = 500

    def __init__(self, store: BaseTraceStorage | None = None) -> None:
        self.store = store
        self._trace: Trace | None = None
        self._run_spans: dict[str, Span] = {}
        self._assistant_cache: OrderedDict[str, StepView] = OrderedDict()

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
        self._assistant_cache = OrderedDict()

    def on_run_started(
        self,
        *,
        run_id: str,
        agent_id: str,
        session_id: str,
        parent_run_id: str | None,
    ) -> str:
        trace = self._require_trace()
        run_trace_id = trace.trace_id
        parent = self._run_spans.get(parent_run_id) if parent_run_id else None
        span = Span(
            trace_id=trace.trace_id,
            parent_span_id=parent.span_id if parent else None,
            kind=SpanKind.AGENT,
            name=agent_id,
            depth=(parent.depth + 1) if parent else 0,
            attributes={
                "agent_id": agent_id,
                "session_id": session_id,
                "nested": parent_run_id is not None,
                "parent_run_id": parent_run_id,
            },
            run_id=run_id,
        )
        if trace.root_span_id is None:
            trace.root_span_id = span.span_id
        trace.add_span(span)
        self._run_spans[run_id] = span
        return run_trace_id

    async def on_step(self, step: StepView, llm: LLMCallContext | None = None) -> None:
        trace = self._require_trace()
        self._cache_tool_calls(step)
        fallback = self._run_spans.get(step.run_id) or (
            self._run_spans.get(step.parent_run_id) if step.parent_run_id else None
        )
        if step.role.value == "assistant":
            trace.add_span(
                _build_assistant_span(
                    trace.trace_id,
                    step,
                    llm,
                    self._run_spans,
                    fallback,
                    self.PREVIEW_LENGTH,
                )
            )
        elif step.role.value == "tool":
            trace.add_span(
                _build_tool_span(
                    trace.trace_id,
                    step,
                    self._assistant_cache,
                    self._run_spans,
                    fallback,
                )
            )

    async def on_run_completed(self, output: RunOutput, *, run_id: str) -> None:
        trace = self._require_trace()
        span = self._run_spans.get(run_id)
        if span is not None:
            span.complete(
                status=SpanStatus.OK,
                output_preview=output.response[: self.PREVIEW_LENGTH]
                if output.response
                else None,
            )
        if span is not None and span.parent_span_id is None:
            trace.complete(status=SpanStatus.OK, final_output=output.response)
        await self._save_trace()

    async def on_run_failed(self, error: Exception, *, run_id: str) -> None:
        trace = self._require_trace()
        span = self._run_spans.get(run_id)
        if span is not None:
            span.complete(status=SpanStatus.ERROR, error_message=str(error))
        if span is not None and span.parent_span_id is None:
            trace.complete(status=SpanStatus.ERROR)
        await self._save_trace()

    async def stop(self) -> None:
        if self._trace is None:
            return
        if self._trace.end_time is None:
            self._trace.complete(status=self._trace.status)
        await self._save_trace()
        self._trace = None
        self._run_spans = {}
        self._assistant_cache = OrderedDict()

    async def on_run_log_entries(self, entries: list[RunLogEntry]) -> None:
        trace = self._trace
        if trace is None:
            return
        for entry in entries:
            if not isinstance(
                entry,
                (CompactionApplied, RetrospectApplied, TerminationDecided),
            ):
                continue
            run_span = self._run_spans.get(entry.run_id)
            trace.add_span(
                _build_runtime_span_from_entry(trace.trace_id, entry, run_span)
            )

    async def build_from_entries(self, entries: list[RunLogEntry]) -> Trace:
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

    _CACHE_MAX_SIZE = 10_000

    def _cache_tool_calls(self, step: StepView) -> None:
        if step.role.value != "assistant" or not step.tool_calls:
            return
        for tc in step.tool_calls:
            tid = tc.get("id")
            if tid:
                self._assistant_cache[tid] = step
        while len(self._assistant_cache) > self._CACHE_MAX_SIZE:
            self._assistant_cache.popitem(last=False)

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
