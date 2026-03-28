"""Agent trace collector — build Trace/Span trees from run and step callbacks."""

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from agiwo.agent.models.run import Run, RunOutput
from agiwo.agent.models.step import LLMCallContext, StepRecord
from agiwo.observability.base import BaseTraceStorage
from agiwo.observability.trace import Span, SpanKind, SpanStatus, Trace
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


def _utc(value: datetime | None) -> datetime | None:
    return (
        value.replace(tzinfo=timezone.utc) if value and value.tzinfo is None else value
    )


def _resolve_parent(
    step: StepRecord,
    run_spans: dict[str, Span],
    fallback: Span | None,
) -> tuple[str | None, int]:
    parent = run_spans.get(step.run_id)
    if not parent and step.parent_run_id:
        parent = run_spans.get(step.parent_run_id)
    parent = parent or fallback
    return (parent.span_id, parent.depth) if parent else (None, 0)


def _extract_tool_args(
    step: StepRecord, cache: dict[str, StepRecord]
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


def _build_tool_span(
    trace_id: str,
    step: StepRecord,
    cache: dict[str, StepRecord],
    run_spans: dict[str, Span],
    fallback: Span | None,
) -> Span:
    m = step.metrics
    start = _utc((m.start_at if m and m.start_at else None) or step.created_at)
    end = _utc(m.end_at if m else None)
    dur = m.duration_ms if m else None
    parent_id, parent_depth = _resolve_parent(step, run_spans, fallback)
    is_error = isinstance(step.content, str) and step.content.startswith("Error:")

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
    step: StepRecord,
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


class AgentTraceCollector:
    """Construct a Trace from explicit run and step callbacks."""

    PREVIEW_LENGTH = 500

    def __init__(self, store: BaseTraceStorage | None = None) -> None:
        self.store = store
        self._trace: Trace | None = None
        self._run_spans: dict[str, Span] = {}
        self._assistant_cache: dict[str, StepRecord] = {}

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
        self._assistant_cache = {}

    def on_run_started(self, run: Run) -> None:
        trace = self._require_trace()
        run.trace_id = trace.trace_id
        parent = self._run_spans.get(run.parent_run_id) if run.parent_run_id else None
        span = Span(
            trace_id=trace.trace_id,
            parent_span_id=parent.span_id if parent else None,
            kind=SpanKind.AGENT,
            name=run.agent_id,
            depth=(parent.depth + 1) if parent else 0,
            attributes={
                "agent_id": run.agent_id,
                "session_id": run.session_id,
                "nested": run.parent_run_id is not None,
                "parent_run_id": run.parent_run_id,
            },
            run_id=run.id,
        )
        if trace.root_span_id is None:
            trace.root_span_id = span.span_id
        trace.add_span(span)
        self._run_spans[run.id] = span

    async def on_step(
        self, step: StepRecord, llm: LLMCallContext | None = None
    ) -> None:
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
        await self._save_trace()

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
        self._assistant_cache = {}

    def _require_trace(self) -> Trace:
        if self._trace is None:
            raise RuntimeError("trace_not_started")
        return self._trace

    def _cache_tool_calls(self, step: StepRecord) -> None:
        if step.role.value != "assistant" or not step.tool_calls:
            return
        for tc in step.tool_calls:
            tid = tc.get("id")
            if tid:
                self._assistant_cache[tid] = step

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
