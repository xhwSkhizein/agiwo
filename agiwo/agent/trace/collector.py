"""Agent trace collector with direct run/step callbacks."""

from uuid import uuid4

from agiwo.agent.runtime import LLMCallContext, Run, RunOutput, StepRecord
from agiwo.agent.trace.span_builder import (
    build_assistant_step_span,
    build_tool_step_span,
)
from agiwo.observability.base import BaseTraceStorage
from agiwo.observability.trace import Span, SpanKind, SpanStatus, Trace
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class AgentTraceCollector:
    """Construct a Trace from explicit run and step callbacks."""

    PREVIEW_LENGTH = 500

    def __init__(self, store: BaseTraceStorage | None = None) -> None:
        self.store = store
        self._trace: Trace | None = None
        self._run_spans: dict[str, Span] = {}
        self._assistant_step_cache: dict[str, StepRecord] = {}

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
        self._assistant_step_cache = {}

    def on_run_started(self, run: Run) -> None:
        trace = self._require_trace()
        run.trace_id = trace.trace_id
        parent_span = self._resolve_parent_run_span(run.parent_run_id)
        span = Span(
            trace_id=trace.trace_id,
            parent_span_id=parent_span.span_id if parent_span else None,
            kind=SpanKind.AGENT,
            name=run.agent_id,
            depth=(parent_span.depth + 1) if parent_span else 0,
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
        self,
        step: StepRecord,
        llm: LLMCallContext | None = None,
    ) -> None:
        trace = self._require_trace()
        self._cache_assistant_tool_calls(step)
        parent_run_span = self._resolve_step_parent(step)
        if step.role.value == "assistant":
            span = build_assistant_step_span(
                trace_id=trace.trace_id,
                step=step,
                llm=llm,
                span_stack=self._run_spans,
                current_span=parent_run_span,
                preview_length=self.PREVIEW_LENGTH,
            )
            trace.add_span(span)
        elif step.role.value == "tool":
            span = build_tool_step_span(
                trace_id=trace.trace_id,
                step=step,
                assistant_step_cache=self._assistant_step_cache,
                span_stack=self._run_spans,
                current_span=parent_run_span,
            )
            trace.add_span(span)
        await self._save_trace()

    async def on_run_completed(
        self,
        output: RunOutput,
        *,
        run_id: str,
    ) -> None:
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

    async def on_run_failed(
        self,
        error: Exception,
        *,
        run_id: str,
    ) -> None:
        trace = self._require_trace()
        span = self._run_spans.get(run_id)
        if span is not None:
            span.complete(status=SpanStatus.ERROR, error_message=str(error))
        if span is not None and span.parent_span_id is None:
            trace.complete(status=SpanStatus.ERROR)
        await self._save_trace()

    async def stop(self) -> None:
        trace = self._trace
        if trace is None:
            return
        if trace.end_time is None:
            trace.complete(status=trace.status)
        await self._save_trace()
        self._trace = None
        self._run_spans = {}
        self._assistant_step_cache = {}

    def _require_trace(self) -> Trace:
        if self._trace is None:
            raise RuntimeError("trace_not_started")
        return self._trace

    def _resolve_parent_run_span(self, parent_run_id: str | None) -> Span | None:
        if parent_run_id is None:
            return None
        return self._run_spans.get(parent_run_id)

    def _resolve_step_parent(self, step: StepRecord) -> Span | None:
        parent = self._run_spans.get(step.run_id)
        if parent is not None:
            return parent
        if step.parent_run_id is not None:
            return self._run_spans.get(step.parent_run_id)
        return None

    def _cache_assistant_tool_calls(self, step: StepRecord) -> None:
        if step.role.value != "assistant" or not step.tool_calls:
            return
        for tool_call in step.tool_calls:
            tool_call_id = tool_call.get("id")
            if tool_call_id:
                self._assistant_step_cache[tool_call_id] = step

    async def _save_trace(self) -> None:
        if self._trace is None or self.store is None:
            return
        try:
            await self.store.save_trace(self._trace)
        except Exception as error:  # noqa: BLE001 - observability boundary
            logger.warning(
                "trace_save_failed",
                trace_id=self._trace.trace_id,
                error=str(error),
            )


__all__ = ["AgentTraceCollector"]
