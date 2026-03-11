"""
Trace collector - builds Trace from StreamEvent stream.

Uses middleware pattern to wrap event streams without modifying core execution logic.
"""

import asyncio
from typing import AsyncIterator
from uuid import uuid4

from agiwo.agent.schema import EventType, StepRecord, StreamEvent
from agiwo.observability.base import BaseTraceStorage
from agiwo.observability.span_builder import (
    build_assistant_step_span,
    build_tool_step_span,
)
from agiwo.observability.trace import Span, SpanKind, SpanStatus, Trace
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class TraceCollector:
    """
    Trace collector - constructs Trace from StreamEvent stream.

    Usage:
        collector = TraceCollector(store)

        async for event in collector.wrap_stream(event_stream):
            yield event  # Events are passed through while building trace
    """

    PREVIEW_LENGTH = 500  # Input/output preview length

    def __init__(self, store: BaseTraceStorage | None = None) -> None:
        """
        Initialize collector.

        Args:
            store: Optional TraceStorage for persistence
        """
        self.store = store

        # Internal state for push mode
        self._trace: Trace | None = None
        self._span_stack: dict[str, Span] = {}
        self._current_span: Span | None = None
        self._assistant_step_cache: dict[str, StepRecord] = {}

    def start(
        self,
        trace_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        input_query: str | None = None,
    ) -> None:
        """Initialize trace collection state."""
        self._trace = Trace(
            trace_id=trace_id or str(uuid4()),
            agent_id=agent_id,
            session_id=session_id,
            user_id=user_id,
            input_query=input_query,
        )
        self._span_stack = {}
        self._current_span = None
        self._assistant_step_cache = {}

    async def collect(self, event: StreamEvent) -> None:
        """Process a single event in push mode."""
        if not self._trace:
            # If start() wasn't called, we can't collect.
            # Ideally should raise error or auto-start, but safe fallback is log/ignore?
            # For strictness, let's assume start() must be called.
            return

        current_span, updated_cache = self._process_event(
            event,
            self._trace,
            self._span_stack,
            self._current_span,
            self._assistant_step_cache,
        )

        self._current_span = current_span
        if updated_cache:
            self._assistant_step_cache.update(updated_cache)

        # Incremental save at critical checkpoints
        should_save = event.type in {
            EventType.RUN_STARTED,
            EventType.STEP_COMPLETED,
            EventType.RUN_COMPLETED,
            EventType.RUN_FAILED,
        }

        if should_save and self.store:
            # Save asynchronously in background to avoid blocking
            asyncio.create_task(self._save_trace_safe())

        # Inject trace fields into event
        event.trace_id = self._trace.trace_id
        if current_span:
            event.span_id = current_span.span_id
            event.parent_span_id = current_span.parent_span_id

    async def stop(self) -> None:
        """Finalize and save trace."""
        if not self._trace:
            return

        trace = self._trace

        # Cleanup
        if not trace.end_time:
            # If active span exists, mark as error or OK?
            # wrap_stream defaulted to OK if not explicitly set.
            status = SpanStatus.OK
            if self._current_span and self._current_span.status == SpanStatus.UNSET:
                # If we are stopping abruptly, maybe we should respect current state?
                pass
            trace.complete(status=status)

        # Final save to ensure complete state
        # Note: Trace is also saved incrementally during execution (see collect())
        # This final save ensures the complete end state and OTLP export
        if self.store:
            try:
                await self.store.save_trace(trace)
            except Exception as e:  # noqa: BLE001 - observability boundary
                logger.error("trace_save_failed", trace_id=trace.trace_id, error=str(e))

        # Reset state
        self._trace = None
        self._span_stack = {}
        self._current_span = None
        self._assistant_step_cache = {}

    def fail(self, error: Exception) -> None:
        """Mark trace as failed."""
        if not self._trace:
            return

        self._trace.complete(status=SpanStatus.ERROR)
        if self._current_span:
            self._current_span.complete(
                status=SpanStatus.ERROR,
                error_message=str(error),
            )
        logger.error(
            "trace_collection_failed", trace_id=self._trace.trace_id, error=str(error)
        )

    async def _save_trace_safe(self) -> None:
        """Save trace to store with error handling (non-blocking)."""
        if not self._trace or not self.store:
            return

        try:
            await self.store.save_trace(self._trace)
        except Exception as e:  # noqa: BLE001 - observability boundary
            # Log but don't raise - incremental saves should not break execution
            logger.warning(
                "trace_incremental_save_failed",
                trace_id=self._trace.trace_id,
                error=str(e),
            )

    async def wrap_stream(
        self,
        event_stream: AsyncIterator[StreamEvent],
        trace_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        input_query: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Wrap event stream to automatically collect trace information.

        Args:
            event_stream: Original event stream
            trace_id: Optional trace ID
            agent_id: Agent ID
            session_id: Session ID
            user_id: User ID
            input_query: User input

        Yields:
            StreamEvent: Original events with injected trace_id/span_id
        """
        self.start(
            trace_id=trace_id,
            agent_id=agent_id,
            session_id=session_id,
            user_id=user_id,
            input_query=input_query,
        )

        try:
            async for event in event_stream:
                await self.collect(event)
                yield event

        except Exception as e:
            self.fail(e)
            raise
        finally:
            await self.stop()

    def _process_event(
        self,
        event: StreamEvent,
        trace: Trace,
        span_stack: dict[str, Span],
        current_span: Span | None,
        assistant_step_cache: dict[str, StepRecord],
    ) -> tuple[Span | None, dict[str, StepRecord]]:
        """
        Process single event, return currently active span and updated assistant_step_cache.

        Returns:
            tuple[Span | None, dict[str, StepRecord]]: (current_span, updated_cache)
        """

        if event.type == EventType.RUN_STARTED:
            return self._handle_run_started(event, trace, span_stack, current_span)
        if event.type == EventType.STEP_COMPLETED:
            return self._handle_step_completed(
                event,
                trace,
                span_stack,
                current_span,
                assistant_step_cache,
            )
        if event.type == EventType.RUN_COMPLETED:
            return self._handle_run_completed(event, trace, span_stack)
        if event.type == EventType.RUN_FAILED:
            return self._handle_run_failed(event, span_stack)
        return current_span, {}

    def _handle_run_started(
        self,
        event: StreamEvent,
        trace: Trace,
        span_stack: dict[str, Span],
        current_span: Span | None,
    ) -> tuple[Span | None, dict[str, StepRecord]]:
        data = event.data or {}
        agent_name = event.agent_id or data.get("agent_id") or "agent"

        if event.parent_run_id is not None:
            parent_span = self._find_parent_span_for_nested(
                event,
                span_stack,
                current_span,
            )
            span = Span(
                trace_id=trace.trace_id,
                parent_span_id=parent_span.span_id if parent_span else None,
                kind=SpanKind.AGENT,
                name=agent_name,
                depth=(parent_span.depth + 1) if parent_span else 1,
                attributes={
                    "agent_id": agent_name,
                    "nested": True,
                    "parent_run_id": event.parent_run_id,
                    "session_id": data.get("session_id"),
                },
                run_id=event.run_id,
            )
            trace.add_span(span)
            span_stack[event.run_id] = span
            return span, {}

        parent = current_span
        span = Span(
            trace_id=trace.trace_id,
            parent_span_id=parent.span_id if parent else None,
            kind=SpanKind.AGENT,
            name=agent_name,
            depth=(parent.depth + 1) if parent else 0,
            attributes={
                "agent_id": agent_name,
                "session_id": data.get("session_id"),
            },
            run_id=event.run_id,
        )
        if not trace.root_span_id:
            trace.root_span_id = span.span_id

        trace.add_span(span)
        span_stack[event.run_id] = span
        return span, {}

    def _handle_step_completed(
        self,
        event: StreamEvent,
        trace: Trace,
        span_stack: dict[str, Span],
        current_span: Span | None,
        assistant_step_cache: dict[str, StepRecord],
    ) -> tuple[Span | None, dict[str, StepRecord]]:
        if not event.step:
            return current_span, {}

        step = event.step
        updated_cache = self._cache_assistant_tool_calls(step)

        if step.role.value == "tool":
            return self._handle_tool_step_completed(
                step,
                trace,
                span_stack,
                current_span,
                assistant_step_cache,
                updated_cache,
            )
        if step.role.value == "assistant":
            return self._handle_assistant_step_completed(
                step,
                event,
                trace,
                span_stack,
                current_span,
                updated_cache,
            )
        return current_span, updated_cache

    def _handle_tool_step_completed(
        self,
        step: StepRecord,
        trace: Trace,
        span_stack: dict[str, Span],
        current_span: Span | None,
        assistant_step_cache: dict[str, StepRecord],
        updated_cache: dict[str, StepRecord],
    ) -> tuple[Span | None, dict[str, StepRecord]]:
        span = build_tool_step_span(
            trace_id=trace.trace_id,
            step=step,
            assistant_step_cache=assistant_step_cache,
            span_stack=span_stack,
            current_span=current_span,
        )
        trace.add_span(span)
        return current_span, updated_cache

    def _handle_assistant_step_completed(
        self,
        step: StepRecord,
        event: StreamEvent,
        trace: Trace,
        span_stack: dict[str, Span],
        current_span: Span | None,
        updated_cache: dict[str, StepRecord],
    ) -> tuple[Span | None, dict[str, StepRecord]]:
        span = build_assistant_step_span(
            trace_id=trace.trace_id,
            step=step,
            llm=event.llm,
            span_stack=span_stack,
            current_span=current_span,
            preview_length=self.PREVIEW_LENGTH,
        )
        trace.add_span(span)
        return current_span, updated_cache

    def _handle_run_completed(
        self,
        event: StreamEvent,
        trace: Trace,
        span_stack: dict[str, Span],
    ) -> tuple[Span | None, dict[str, StepRecord]]:
        span = span_stack.get(event.run_id)
        if span:
            response = event.data.get("response") if event.data else None
            span.complete(
                status=SpanStatus.OK,
                output_preview=response[: self.PREVIEW_LENGTH]
                if response
                else None,
            )
            trace.final_output = response
        return span_stack.get(event.run_id), {}

    def _handle_run_failed(
        self,
        event: StreamEvent,
        span_stack: dict[str, Span],
    ) -> tuple[Span | None, dict[str, StepRecord]]:
        span = span_stack.get(event.run_id)
        if span:
            error = event.data.get("error") if event.data else "Unknown error"
            span.complete(status=SpanStatus.ERROR, error_message=error)
        return span_stack.get(event.run_id), {}

    def _cache_assistant_tool_calls(
        self,
        step: StepRecord,
    ) -> dict[str, StepRecord]:
        if step.role.value != "assistant" or not step.tool_calls:
            return {}

        updated_cache: dict[str, StepRecord] = {}
        for tool_call in step.tool_calls:
            tool_call_id = tool_call.get("id")
            if tool_call_id:
                updated_cache[tool_call_id] = step
        return updated_cache

    def _find_parent_span_for_nested(
        self,
        event: StreamEvent,
        span_stack: dict[str, Span],
        current_span: Span | None,
    ) -> Span | None:
        """
        Find parent span for nested execution.

        Strategy:
        1. Look for parent_run_id in span_stack (parent Agent Span)
        2. Look for current_span (might be a tool call span)
        3. Fallback to root span
        """
        # First, try to find parent Agent Span by parent_run_id
        if event.parent_run_id:
            parent_agent_span = span_stack.get(event.parent_run_id)
            if parent_agent_span:
                return parent_agent_span

        # Second, check if current_span is a tool call span (nested agent called via tool)
        if current_span and current_span.kind == SpanKind.TOOL_CALL:
            return current_span

        # Third, use current_span if available
        if current_span:
            return current_span

        # Fallback: find root span
        for span in span_stack.values():
            if span.depth == 0:
                return span

        return None

__all__ = ["TraceCollector"]
