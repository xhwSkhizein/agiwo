"""Tests for ToolResult.termination_reason propagation in AgentExecutor."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from agiwo.agent.execution_context import ExecutionContext, SessionSequenceCounter
from agiwo.agent.hooks import AgentHooks
from agiwo.agent.inner.event_emitter import EventEmitter
from agiwo.agent.inner.executor import AgentExecutor
from agiwo.agent.options import AgentOptions
from agiwo.agent.schema import TerminationReason
from agiwo.agent.stream_channel import StreamChannel
from agiwo.llm.base import Model, StreamChunk
from agiwo.tool.base import BaseTool, ToolResult


class MockModel(Model):
    """Mock model that returns predefined responses."""

    def __init__(self, chunks_sequence: list[list[StreamChunk]]) -> None:
        super().__init__(id="mock", name="mock", temperature=0.7)
        self._chunks_sequence = chunks_sequence
        self._call_count = 0

    async def arun_stream(self, messages, tools=None):
        chunks = self._chunks_sequence[self._call_count]
        self._call_count += 1
        for chunk in chunks:
            yield chunk


class MockTerminatingTool(BaseTool):
    """Tool that returns a termination_reason."""

    def __init__(self, reason: TerminationReason = TerminationReason.SLEEPING) -> None:
        self._reason = reason
        super().__init__()

    def get_name(self) -> str:
        return "terminating_tool"

    def get_description(self) -> str:
        return "A tool that terminates execution"

    def get_parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    def is_concurrency_safe(self) -> bool:
        return True

    async def execute(self, parameters, context, abort_signal=None) -> ToolResult:
        import time

        now = time.time()
        return ToolResult(
            tool_name=self.get_name(),
            tool_call_id=parameters.get("tool_call_id", ""),
            input_args={},
            content="Going to sleep",
            output=None,
            start_time=now,
            end_time=now,
            duration=0,
            termination_reason=self._reason,
        )


class MockNormalTool(BaseTool):
    """Tool that returns normally."""

    def get_name(self) -> str:
        return "normal_tool"

    def get_description(self) -> str:
        return "A normal tool"

    def get_parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    def is_concurrency_safe(self) -> bool:
        return True

    async def execute(self, parameters, context, abort_signal=None) -> ToolResult:
        import time

        now = time.time()
        return ToolResult(
            tool_name=self.get_name(),
            tool_call_id=parameters.get("tool_call_id", ""),
            input_args={},
            content="Done",
            output=None,
            start_time=now,
            end_time=now,
            duration=0,
        )


def _make_context() -> ExecutionContext:
    return ExecutionContext(
        session_id="sess-1",
        run_id="run-1",
        channel=StreamChannel(),
        agent_id="test-agent",
        sequence_counter=SessionSequenceCounter(0),
    )


class TestExecutorTermination:
    @pytest.mark.asyncio
    async def test_termination_reason_propagated(self):
        """When a tool returns termination_reason, executor should stop and set it."""
        tool = MockTerminatingTool(TerminationReason.SLEEPING)

        # Model returns a tool call, then (if called again) normal completion
        chunks_with_tool_call = [
            StreamChunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "tc-1",
                        "function": {"name": "terminating_tool", "arguments": "{}"},
                    }
                ]
            ),
            StreamChunk(finish_reason="tool_calls"),
        ]
        model = MockModel([chunks_with_tool_call])

        context = _make_context()
        emitter = EventEmitter(context)
        executor = AgentExecutor(
            model=model,
            tools=[tool],
            emitter=emitter,
            options=AgentOptions(max_steps=10),
        )

        output = await executor.execute(
            messages=[{"role": "user", "content": "test"}],
            context=context,
        )

        assert output.termination_reason == TerminationReason.SLEEPING
        # Model should have been called only once (no second LLM call after termination)
        assert model._call_count == 1

    @pytest.mark.asyncio
    async def test_normal_tool_no_termination(self):
        """When tool returns no termination_reason, executor continues normally."""
        tool = MockNormalTool()

        chunks_with_tool_call = [
            StreamChunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "tc-1",
                        "function": {"name": "normal_tool", "arguments": "{}"},
                    }
                ]
            ),
            StreamChunk(finish_reason="tool_calls"),
        ]
        chunks_completion = [
            StreamChunk(content="Final answer"),
            StreamChunk(finish_reason="stop"),
        ]
        model = MockModel([chunks_with_tool_call, chunks_completion])

        context = _make_context()
        emitter = EventEmitter(context)
        executor = AgentExecutor(
            model=model,
            tools=[tool],
            emitter=emitter,
            options=AgentOptions(max_steps=10),
        )

        output = await executor.execute(
            messages=[{"role": "user", "content": "test"}],
            context=context,
        )

        assert output.termination_reason == TerminationReason.COMPLETED
        assert model._call_count == 2  # tool call + completion
