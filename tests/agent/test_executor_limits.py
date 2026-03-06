"""Tests for execution limit semantics in AgentExecutor."""

import time

import pytest

from agiwo.agent.execution_context import ExecutionContext, SessionSequenceCounter
from agiwo.agent.inner.event_emitter import EventEmitter
from agiwo.agent.inner.executor import AgentExecutor
from agiwo.agent.options import AgentOptions
from agiwo.agent.schema import StepRecord, TerminationReason
from agiwo.agent.stream_channel import StreamChannel
from agiwo.llm.base import Model, StreamChunk
from agiwo.tool.base import BaseTool, ToolResult


class MockModel(Model):
    """Mock model that returns predefined streaming chunks per call."""

    def __init__(self, chunks_sequence: list[list[StreamChunk]], **kwargs) -> None:
        super().__init__(id="mock", name="mock", temperature=0.7, **kwargs)
        self._chunks_sequence = chunks_sequence
        self._call_count = 0

    async def arun_stream(self, messages, tools=None):
        chunks = self._chunks_sequence[self._call_count]
        self._call_count += 1
        for chunk in chunks:
            yield chunk


class CountingTool(BaseTool):
    """Tool that records execution count."""

    def __init__(self) -> None:
        self.calls = 0
        super().__init__()

    def get_name(self) -> str:
        return "counting_tool"

    def get_description(self) -> str:
        return "Counting tool for tests"

    def get_parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    def is_concurrency_safe(self) -> bool:
        return True

    async def execute(self, parameters, context, abort_signal=None) -> ToolResult:
        self.calls += 1
        now = time.time()
        return ToolResult(
            tool_name=self.get_name(),
            tool_call_id=parameters.get("tool_call_id", ""),
            input_args={},
            content="ok",
            output=None,
            start_time=now,
            end_time=now,
            duration=0,
        )


def _make_context() -> ExecutionContext:
    return ExecutionContext(
        session_id="sess-limit",
        run_id="run-limit",
        channel=StreamChannel(),
        agent_id="agent-limit",
        agent_name="test-agent",
        sequence_counter=SessionSequenceCounter(0),
    )


def _usage(input_tokens: int, output_tokens: int) -> dict[str, int]:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
    }


@pytest.mark.asyncio
async def test_no_tool_call_over_limit_is_not_completed():
    model = MockModel(
        [
            [
                StreamChunk(content="hello"),
                StreamChunk(usage=_usage(80, 30), finish_reason="stop"),
            ]
        ]
    )
    executor = AgentExecutor(
        model=model,
        tools=[],
        emitter=EventEmitter(_make_context()),
        options=AgentOptions(max_context_window_tokens=100),
    )

    ctx = _make_context()
    user_step = StepRecord.user(ctx, sequence=1, user_input="test")
    output = await executor.execute(
        system_prompt="You are a test assistant.",
        user_step=user_step,
        context=ctx,
    )

    assert output.termination_reason == TerminationReason.MAX_CONTEXT_WINDOW_TOKENS
    assert model._call_count == 1


@pytest.mark.asyncio
async def test_execute_tools_first_then_stop_on_limit():
    tool = CountingTool()
    model = MockModel(
        [
            [
                StreamChunk(
                    tool_calls=[
                        {
                            "index": 0,
                            "id": "tc-1",
                            "type": "function",
                            "function": {"name": "counting_tool", "arguments": "{}"},
                        }
                    ]
                ),
                StreamChunk(usage=_usage(80, 30), finish_reason="tool_calls"),
            ]
        ]
    )
    executor = AgentExecutor(
        model=model,
        tools=[tool],
        emitter=EventEmitter(_make_context()),
        options=AgentOptions(max_context_window_tokens=100),
    )

    ctx = _make_context()
    user_step = StepRecord.user(ctx, sequence=1, user_input="run")
    output = await executor.execute(
        system_prompt="You are a test assistant.",
        user_step=user_step,
        context=ctx,
    )

    assert tool.calls == 1
    assert output.termination_reason == TerminationReason.MAX_CONTEXT_WINDOW_TOKENS
    assert model._call_count == 1


@pytest.mark.asyncio
async def test_model_output_limit_finish_reason_stops_without_summary():
    model = MockModel(
        [
            [
                StreamChunk(content="partial"),
                StreamChunk(usage=_usage(20, 50), finish_reason="length"),
            ]
        ]
    )
    executor = AgentExecutor(
        model=model,
        tools=[],
        emitter=EventEmitter(_make_context()),
        options=AgentOptions(),
    )

    ctx = _make_context()
    user_step = StepRecord.user(ctx, sequence=1, user_input="run")
    output = await executor.execute(
        system_prompt="You are a test assistant.",
        user_step=user_step,
        context=ctx,
    )

    assert output.termination_reason == TerminationReason.MAX_OUTPUT_TOKENS_PER_CALL
    assert model._call_count == 1


@pytest.mark.asyncio
async def test_max_tokens_per_run_limit():
    tool = CountingTool()
    model = MockModel(
        [
            [
                StreamChunk(
                    tool_calls=[
                        {
                            "index": 0,
                            "id": "tc-1",
                            "type": "function",
                            "function": {"name": "counting_tool", "arguments": "{}"},
                        }
                    ]
                ),
                StreamChunk(usage=_usage(30, 30), finish_reason="tool_calls"),
            ],
            [
                StreamChunk(content="second"),
                StreamChunk(usage=_usage(25, 25), finish_reason="stop"),
            ],
        ]
    )
    executor = AgentExecutor(
        model=model,
        tools=[tool],
        emitter=EventEmitter(_make_context()),
        options=AgentOptions(
            max_context_window_tokens=100000,
            max_tokens_per_run=100,
        ),
    )

    ctx = _make_context()
    user_step = StepRecord.user(ctx, sequence=1, user_input="run")
    output = await executor.execute(
        system_prompt="You are a test assistant.",
        user_step=user_step,
        context=ctx,
    )

    assert tool.calls == 1
    assert output.termination_reason == TerminationReason.MAX_TOKENS_PER_RUN
    assert model._call_count == 2


@pytest.mark.asyncio
async def test_max_run_token_cost_limit():
    model = MockModel(
        [
            [
                StreamChunk(content="costy"),
                StreamChunk(usage=_usage(30, 30), finish_reason="stop"),
            ]
        ],
        input_price=1.0,
        output_price=1.0,
        cache_hit_price=0.0,
    )
    executor = AgentExecutor(
        model=model,
        tools=[],
        emitter=EventEmitter(_make_context()),
        options=AgentOptions(max_run_token_cost=0.00005),
    )

    ctx = _make_context()
    user_step = StepRecord.user(ctx, sequence=1, user_input="cost")
    output = await executor.execute(
        system_prompt="You are a test assistant.",
        user_step=user_step,
        context=ctx,
    )

    assert output.termination_reason == TerminationReason.MAX_RUN_TOKEN_COST
    assert output.metrics is not None
    assert output.metrics.token_cost > 0
    assert model._call_count == 1
