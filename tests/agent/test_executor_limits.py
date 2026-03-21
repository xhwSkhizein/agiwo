"""Tests for execution limit semantics in ExecutionEngine."""

import time

import pytest

from agiwo.agent import TerminationReason
from agiwo.agent.engine.context import AgentRunContext
from agiwo.agent.options import AgentOptions
from agiwo.llm.base import Model, StreamChunk
from agiwo.tool.base import BaseTool, ToolResult
from tests.utils.agent_context import build_agent_context
from tests.utils.execution_engine import _build_executor


class MockModel(Model):
    """Mock model that returns predefined streaming chunks per call."""

    def __init__(self, chunks_sequence: list[list[StreamChunk]], **kwargs) -> None:
        kwargs.setdefault("provider", "openai")
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


class BigSchemaTool(CountingTool):
    def get_name(self) -> str:
        return "big_schema_tool"

    def get_description(self) -> str:
        return "x" * 10000


def _make_context() -> AgentRunContext:
    return build_agent_context(
        session_id="sess-limit",
        run_id="run-limit",
        agent_id="agent-limit",
        agent_name="test-agent",
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
async def test_max_input_tokens_per_call_limit_hits_after_llm_call():
    model = MockModel(
        [
            [
                StreamChunk(content="response"),
                StreamChunk(usage=_usage(100, 10), finish_reason="stop"),
            ]
        ]
    )
    ctx = _make_context()
    executor = _build_executor(
        model=model,
        tools=[],
        context=ctx,
        options=AgentOptions(max_input_tokens_per_call=1),
    )
    output = await executor.execute("this should exceed one token", context=ctx)

    assert output.termination_reason == TerminationReason.MAX_INPUT_TOKENS_PER_CALL
    assert model._call_count == 1


@pytest.mark.asyncio
async def test_input_limit_counts_tool_schema_tokens():
    model = MockModel(
        [
            [
                StreamChunk(content="response"),
                StreamChunk(usage=_usage(5000, 10), finish_reason="stop"),
            ]
        ]
    )
    tool = BigSchemaTool()
    ctx = _make_context()
    executor = _build_executor(
        model=model,
        tools=[tool],
        context=ctx,
        options=AgentOptions(max_input_tokens_per_call=100),
    )
    output = await executor.execute("x", context=ctx)

    assert output.termination_reason == TerminationReason.MAX_INPUT_TOKENS_PER_CALL
    assert model._call_count == 1
    assert tool.calls == 0


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
    ctx = _make_context()
    executor = _build_executor(
        model=model,
        tools=[],
        context=ctx,
    )
    output = await executor.execute("run", context=ctx)

    assert output.termination_reason == TerminationReason.MAX_OUTPUT_TOKENS
    assert model._call_count == 1


@pytest.mark.asyncio
async def test_max_run_cost_limit_checks_before_tool_execution():
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
                StreamChunk(content="summary"),
                StreamChunk(finish_reason="stop"),
            ],
        ],
        input_price=1.0,
        output_price=1.0,
        cache_hit_price=0.0,
    )
    ctx = _make_context()
    executor = _build_executor(
        model=model,
        tools=[tool],
        context=ctx,
        options=AgentOptions(max_run_cost=0.00005),
    )
    output = await executor.execute("run", context=ctx)

    assert output.termination_reason == TerminationReason.MAX_RUN_COST
    assert model._call_count == 2
    assert tool.calls == 0


@pytest.mark.asyncio
async def test_max_run_cost_limit_uses_tiktoken_fallback_when_usage_missing():
    model = MockModel(
        [
            [
                StreamChunk(content="fallback cost"),
                StreamChunk(finish_reason="stop"),
            ],
            [
                StreamChunk(content="summary"),
                StreamChunk(finish_reason="stop"),
            ],
        ],
        input_price=1.0,
        output_price=1.0,
        cache_hit_price=0.0,
    )
    ctx = _make_context()
    executor = _build_executor(
        model=model,
        tools=[],
        context=ctx,
        options=AgentOptions(max_run_cost=0.000001),
    )
    output = await executor.execute("run", context=ctx)

    assert output.termination_reason == TerminationReason.MAX_RUN_COST
    assert output.metrics is not None
    assert output.metrics.token_cost > 0


def test_default_max_input_tokens_per_call_uses_model_context_window():
    model = MockModel([[]], max_context_window=20000, max_output_tokens=1000)
    context = _make_context()
    executor = _build_executor(model=model, tools=[], context=context)

    assert executor.max_input_tokens_per_call == 19000


def test_default_max_input_tokens_per_call_has_minimum_floor():
    model = MockModel([[]], max_context_window=2000, max_output_tokens=5000)
    context = _make_context()
    with pytest.raises(
        ValueError,
        match="max_context_window must be greater than max_output_tokens",
    ):
        _build_executor(model=model, tools=[], context=context)


@pytest.mark.asyncio
async def test_summary_still_runs_when_max_run_cost_is_zero():
    model = MockModel(
        [
            [
                StreamChunk(content="summary content"),
                StreamChunk(finish_reason="stop"),
            ]
        ],
        input_price=1.0,
        output_price=1.0,
    )
    ctx = _make_context()
    executor = _build_executor(
        model=model,
        tools=[],
        context=ctx,
        options=AgentOptions(max_steps=0, max_run_cost=0.0),
    )
    output = await executor.execute("run", context=ctx)

    assert output.termination_reason == TerminationReason.MAX_STEPS
    assert model._call_count == 1
    assert output.metrics is not None
    assert output.metrics.token_cost > 0


@pytest.mark.asyncio
async def test_max_run_cost_still_generates_summary():
    model = MockModel(
        [
            [
                StreamChunk(content="costly"),
                StreamChunk(usage=_usage(30, 30), finish_reason="stop"),
            ],
            [
                StreamChunk(content="summary"),
                StreamChunk(finish_reason="stop"),
            ],
        ],
        input_price=1.0,
        output_price=1.0,
        cache_hit_price=0.0,
    )
    ctx = _make_context()
    executor = _build_executor(
        model=model,
        tools=[],
        context=ctx,
        options=AgentOptions(max_run_cost=0.00005),
    )
    output = await executor.execute("run", context=ctx)

    assert output.termination_reason == TerminationReason.MAX_RUN_COST
    assert model._call_count == 2
