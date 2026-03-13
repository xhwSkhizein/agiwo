from agiwo.agent import StepMetrics, StepRecord
from agiwo.agent.execution_context import ExecutionContext, SessionSequenceCounter
from agiwo.agent.stream_channel import StreamChannel
from agiwo.llm.base import Model
from agiwo.llm.usage_resolver import StepMetricsResolver


class MockModel(Model):
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("provider", "openai")
        super().__init__(**kwargs)

    async def arun_stream(self, messages, tools=None):
        if False:
            yield None


def _make_context() -> ExecutionContext:
    return ExecutionContext(
        session_id="sess-cost",
        run_id="run-cost",
        channel=StreamChannel(),
        agent_id="agent-cost",
        agent_name="test-agent",
        sequence_counter=SessionSequenceCounter(0),
    )


def test_metrics_resolver_backfills_missing_fields() -> None:
    model = MockModel(id="mock", name="mock")
    resolver = StepMetricsResolver(model)
    step = StepRecord.assistant(
        _make_context(),
        sequence=1,
        content="hello world",
        reasoning_content="because",
        tool_calls=[
            {
                "id": "tc-1",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
        ],
        metrics=StepMetrics(),
    )

    request_estimate = resolver.estimate_request([{"role": "user", "content": "x"}], None)
    resolver.resolve(step, request_estimate)

    assert step.metrics is not None
    assert step.metrics.input_tokens is not None
    assert step.metrics.output_tokens is not None
    assert step.metrics.output_tokens > 0
    assert step.metrics.total_tokens == step.metrics.input_tokens + step.metrics.output_tokens
    assert step.metrics.cache_read_tokens == 0
    assert step.metrics.cache_creation_tokens == 0
    assert step.metrics.usage_source == "estimated"
    assert step.metrics.token_cost is not None


def test_metrics_resolver_computes_cost() -> None:
    model = MockModel(id="mock", name="mock", input_price=3.0, output_price=15.0)
    resolver = StepMetricsResolver(model)
    step = StepRecord.assistant(
        _make_context(),
        sequence=1,
        content="hello",
        metrics=StepMetrics(
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            cache_read_tokens=0,
            cache_creation_tokens=0,
        ),
    )

    resolver.resolve(step, None)

    assert step.metrics is not None
    assert step.metrics.token_cost is not None
    expected = (1000 * 3.0 + 500 * 15.0) / 1_000_000.0
    assert abs(step.metrics.token_cost - expected) < 1e-10
