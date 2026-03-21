"""Tests for runtime termination propagation in ExecutionEngine."""

import time

import pytest

from agiwo.agent import AgentHooks, TerminationReason
from agiwo.agent.engine.context import AgentRunContext
from agiwo.agent.engine.engine import ExecutionEngine
from agiwo.agent.lifecycle.definition import ResolvedExecutionDefinition
from agiwo.agent.options import AgentOptions
from agiwo.agent.runtime_tools import RuntimeToolOutcome
from agiwo.llm.base import Model, StreamChunk
from agiwo.tool.base import ToolDefinition, ToolResult
from tests.utils.agent_context import build_agent_context


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


class MockTerminatingTool:
    """Runtime tool that returns a termination reason."""

    cacheable = False
    timeout_seconds = 30

    def __init__(self, reason: TerminationReason = TerminationReason.SLEEPING) -> None:
        self._reason = reason

    def get_name(self) -> str:
        return "terminating_tool"

    def get_description(self) -> str:
        return "A tool that terminates execution"

    def get_parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    def is_concurrency_safe(self) -> bool:
        return True

    def get_short_description(self) -> str:
        return self.get_description()

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.get_name(),
            description=self.get_description(),
            parameters=self.get_parameters(),
            is_concurrency_safe=self.is_concurrency_safe(),
        )

    async def execute_for_agent(
        self, parameters, context, abort_signal=None
    ) -> RuntimeToolOutcome:
        now = time.time()
        return RuntimeToolOutcome(
            result=ToolResult.success(
                tool_name=self.get_name(),
                tool_call_id=parameters.get("tool_call_id", ""),
                input_args={},
                content="Going to sleep",
                output=None,
                start_time=now,
            ),
            termination_reason=self._reason,
        )


class MockNormalTool:
    """Runtime tool that returns normally."""

    cacheable = False
    timeout_seconds = 30

    def get_name(self) -> str:
        return "normal_tool"

    def get_description(self) -> str:
        return "A normal tool"

    def get_parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    def is_concurrency_safe(self) -> bool:
        return True

    def get_short_description(self) -> str:
        return self.get_description()

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.get_name(),
            description=self.get_description(),
            parameters=self.get_parameters(),
            is_concurrency_safe=self.is_concurrency_safe(),
        )

    async def execute_for_agent(
        self, parameters, context, abort_signal=None
    ) -> RuntimeToolOutcome:
        now = time.time()
        return RuntimeToolOutcome(
            result=ToolResult.success(
                tool_name=self.get_name(),
                tool_call_id=parameters.get("tool_call_id", ""),
                input_args={},
                content="Done",
                output=None,
                start_time=now,
            )
        )


def _make_context() -> AgentRunContext:
    return build_agent_context(
        session_id="sess-1",
        run_id="run-1",
        agent_id="test-agent",
        agent_name="test-agent",
    )


def _build_executor(
    *,
    model: Model,
    tools: list[object],
    context: AgentRunContext,
    options: AgentOptions | None = None,
) -> ExecutionEngine:
    definition = ResolvedExecutionDefinition(
        agent_id=context.agent_id,
        agent_name=context.agent_name,
        description="test-agent",
        model=model,
        hooks=AgentHooks(),
        options=options or AgentOptions(),
        tools=tuple(tools),
        system_prompt="You are a test assistant.",
    )
    return ExecutionEngine(
        definition=definition,
        step_observers=(),
        root_path="/tmp",
    )


class TestExecutorTermination:
    @pytest.mark.asyncio
    async def test_termination_reason_propagated(self):
        """When a runtime tool returns termination_reason, executor should stop and set it."""
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
        executor = _build_executor(
            model=model,
            tools=[tool],
            context=context,
            options=AgentOptions(max_steps=10),
        )
        output = await executor.execute("test", context=context)

        assert output.termination_reason == TerminationReason.SLEEPING
        # Model should have been called only once (no second LLM call after termination)
        assert model._call_count == 1

    @pytest.mark.asyncio
    async def test_normal_tool_no_termination(self):
        """When runtime tool returns no termination_reason, executor continues normally."""
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
        executor = _build_executor(
            model=model,
            tools=[tool],
            context=context,
            options=AgentOptions(max_steps=10),
        )
        output = await executor.execute("test", context=context)

        assert output.termination_reason == TerminationReason.COMPLETED
        assert model._call_count == 2  # tool call + completion
