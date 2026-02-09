"""
Agent lifecycle hooks for extensibility.

Hooks allow SDK users to inject custom behavior at key execution points
without subclassing the Agent. All hooks are optional and async.
"""

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from agiwo.agent.schema import UserInput
from agiwo.agent.execution_context import ExecutionContext
from agiwo.agent.schema import RunOutput, StepRecord, StreamEvent, MemoryRecord
from agiwo.tool.base import ToolResult


# Lifecycle hooks
BeforeRunHook = Callable[[UserInput, ExecutionContext], Awaitable[str | None]]
AfterRunHook = Callable[[RunOutput, ExecutionContext], Awaitable[None]]
BeforeToolCallHook = Callable[
    [str, str, dict[str, Any]], Awaitable[dict[str, Any] | None]
]
AfterToolCallHook = Callable[[str, str, dict[str, Any], ToolResult], Awaitable[None]]
BeforeLLMCallHook = Callable[[list[dict]], Awaitable[list[dict] | None]]
AfterLLMCallHook = Callable[[StepRecord], Awaitable[None]]
OnStepHook = Callable[[StepRecord], Awaitable[None]]
OnEventHook = Callable[[StreamEvent], Awaitable[None]]

# Memory hooks
MemoryWriteHook = Callable[[UserInput, RunOutput, ExecutionContext], Awaitable[None]]
MemoryRetrieveHook = Callable[
    [UserInput, ExecutionContext], Awaitable[list[MemoryRecord]]
]


@dataclass
class AgentHooks:
    """
    Lifecycle hooks for Agent execution.

    All hooks are optional. Provide only the ones you need.

    Lifecycle hooks:
        on_before_run: Called before execution starts. Receives (user_input, context).
            Return modified input or None to keep original.
        on_after_run: Called after execution completes. Receives (output, context).
        on_before_tool_call: Called before each tool execution. Receives (tc_id, tool_name, args).
            Return modified args or None to keep original.
        on_after_tool_call: Called after each tool execution. Receives (tc_id, tool_name, args, ToolResult).
        on_before_llm_call: Called before each LLM call. Receives messages list.
            Return modified messages or None to keep original.
        on_after_llm_call: Called after each LLM call. Receives the assistant StepRecord.
        on_step: Called when any step (user/assistant/tool) is committed.
        on_event: Called for every StreamEvent emitted.

    Memory hooks:
        on_memory_write: Called to persist information to external memory.
            Receives (query, response, context).
        on_memory_retrieve: Called to retrieve memories for context enrichment.
            Receives (user_input, context). Returns list of MemoryRecord.
    """

    # Lifecycle
    on_before_run: BeforeRunHook | None = None
    on_after_run: AfterRunHook | None = None
    on_before_tool_call: BeforeToolCallHook | None = None
    on_after_tool_call: AfterToolCallHook | None = None
    on_before_llm_call: BeforeLLMCallHook | None = None
    on_after_llm_call: AfterLLMCallHook | None = None
    on_step: OnStepHook | None = None
    on_event: OnEventHook | None = None

    # Memory
    on_memory_write: MemoryWriteHook | None = None
    on_memory_retrieve: MemoryRetrieveHook | None = None
