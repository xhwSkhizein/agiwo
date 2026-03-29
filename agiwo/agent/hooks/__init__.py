"""
Agent lifecycle hooks for extensibility.

Hooks allow SDK users to inject custom behavior at key execution points
without subclassing the Agent. All hooks are optional and async.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from agiwo.agent.models.input import UserInput
from agiwo.agent.models.memory import MemoryRecord
from agiwo.agent.models.run import RunOutput
from agiwo.agent.models.step import StepRecord
from agiwo.tool.base import ToolResult

if TYPE_CHECKING:
    from agiwo.agent.runtime.context import RunContext


BeforeRunHook = Callable[[UserInput, "RunContext"], Awaitable[str | None]]
AfterRunHook = Callable[[RunOutput, "RunContext"], Awaitable[None]]
BeforeToolCallHook = Callable[
    [str, str, dict[str, Any]], Awaitable[dict[str, Any] | None]
]
AfterToolCallHook = Callable[[str, str, dict[str, Any], ToolResult], Awaitable[None]]
BeforeLLMCallHook = Callable[[list[dict]], Awaitable[list[dict] | None]]
AfterLLMCallHook = Callable[[StepRecord], Awaitable[None]]
OnStepHook = Callable[[StepRecord], Awaitable[None]]

MemoryWriteHook = Callable[[UserInput, RunOutput, "RunContext"], Awaitable[None]]
MemoryRetrieveHook = Callable[[UserInput, "RunContext"], Awaitable[list[MemoryRecord]]]
OnCompactionFailed = Callable[[str, str, int], Awaitable[None]]


@dataclass
class AgentHooks:
    """Lifecycle hooks for Agent execution."""

    on_before_run: BeforeRunHook | None = None
    on_after_run: AfterRunHook | None = None
    on_before_tool_call: BeforeToolCallHook | None = None
    on_after_tool_call: AfterToolCallHook | None = None
    on_before_llm_call: BeforeLLMCallHook | None = None
    on_after_llm_call: AfterLLMCallHook | None = None
    on_step: OnStepHook | None = None
    on_memory_write: MemoryWriteHook | None = None
    on_memory_retrieve: MemoryRetrieveHook | None = None
    on_compaction_failed: OnCompactionFailed | None = None


__all__ = [
    "AfterLLMCallHook",
    "AfterRunHook",
    "AfterToolCallHook",
    "AgentHooks",
    "BeforeLLMCallHook",
    "BeforeRunHook",
    "BeforeToolCallHook",
    "MemoryRetrieveHook",
    "MemoryWriteHook",
    "OnCompactionFailed",
    "OnStepHook",
]
