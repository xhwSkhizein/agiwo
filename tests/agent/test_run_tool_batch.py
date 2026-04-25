from dataclasses import dataclass, field

import pytest

import agiwo.agent.run_tool_batch as run_tool_batch_module
from agiwo.agent.models.config import AgentOptions
from agiwo.agent.models.log import build_committed_step_entry
from agiwo.agent.models.review import Milestone
from agiwo.agent.models.run import RunLedger, TerminationReason
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.storage.base import InMemoryRunLogStorage
from agiwo.tool.base import ToolResult
from agiwo.utils.abort_signal import AbortSignal


@dataclass
class _FakeHooks:
    review_advice: str | None = None
    before_review_calls: list[dict[str, object]] = field(default_factory=list)
    after_step_back_calls: list[object] = field(default_factory=list)

    async def after_tool_call(
        self,
        tool_call_id: str,
        tool_name: str,
        parameters: dict[str, object],
        result: object,
        context: object,
    ) -> None:
        del tool_call_id, tool_name, parameters, result, context

    async def before_review(
        self,
        *,
        trigger_reason: str,
        milestone: object | None,
        step_count: int,
        context: object | None = None,
    ) -> str | None:
        self.before_review_calls.append(
            {
                "trigger_reason": trigger_reason,
                "milestone": milestone,
                "step_count": step_count,
                "context": context,
            }
        )
        return self.review_advice

    async def after_step_back(
        self, outcome: object, context: object | None = None
    ) -> None:
        del context
        self.after_step_back_calls.append(outcome)


@dataclass
class _FakeContext:
    config: AgentOptions
    ledger: RunLedger
    hooks: _FakeHooks
    session_runtime: SessionRuntime
    session_id: str = "sess-1"
    run_id: str = "run-1"
    agent_id: str = "agent-1"
    parent_run_id: str | None = None
    depth: int = 0
    is_terminal: bool = False


@dataclass
class _FakeRuntime:
    tools_map: dict[str, object]
    abort_signal: AbortSignal = field(default_factory=AbortSignal)


def _review_tools_map() -> dict[str, object]:
    return {
        "review_trajectory": object(),
        "declare_milestones": object(),
    }


async def _commit_step_to_ledger_and_storage(
    context: _FakeContext,
    step,
):
    context.ledger.messages.append(step.to_message())
    await context.session_runtime.append_run_log_entries(
        [build_committed_step_entry(step)]
    )
    return step


@pytest.mark.asyncio
async def test_execute_tool_batch_cycle_injects_hook_review_advice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_execute_tool_batch(*args, **kwargs):
        del args, kwargs
        return [
            ToolResult.success(
                tool_name="search",
                tool_call_id="tc_search",
                content="Found results",
                output={},
            )
        ]

    monkeypatch.setattr(
        run_tool_batch_module, "execute_tool_batch", fake_execute_tool_batch
    )

    storage = InMemoryRunLogStorage()
    session_runtime = SessionRuntime(session_id="sess-1", run_log_storage=storage)
    hooks = _FakeHooks(review_advice="Focus on auth.py before broadening the search.")
    ledger = RunLedger()
    ledger.review.milestones = [
        Milestone(id="locate", description="Locate the auth bug", status="active")
    ]
    context = _FakeContext(
        config=AgentOptions(
            enable_goal_directed_review=True,
            review_step_interval=1,
        ),
        ledger=ledger,
        hooks=hooks,
        session_runtime=session_runtime,
    )
    runtime = _FakeRuntime(tools_map=_review_tools_map())
    committed_steps = []

    async def commit_step(step):
        committed_steps.append(step)
        return await _commit_step_to_ledger_and_storage(context, step)

    async def set_termination_reason(reason: TerminationReason, tool_name: str) -> None:
        del reason, tool_name

    terminated = await run_tool_batch_module.execute_tool_batch_cycle(
        context=context,
        runtime=runtime,
        tool_calls=[
            {"id": "tc_search", "type": "function", "function": {"name": "search"}}
        ],
        assistant_step_id="assistant-step",
        set_termination_reason=set_termination_reason,
        commit_step=commit_step,
    )

    assert terminated is False
    assert len(committed_steps) == 1
    assert "<system-review>" in committed_steps[0].content
    assert (
        "Hook advice: Focus on auth.py before broadening the search."
        in committed_steps[0].content
    )
    assert hooks.before_review_calls[0]["trigger_reason"] == "step_interval"
    assert hooks.before_review_calls[0]["step_count"] == 1


@pytest.mark.asyncio
async def test_execute_tool_batch_cycle_injects_review_without_hook_advice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_execute_tool_batch(*args, **kwargs):
        del args, kwargs
        return [
            ToolResult.success(
                tool_name="search",
                tool_call_id="tc_search",
                content="Found results",
                output={},
            )
        ]

    monkeypatch.setattr(
        run_tool_batch_module, "execute_tool_batch", fake_execute_tool_batch
    )

    storage = InMemoryRunLogStorage()
    session_runtime = SessionRuntime(session_id="sess-1", run_log_storage=storage)
    hooks = _FakeHooks(review_advice=None)
    ledger = RunLedger()
    ledger.review.milestones = [
        Milestone(id="locate", description="Locate the auth bug", status="active")
    ]
    context = _FakeContext(
        config=AgentOptions(
            enable_goal_directed_review=True,
            review_step_interval=1,
        ),
        ledger=ledger,
        hooks=hooks,
        session_runtime=session_runtime,
    )
    runtime = _FakeRuntime(tools_map=_review_tools_map())

    async def commit_step(step):
        return await _commit_step_to_ledger_and_storage(context, step)

    async def set_termination_reason(reason: TerminationReason, tool_name: str) -> None:
        del reason, tool_name

    await run_tool_batch_module.execute_tool_batch_cycle(
        context=context,
        runtime=runtime,
        tool_calls=[
            {"id": "tc_search", "type": "function", "function": {"name": "search"}}
        ],
        assistant_step_id="assistant-step",
        set_termination_reason=set_termination_reason,
        commit_step=commit_step,
    )

    tool_messages = [
        msg for msg in context.ledger.messages if msg.get("role") == "tool"
    ]
    assert len(tool_messages) == 1
    assert "<system-review>" in tool_messages[0]["content"]
    assert "Hook advice:" not in tool_messages[0]["content"]


@pytest.mark.asyncio
async def test_execute_tool_batch_cycle_calls_after_step_back_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_execute_tool_batch(*args, **kwargs):
        del args, kwargs
        return [
            ToolResult.success(
                tool_name="search",
                tool_call_id="tc_search",
                content="Verbose search output",
                output={},
            ),
            ToolResult.success(
                tool_name="review_trajectory",
                tool_call_id="tc_review",
                content="Trajectory review: aligned=False. narrow the search",
                output={"aligned": False, "experience": "narrow the search"},
            ),
        ]

    monkeypatch.setattr(
        run_tool_batch_module, "execute_tool_batch", fake_execute_tool_batch
    )

    storage = InMemoryRunLogStorage()
    session_runtime = SessionRuntime(session_id="sess-1", run_log_storage=storage)
    hooks = _FakeHooks()
    context = _FakeContext(
        config=AgentOptions(
            enable_goal_directed_review=True,
            review_step_interval=100,
        ),
        ledger=RunLedger(),
        hooks=hooks,
        session_runtime=session_runtime,
    )
    runtime = _FakeRuntime(tools_map=_review_tools_map())

    async def commit_step(step):
        return await _commit_step_to_ledger_and_storage(context, step)

    async def set_termination_reason(reason: TerminationReason, tool_name: str) -> None:
        del reason, tool_name

    terminated = await run_tool_batch_module.execute_tool_batch_cycle(
        context=context,
        runtime=runtime,
        tool_calls=[
            {"id": "tc_search", "type": "function", "function": {"name": "search"}},
            {
                "id": "tc_review",
                "type": "function",
                "function": {"name": "review_trajectory"},
            },
        ],
        assistant_step_id="assistant-step",
        set_termination_reason=set_termination_reason,
        commit_step=commit_step,
    )

    assert terminated is False
    assert len(hooks.after_step_back_calls) == 1
    outcome = hooks.after_step_back_calls[0]
    assert outcome.mode == "step_back"
    assert outcome.affected_count == 1
    assert outcome.experience == "narrow the search"
    tool_messages = [
        msg for msg in context.ledger.messages if msg.get("role") == "tool"
    ]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "tc_search"
    assert tool_messages[0]["content"] == "[EXPERIENCE] narrow the search"


def test_remove_review_tool_call_omits_empty_tool_calls_key() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "Need to review this path.",
            "tool_calls": [
                {
                    "id": "tc_review",
                    "type": "function",
                    "function": {"name": "review_trajectory", "arguments": "{}"},
                }
            ],
        }
    ]

    run_tool_batch_module._remove_review_tool_call(
        messages,
        review_tool_call_id="tc_review",
    )

    assert len(messages) == 1
    assert "tool_calls" not in messages[0]
