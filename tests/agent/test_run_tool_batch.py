from dataclasses import dataclass, field

import pytest

from agiwo.agent.introspect.models import Milestone
import agiwo.agent.run_tool_batch as run_tool_batch_module
from agiwo.agent.models.config import AgentOptions
from agiwo.agent.models.log import (
    IntrospectionCheckpointRecorded,
    IntrospectionOutcomeRecorded,
    IntrospectionTriggered,
    RunPlanUpdated,
    build_committed_step_entry,
)
from agiwo.agent.models.run import RunLedger, TerminationReason
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.storage.base import InMemoryRunLogStorage
from agiwo.tool.base import ToolResult
from agiwo.utils.abort_signal import AbortSignal


@dataclass
class _FakeHooks:
    review_advice: str | None = None
    before_review_calls: list[dict[str, object]] = field(default_factory=list)

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
        "update_plan": object(),
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
    ledger.plan.milestones = [
        Milestone(id="locate", description="Locate the auth bug", status="active")
    ]
    context = _FakeContext(
        config=AgentOptions(
            enable_trajectory_review=True,
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
    entries = await storage.list_entries(session_id=context.session_id)
    trigger_facts = [
        entry for entry in entries if isinstance(entry, IntrospectionTriggered)
    ]
    assert len(trigger_facts) == 1
    assert trigger_facts[0].trigger_reason == "step_interval"
    assert trigger_facts[0].trigger_tool_call_id == "tc_search"
    assert trigger_facts[0].trigger_tool_step_id == committed_steps[0].id
    assert trigger_facts[0].notice_step_id == committed_steps[0].id


@pytest.mark.asyncio
async def test_execute_tool_batch_cycle_records_run_plan_updated_fact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_execute_tool_batch(*args, **kwargs):
        del args, kwargs
        return [
            ToolResult.success(
                tool_name="update_plan",
                tool_call_id="tc_update",
                content="Plan changes accepted",
                output={
                    "changes": [
                        {
                            "id": "locate",
                            "description": "Locate the auth bug",
                        }
                    ]
                },
            )
        ]

    monkeypatch.setattr(
        run_tool_batch_module, "execute_tool_batch", fake_execute_tool_batch
    )

    storage = InMemoryRunLogStorage()
    session_runtime = SessionRuntime(session_id="sess-1", run_log_storage=storage)
    context = _FakeContext(
        config=AgentOptions(
            enable_trajectory_review=True,
            review_step_interval=100,
        ),
        ledger=RunLedger(),
        hooks=_FakeHooks(),
        session_runtime=session_runtime,
    )
    runtime = _FakeRuntime(tools_map=_review_tools_map())
    committed_steps = []

    async def commit_step(step):
        committed_step = await _commit_step_to_ledger_and_storage(context, step)
        committed_steps.append(committed_step)
        return committed_step

    async def set_termination_reason(reason: TerminationReason, tool_name: str) -> None:
        del reason, tool_name

    await run_tool_batch_module.execute_tool_batch_cycle(
        context=context,
        runtime=runtime,
        tool_calls=[
            {
                "id": "tc_update",
                "type": "function",
                "function": {"name": "update_plan"},
            }
        ],
        assistant_step_id="assistant-step",
        set_termination_reason=set_termination_reason,
        commit_step=commit_step,
    )

    entries = await storage.list_entries(session_id=context.session_id)
    plan_facts = [entry for entry in entries if isinstance(entry, RunPlanUpdated)]
    assert len(plan_facts) == 1
    assert plan_facts[0].source_tool_call_id == "tc_update"
    assert plan_facts[0].source_step_id == committed_steps[0].id
    assert plan_facts[0].revision == 1
    assert plan_facts[0].active_milestone_id == "locate"
    assert [(m.id, m.status) for m in plan_facts[0].milestones] == [
        ("locate", "active")
    ]


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
    ledger.plan.milestones = [
        Milestone(id="locate", description="Locate the auth bug", status="active")
    ]
    context = _FakeContext(
        config=AgentOptions(
            enable_trajectory_review=True,
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
async def test_execute_tool_batch_cycle_injects_only_one_review_per_tool_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_execute_tool_batch(*args, **kwargs):
        del args, kwargs
        return [
            ToolResult.success(
                tool_name="search",
                tool_call_id=f"tc_search_{index}",
                content=f"Found results {index}",
                output={},
            )
            for index in range(4)
        ]

    monkeypatch.setattr(
        run_tool_batch_module, "execute_tool_batch", fake_execute_tool_batch
    )

    storage = InMemoryRunLogStorage()
    session_runtime = SessionRuntime(session_id="sess-1", run_log_storage=storage)
    hooks = _FakeHooks(review_advice=None)
    ledger = RunLedger()
    ledger.plan.milestones = [
        Milestone(id="locate", description="Locate the auth bug", status="active")
    ]
    context = _FakeContext(
        config=AgentOptions(
            enable_trajectory_review=True,
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
            {
                "id": f"tc_search_{index}",
                "type": "function",
                "function": {"name": "search"},
            }
            for index in range(4)
        ],
        assistant_step_id="assistant-step",
        set_termination_reason=set_termination_reason,
        commit_step=commit_step,
    )

    tool_messages = [
        msg for msg in context.ledger.messages if msg.get("role") == "tool"
    ]
    assert sum("<system-review>" in msg["content"] for msg in tool_messages) == 1


@pytest.mark.asyncio
async def test_aligned_review_keeps_messages_append_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches = [
        [
            ToolResult.success(
                tool_name="search",
                tool_call_id="tc_search",
                content="Found results",
                output={},
            )
        ],
        [
            ToolResult.success(
                tool_name="review_trajectory",
                tool_call_id="tc_review",
                content="Trajectory review: aligned=True.",
                output={
                    "aligned": True,
                    "experience": "",
                    "tool_usefulness": [
                        {
                            "tool_call_id": "tc_search",
                            "tool_name": "search",
                            "score": 2,
                        }
                    ],
                },
            )
        ],
    ]

    async def fake_execute_tool_batch(*args, **kwargs):
        del args, kwargs
        return batches.pop(0)

    monkeypatch.setattr(
        run_tool_batch_module, "execute_tool_batch", fake_execute_tool_batch
    )

    storage = InMemoryRunLogStorage()
    session_runtime = SessionRuntime(session_id="sess-1", run_log_storage=storage)
    hooks = _FakeHooks(review_advice=None)
    ledger = RunLedger()
    ledger.plan.milestones = [
        Milestone(id="locate", description="Locate the auth bug", status="active")
    ]
    context = _FakeContext(
        config=AgentOptions(
            enable_trajectory_review=True,
            review_step_interval=1,
        ),
        ledger=ledger,
        hooks=hooks,
        session_runtime=session_runtime,
    )
    runtime = _FakeRuntime(tools_map=_review_tools_map())
    message_snapshots: list[list[dict[str, object]]] = []

    async def commit_step(step):
        committed = await _commit_step_to_ledger_and_storage(context, step)
        message_snapshots.append([dict(message) for message in context.ledger.messages])
        return committed

    async def set_termination_reason(reason: TerminationReason, tool_name: str) -> None:
        del reason, tool_name

    await run_tool_batch_module.execute_tool_batch_cycle(
        context=context,
        runtime=runtime,
        tool_calls=[
            {"id": "tc_search", "type": "function", "function": {"name": "search"}}
        ],
        assistant_step_id="assistant-step-search",
        set_termination_reason=set_termination_reason,
        commit_step=commit_step,
    )
    assert "<system-review>" in context.ledger.messages[-1]["content"]

    await run_tool_batch_module.execute_tool_batch_cycle(
        context=context,
        runtime=runtime,
        tool_calls=[
            {
                "id": "tc_review",
                "type": "function",
                "function": {"name": "review_trajectory"},
            }
        ],
        assistant_step_id="assistant-step-review",
        set_termination_reason=set_termination_reason,
        commit_step=commit_step,
    )

    search_message = next(
        msg for msg in context.ledger.messages if msg.get("tool_call_id") == "tc_search"
    )
    assert search_message["content"].startswith("Found results")
    assert "<system-review>" in search_message["content"]
    assert any(
        msg.get("tool_call_id") == "tc_review" for msg in context.ledger.messages
    )
    assert len(message_snapshots[0]) + 1 == len(context.ledger.messages)

    replayed_steps = await storage.list_step_views(
        session_id=context.session_id,
    )
    search_step = next(
        step for step in replayed_steps if step.tool_call_id == "tc_search"
    )
    assert search_step.to_message()["content"].startswith("Found results")
    assert "<system-review>" in search_step.to_message()["content"]
    entries = await storage.list_entries(session_id=context.session_id)
    checkpoints = [
        entry for entry in entries if isinstance(entry, IntrospectionCheckpointRecorded)
    ]
    outcomes = [
        entry for entry in entries if isinstance(entry, IntrospectionOutcomeRecorded)
    ]
    assert len(checkpoints) == 1
    assert checkpoints[0].review_tool_call_id == "tc_review"
    assert len(outcomes) == 1
    assert outcomes[0].aligned is True
    assert outcomes[0].review_tool_call_id == "tc_review"
    assert outcomes[0].tool_usefulness == [
        {"tool_call_id": "tc_search", "tool_name": "search", "score": 2}
    ]


@pytest.mark.asyncio
async def test_misaligned_review_records_usefulness_without_rewriting_history(
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
                output={
                    "aligned": False,
                    "experience": "narrow the search",
                    "tool_usefulness": [
                        {
                            "tool_call_id": "tc_search",
                            "tool_name": "search",
                            "score": 1,
                        }
                    ],
                },
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
            enable_trajectory_review=True,
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
    tool_messages = [
        msg for msg in context.ledger.messages if msg.get("role") == "tool"
    ]
    assert len(tool_messages) == 2
    assert tool_messages[0]["content"] == "Verbose search output"
    assert tool_messages[1]["tool_call_id"] == "tc_review"
    entries = await storage.list_entries(session_id=context.session_id)
    outcomes = [
        entry for entry in entries if isinstance(entry, IntrospectionOutcomeRecorded)
    ]
    assert len(outcomes) == 1
    assert outcomes[0].aligned is False
    assert outcomes[0].experience == "narrow the search"
    assert outcomes[0].tool_usefulness[0]["score"] == 1


@pytest.mark.asyncio
async def test_review_disabled_skips_notice_and_outcome(
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
    context = _FakeContext(
        config=AgentOptions(enable_trajectory_review=False, review_step_interval=1),
        ledger=RunLedger(),
        hooks=_FakeHooks(),
        session_runtime=session_runtime,
    )
    runtime = _FakeRuntime(tools_map={"update_plan": object()})

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

    assert "<system-review>" not in context.ledger.messages[-1]["content"]
    entries = await storage.list_entries(session_id=context.session_id)
    assert not any(isinstance(entry, IntrospectionTriggered) for entry in entries)
