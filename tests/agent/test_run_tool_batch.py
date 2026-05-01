import asyncio
from dataclasses import dataclass, field

import pytest

from agiwo.agent.introspect.models import Milestone
import agiwo.agent.run_tool_batch as run_tool_batch_module
from agiwo.agent.models.config import AgentOptions
from agiwo.agent.models.log import (
    ContextStepsHidden,
    GoalMilestonesUpdated,
    IntrospectionCheckpointRecorded,
    IntrospectionOutcomeRecorded,
    IntrospectionTriggered,
    build_committed_step_entry,
)
from agiwo.agent.models.run import RunLedger, TerminationReason
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.runtime.state_ops import remove_tool_call_from_messages
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
    ledger.goal.milestones = [
        Milestone(id="locate", description="Locate the auth bug", status="active")
    ]
    ledger.goal.active_milestone_id = "locate"
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
async def test_execute_tool_batch_cycle_records_declared_milestone_fact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_execute_tool_batch(*args, **kwargs):
        del args, kwargs
        return [
            ToolResult.success(
                tool_name="declare_milestones",
                tool_call_id="tc_declare",
                content="Milestones declared",
                output={
                    "milestones": [
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
            enable_goal_directed_review=True,
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
                "id": "tc_declare",
                "type": "function",
                "function": {"name": "declare_milestones"},
            }
        ],
        assistant_step_id="assistant-step",
        set_termination_reason=set_termination_reason,
        commit_step=commit_step,
    )

    entries = await storage.list_entries(session_id=context.session_id)
    milestone_facts = [
        entry for entry in entries if isinstance(entry, GoalMilestonesUpdated)
    ]
    assert len(milestone_facts) == 1
    assert milestone_facts[0].source_tool_call_id == "tc_declare"
    assert milestone_facts[0].source_step_id == committed_steps[0].id
    assert milestone_facts[0].active_milestone_id == "locate"
    assert [(m.id, m.status) for m in milestone_facts[0].milestones] == [
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
    ledger.goal.milestones = [
        Milestone(id="locate", description="Locate the auth bug", status="active")
    ]
    ledger.goal.active_milestone_id = "locate"
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
    ledger.goal.milestones = [
        Milestone(id="locate", description="Locate the auth bug", status="active")
    ]
    ledger.goal.active_milestone_id = "locate"
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
async def test_aligned_review_cleans_prior_system_review_from_context_and_storage(
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
                output={"aligned": True, "experience": ""},
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
    ledger.goal.milestones = [
        Milestone(id="locate", description="Locate the auth bug", status="active")
    ]
    ledger.goal.active_milestone_id = "locate"
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
    assert search_message["content"] == "Found results"

    replayed_steps = await storage.list_step_views(
        session_id=context.session_id,
        include_hidden_from_context=False,
    )
    search_step = next(
        step for step in replayed_steps if step.tool_call_id == "tc_search"
    )
    assert search_step.to_message()["content"] == "Found results"
    entries = await storage.list_entries(session_id=context.session_id)
    checkpoints = [
        entry for entry in entries if isinstance(entry, IntrospectionCheckpointRecorded)
    ]
    outcomes = [
        entry for entry in entries if isinstance(entry, IntrospectionOutcomeRecorded)
    ]
    assert len(checkpoints) == 1
    assert checkpoints[0].review_tool_call_id == "tc_review"
    assert checkpoints[0].review_step_id
    assert checkpoints[0].milestone_id == "locate"
    assert len(outcomes) == 1
    assert outcomes[0].aligned is True
    assert outcomes[0].mode == "metadata_only"
    assert outcomes[0].review_tool_call_id == "tc_review"
    assert outcomes[0].hidden_step_ids == [
        "assistant-step-review",
        checkpoints[0].review_step_id,
    ]
    assert outcomes[0].notice_cleaned_step_ids == [search_step.id]


@pytest.mark.asyncio
async def test_aligned_review_publishes_hidden_steps_to_live_stream(
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
                output={"aligned": True, "experience": ""},
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
    stream = session_runtime.subscribe()
    ledger = RunLedger()
    ledger.goal.milestones = [
        Milestone(id="locate", description="Locate the auth bug", status="active")
    ]
    ledger.goal.active_milestone_id = "locate"
    context = _FakeContext(
        config=AgentOptions(
            enable_goal_directed_review=True,
            review_step_interval=1,
        ),
        ledger=ledger,
        hooks=_FakeHooks(),
        session_runtime=session_runtime,
    )
    runtime = _FakeRuntime(tools_map=_review_tools_map())

    async def commit_step(step):
        committed_step = await _commit_step_to_ledger_and_storage(context, step)
        await session_runtime.project_run_log_entries(
            [build_committed_step_entry(committed_step)],
            run_id=context.run_id,
            agent_id=context.agent_id,
            parent_run_id=context.parent_run_id,
            depth=context.depth,
        )
        return committed_step

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
    await stream.__anext__()

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

    stream_items = [
        await asyncio.wait_for(stream.__anext__(), timeout=1) for _ in range(2)
    ]
    hidden_events = [
        item for item in stream_items if item.type == "context_steps_hidden"
    ]
    assert len(hidden_events) == 1
    assert hidden_events[0].reason == "introspection_metadata"
    assert "assistant-step-review" in hidden_events[0].step_ids

    entries = await storage.list_entries(session_id=context.session_id)
    hidden_facts = [entry for entry in entries if isinstance(entry, ContextStepsHidden)]
    assert len(hidden_facts) == 1
    assert hidden_events[0].step_ids == hidden_facts[0].step_ids


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
    assert outcome.repair_plan is not None
    assert outcome.repair_plan.affected_count == 1
    assert outcome.experience == "narrow the search"
    tool_messages = [
        msg for msg in context.ledger.messages if msg.get("role") == "tool"
    ]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "tc_search"
    assert tool_messages[0]["content"] == "[EXPERIENCE] narrow the search"
    entries = await storage.list_entries(session_id=context.session_id)
    outcomes = [
        entry for entry in entries if isinstance(entry, IntrospectionOutcomeRecorded)
    ]
    assert len(outcomes) == 1
    assert outcomes[0].aligned is False
    assert outcomes[0].mode == "step_back"
    assert outcomes[0].experience == "narrow the search"
    assert outcomes[0].review_tool_call_id == "tc_review"
    assert outcomes[0].condensed_step_ids
    assert outcomes[0].notice_cleaned_step_ids == []


@pytest.mark.asyncio
async def test_step_back_excludes_trailing_batch_tool_after_review_outcome(
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
            ToolResult.success(
                tool_name="read",
                tool_call_id="tc_read",
                content="Read output after review",
                output={},
            ),
        ]

    monkeypatch.setattr(
        run_tool_batch_module, "execute_tool_batch", fake_execute_tool_batch
    )

    storage = InMemoryRunLogStorage()
    session_runtime = SessionRuntime(session_id="sess-1", run_log_storage=storage)
    context = _FakeContext(
        config=AgentOptions(
            enable_goal_directed_review=True,
            review_step_interval=100,
        ),
        ledger=RunLedger(),
        hooks=_FakeHooks(),
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
            {"id": "tc_search", "type": "function", "function": {"name": "search"}},
            {
                "id": "tc_review",
                "type": "function",
                "function": {"name": "review_trajectory"},
            },
            {"id": "tc_read", "type": "function", "function": {"name": "read"}},
        ],
        assistant_step_id="assistant-step",
        set_termination_reason=set_termination_reason,
        commit_step=commit_step,
    )

    tool_messages = [
        msg for msg in context.ledger.messages if msg.get("role") == "tool"
    ]
    assert [(msg.get("tool_call_id"), msg.get("content")) for msg in tool_messages] == [
        ("tc_search", "[EXPERIENCE] narrow the search"),
        ("tc_read", "Read output after review"),
    ]
    entries = await storage.list_entries(session_id=context.session_id)
    outcomes = [
        entry for entry in entries if isinstance(entry, IntrospectionOutcomeRecorded)
    ]
    assert len(outcomes) == 1
    assert outcomes[0].condensed_step_ids
    condensed_steps = await storage.list_step_views(
        session_id=context.session_id,
        include_hidden_from_context=True,
    )
    read_step = next(step for step in condensed_steps if step.tool_call_id == "tc_read")
    assert read_step.condensed_content is None


def test_remove_review_tool_call_omits_empty_tool_calls_key() -> None:
    context = _FakeContext(
        config=AgentOptions(),
        ledger=RunLedger(
            messages=[
                {
                    "role": "assistant",
                    "content": "Need to review this path.",
                    "tool_calls": [
                        {
                            "id": "tc_review",
                            "type": "function",
                            "function": {
                                "name": "review_trajectory",
                                "arguments": "{}",
                            },
                        }
                    ],
                }
            ]
        ),
        hooks=_FakeHooks(),
        session_runtime=SessionRuntime(
            session_id="sess-1",
            run_log_storage=InMemoryRunLogStorage(),
        ),
    )

    remove_tool_call_from_messages(context, tool_call_id="tc_review")

    assert len(context.ledger.messages) == 1
    assert "tool_calls" not in context.ledger.messages[0]


def test_remove_tool_call_from_messages_treats_empty_string_as_real_id() -> None:
    context = _FakeContext(
        config=AgentOptions(),
        ledger=RunLedger(
            messages=[
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "",
                            "type": "function",
                            "function": {
                                "name": "review_trajectory",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "", "content": "review"},
            ]
        ),
        hooks=_FakeHooks(),
        session_runtime=SessionRuntime(
            session_id="sess-1",
            run_log_storage=InMemoryRunLogStorage(),
        ),
    )

    remove_tool_call_from_messages(context, tool_call_id="")

    assert context.ledger.messages == []


def test_remove_review_tool_call_preserves_structured_assistant_content() -> None:
    context = _FakeContext(
        config=AgentOptions(),
        ledger=RunLedger(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Keep this structured content."}
                    ],
                    "tool_calls": [
                        {
                            "id": "tc_review",
                            "type": "function",
                            "function": {
                                "name": "review_trajectory",
                                "arguments": "{}",
                            },
                        }
                    ],
                }
            ]
        ),
        hooks=_FakeHooks(),
        session_runtime=SessionRuntime(
            session_id="sess-1",
            run_log_storage=InMemoryRunLogStorage(),
        ),
    )

    remove_tool_call_from_messages(context, tool_call_id="tc_review")

    assert len(context.ledger.messages) == 1
    assert context.ledger.messages[0]["content"] == [
        {"type": "text", "text": "Keep this structured content."}
    ]
    assert "tool_calls" not in context.ledger.messages[0]
