"""P3-04: message-tail resume rules and PAUSED without Outcome/termination."""

import asyncio
from collections.abc import AsyncIterator

import pytest

from agiwo.agent import Agent, RunTreeRole, RunExecutionRequest, RunStatus
from agiwo.agent.budget_gate import PermissiveLlmBudgetGate
from agiwo.agent.models.config import AgentConfig
from agiwo.agent.models.input import ContentPart, ContentType, UserMessage
from agiwo.agent.models.log import (
    ContextAssembled,
    RunCheckpoint,
    RunPaused,
    RunStarted,
)
from agiwo.agent.resume import build_resume_plan
from agiwo.agent.storage.serialization import build_run_view_from_entries
from agiwo.llm.base import Model, StreamChunk


class _GateModel(Model):
    def __init__(self) -> None:
        super().__init__(id="gate", name="gate", temperature=0.0)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def arun_stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        del messages, tools
        self.started.set()
        await self.release.wait()
        yield StreamChunk(content="done")
        yield StreamChunk(finish_reason="stop")


def _user(text: str = "hi") -> UserMessage:
    return UserMessage(
        content=[ContentPart(type=ContentType.TEXT, text=text)],
        is_user_provided=True,
    )


def _messages_tail(role: str, *, tool_calls: list | None = None) -> list[dict]:
    msg: dict = {"role": role, "content": "x"}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return [{"role": "user", "content": "hi"}, msg]


def test_resume_plan_assistant_with_tool_calls() -> None:
    tool_calls = [{"id": "c1", "type": "function", "function": {"name": "t"}}]
    entries = [
        RunStarted(
            sequence=1,
            session_id="s",
            run_id="r",
            agent_id="a",
            user_input=_user(),
        ),
        ContextAssembled(
            sequence=2,
            session_id="s",
            run_id="r",
            agent_id="a",
            messages=_messages_tail("assistant", tool_calls=tool_calls),
        ),
        RunCheckpoint(
            sequence=3,
            session_id="s",
            run_id="r",
            agent_id="a",
            checkpoint_id="chk1",
            last_committed_sequence=2,
        ),
        RunPaused(
            sequence=4,
            session_id="s",
            run_id="r",
            agent_id="a",
            checkpoint_id="chk1",
            reason="user_pause",
        ),
    ]
    plan = build_resume_plan(entries)
    assert plan.pending_tool_calls == tool_calls
    assert plan.continue_user_message is None


@pytest.mark.parametrize(
    "last",
    [
        {"role": "assistant", "content": "ok"},
        {"role": "tool", "content": "result", "tool_call_id": "c1"},
    ],
)
def test_resume_plan_continues_with_system_user(last: dict) -> None:
    entries = [
        RunStarted(
            sequence=1,
            session_id="s",
            run_id="r",
            agent_id="a",
            user_input=_user(),
        ),
        ContextAssembled(
            sequence=2,
            session_id="s",
            run_id="r",
            agent_id="a",
            messages=[{"role": "user", "content": "hi"}, last],
        ),
        RunCheckpoint(
            sequence=3,
            session_id="s",
            run_id="r",
            agent_id="a",
            checkpoint_id="chk1",
            last_committed_sequence=2,
        ),
        RunPaused(
            sequence=4,
            session_id="s",
            run_id="r",
            agent_id="a",
            checkpoint_id="chk1",
            reason="user_pause",
        ),
    ]
    plan = build_resume_plan(entries)
    assert plan.pending_tool_calls is None
    assert plan.continue_user_message is not None
    assert plan.continue_user_message.is_user_provided is False


def test_paused_projection_has_no_termination() -> None:
    entries = [
        RunStarted(
            sequence=1,
            session_id="s",
            run_id="r",
            agent_id="a",
            user_input=_user(),
        ),
        RunCheckpoint(
            sequence=2,
            session_id="s",
            run_id="r",
            agent_id="a",
            checkpoint_id="chk1",
            last_committed_sequence=1,
        ),
        RunPaused(
            sequence=3,
            session_id="s",
            run_id="r",
            agent_id="a",
            checkpoint_id="chk1",
            reason="budget",
        ),
    ]
    view = build_run_view_from_entries(entries)
    assert view is not None
    assert view.status is RunStatus.PAUSED
    assert view.termination_reason is None


@pytest.mark.asyncio
async def test_cooperative_pause_same_run_id() -> None:
    model = _GateModel()
    agent = Agent(
        AgentConfig(name="t", description="t"),
        model=model,
        id="agent-pause",
    )
    agent.llm_budget_gate = PermissiveLlmBudgetGate()
    handle = agent._start_runtime(
        UserMessage.from_system("sys"),
        session_id="sess-pause",
        execution_request=RunExecutionRequest(
            run_id="run_pause_1",
            objective_id="o1",
            run_tree_role=RunTreeRole.ROOT,
        ),
    )
    await asyncio.wait_for(model.started.wait(), timeout=5)
    handle.request_pause("user_pause")
    model.release.set()
    result = await handle.wait()
    assert result.paused is True
    assert result.run_id == "run_pause_1"
    assert result.termination_reason is None
    assert result.checkpoint_id is not None
    view = await agent.run_log_storage.get_run_view("run_pause_1")
    assert view is not None
    assert view.status is RunStatus.PAUSED
    await agent.close()
