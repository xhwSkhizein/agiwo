import asyncio
import inspect
from collections.abc import AsyncIterator

import pytest

from agiwo.agent import (
    AgentConfig,
    AgentOptions,
    AgentSpec,
    MainAgent,
    MainAgentState,
    MessageRole,
)
from agiwo.agent.hooks import HookRegistry
from agiwo.agent.models.log import UserStepCommitted
from agiwo.agent import run_bootstrap
from agiwo.llm.base import Model, StreamChunk


class FixedResponseModel(Model):
    def __init__(
        self,
        response: str = "ok",
        *,
        start_event: asyncio.Event | None = None,
    ) -> None:
        super().__init__(id="fixed-model", name="fixed-model", temperature=0.0)
        self._response = response
        self._start_event = start_event

    async def arun_stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        del messages, tools
        if self._start_event is not None:
            await self._start_event.wait()
        yield StreamChunk(content=self._response)
        yield StreamChunk(finish_reason="stop")


def _build_spec() -> AgentSpec:
    return AgentSpec(
        config=AgentConfig(
            name="main-agent",
            description="AB-02 test agent",
            system_prompt="Test prompt",
            options=AgentOptions(max_steps_per_run=5),
        )
    )


def _build_main_agent(
    *,
    session_id: str = "session-1",
    agent_id: str = "agent-1",
    model: Model | None = None,
) -> MainAgent:
    return MainAgent(
        session_id=session_id,
        agent_id=agent_id,
        spec=_build_spec(),
        model=model or FixedResponseModel(),
        hooks=HookRegistry(),
    )


def _session_history_run_id(session_id: str) -> str:
    return f"session-history-{session_id}"


async def _count_session_history_user_steps(
    main_agent: MainAgent,
    *,
    session_id: str,
) -> int:
    entries = await main_agent.run_log_storage.list_entries(session_id=session_id)
    return sum(
        1
        for entry in entries
        if isinstance(entry, UserStepCommitted)
        and entry.run_id == _session_history_run_id(session_id)
    )


@pytest.mark.asyncio
async def test_idle_accept_runs_to_completion_and_returns_idle() -> None:
    main = _build_main_agent()

    handle = await main.accept("hello")

    assert handle is not None
    assert main.state is MainAgentState.RUNNING

    output = await main.wait_current_run()

    assert output.error is None
    assert main.state is MainAgentState.IDLE
    assert main._handle is None


@pytest.mark.asyncio
async def test_busy_accept_records_two_user_steps_without_double_run_commit() -> None:
    start_event = asyncio.Event()
    main = _build_main_agent(model=FixedResponseModel(start_event=start_event))

    first_handle = await main.accept("first message")
    assert first_handle is not None
    assert main.state is MainAgentState.RUNNING

    second_handle = await main.accept("second message")

    assert second_handle is first_handle
    assert (
        await _count_session_history_user_steps(main, session_id=main.session_id) == 2
    )

    start_event.set()
    await main.wait_current_run()

    assert main.state is MainAgentState.IDLE


@pytest.mark.asyncio
async def test_second_run_sees_first_run_history_for_same_agent_id() -> None:
    session_id = "session-history-chain"
    agent_id = "main-agent-id"
    main = _build_main_agent(session_id=session_id, agent_id=agent_id)

    await main.accept("hello")
    await main.wait_current_run()
    assert main.state is MainAgentState.IDLE

    await main.accept("follow up")
    await main.wait_current_run()

    steps = await main.run_log_storage.list_step_views(
        session_id=session_id,
        agent_id=agent_id,
    )
    user_texts = [
        step.get_display_text()
        for step in steps
        if step.role is MessageRole.USER and step.get_display_text()
    ]
    assistant_texts = [
        step.get_display_text()
        for step in steps
        if step.role is MessageRole.ASSISTANT and step.get_display_text()
    ]

    assert "hello" in user_texts
    assert "follow up" in user_texts
    assert assistant_texts.count("ok") >= 2


@pytest.mark.asyncio
async def test_cancel_mid_run_returns_idle_and_accept_works_again() -> None:
    start_event = asyncio.Event()
    main = _build_main_agent(model=FixedResponseModel(start_event=start_event))

    await main.accept("block until cancel")
    assert main.state is MainAgentState.RUNNING

    cancel_task = asyncio.create_task(main.cancel("test cancel"))
    await asyncio.sleep(0.05)
    start_event.set()
    await cancel_task

    assert main.state is MainAgentState.IDLE

    handle = await main.accept("after cancel")
    assert handle is not None
    output = await handle.wait()

    assert output.error is None
    assert main.state is MainAgentState.IDLE


def test_run_bootstrap_loads_history_by_agent_id_not_run_id_only() -> None:
    source = inspect.getsource(run_bootstrap._load_existing_steps)
    assert "agent_id=context.agent_id" in source
    assert (
        "run_id=context.run_id" not in source.split("list_step_views")[1].split(")")[0]
    )
