"""C-02: MainAgent SessionIntent hooks on accept and Run end."""

import asyncio
from pathlib import Path
from collections.abc import AsyncIterator

import pytest

from agiwo.agent import (
    AgentConfig,
    AgentOptions,
    AgentSpec,
    InMemorySessionIntentStore,
    MainAgent,
    MainAgentState,
    RunLogStorageConfig,
    summarize_run_report,
)
from agiwo.agent.hooks import HookRegistry
from agiwo.agent.intent.report import REPORT_SUMMARY_MAX_LEN
from agiwo.agent.models.config import AgentStorageOptions
from agiwo.agent.models.log import RunPlanUpdated, UserStepCommitted
from agiwo.agent.models.plan import Milestone
from agiwo.config.termination import TerminationReason
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


def _build_spec(*, max_steps: int = 5) -> AgentSpec:
    return AgentSpec(
        config=AgentConfig(
            name="main-agent",
            description="C-02 test agent",
            system_prompt="Test prompt",
            options=AgentOptions(
                max_steps_per_run=max_steps,
                storage=AgentStorageOptions(
                    run_log_storage=RunLogStorageConfig(storage_type="memory"),
                ),
            ),
        )
    )


def _build_main_agent(
    *,
    session_id: str = "session-intent",
    agent_id: str = "agent-intent",
    model: Model | None = None,
    intent_store: InMemorySessionIntentStore | None = None,
) -> MainAgent:
    return MainAgent(
        session_id=session_id,
        agent_id=agent_id,
        spec=_build_spec(),
        model=model or FixedResponseModel(),
        hooks=HookRegistry(),
        session_intent_store=intent_store or InMemorySessionIntentStore(),
    )


async def _count_session_history_user_steps(main_agent: MainAgent) -> int:
    session_id = main_agent.session_id
    run_id = f"session-history-{session_id}"
    entries = await main_agent.run_log_storage.list_entries(session_id=session_id)
    return sum(
        1
        for entry in entries
        if isinstance(entry, UserStepCommitted) and entry.run_id == run_id
    )


@pytest.mark.asyncio
async def test_accept_appends_full_user_text_after_runlog_write() -> None:
    intent_store = InMemorySessionIntentStore()
    main = _build_main_agent(intent_store=intent_store)
    long_text = "x" * 4000

    await main.accept(long_text)
    await main.wait_current_run()

    assert await _count_session_history_user_steps(main) == 1
    intent = await intent_store.get(main.session_id)
    assert intent is not None
    user_entries = [entry for entry in intent.entries if entry.kind == "user_input"]
    assert len(user_entries) == 1
    assert user_entries[0].text == long_text


@pytest.mark.asyncio
async def test_successful_run_appends_summarized_report() -> None:
    intent_store = InMemorySessionIntentStore()
    raw_response = "r" * (REPORT_SUMMARY_MAX_LEN + 200)
    main = _build_main_agent(
        intent_store=intent_store,
        model=FixedResponseModel(response=raw_response),
    )

    await main.accept("do work")
    output = await main.wait_current_run()

    assert output.termination_reason is TerminationReason.COMPLETED
    intent = await intent_store.get(main.session_id)
    assert intent is not None
    report_entries = [entry for entry in intent.entries if entry.kind == "run_report"]
    assert len(report_entries) == 1
    assert len(report_entries[0].text) < len(raw_response)
    assert report_entries[0].text == summarize_run_report(response=raw_response)
    assert report_entries[0].run_id is not None


@pytest.mark.asyncio
async def test_cancelled_run_does_not_append_run_report() -> None:
    start_event = asyncio.Event()
    intent_store = InMemorySessionIntentStore()
    main = _build_main_agent(
        intent_store=intent_store,
        model=FixedResponseModel(start_event=start_event),
    )

    await main.accept("block")
    cancel_task = asyncio.create_task(main.cancel("user cancel"))
    await asyncio.sleep(0.05)
    start_event.set()
    await cancel_task

    intent = await intent_store.get(main.session_id)
    assert intent is not None
    assert [entry.kind for entry in intent.entries] == ["user_input"]
    assert not any(entry.kind == "run_report" for entry in intent.entries)


@pytest.mark.asyncio
async def test_successful_run_snapshots_last_run_plan_when_present() -> None:
    intent_store = InMemorySessionIntentStore()
    main = _build_main_agent(intent_store=intent_store)

    handle = await main.accept("plan work")
    assert handle is not None
    sequence = await main.run_log_storage.allocate_sequence(main.session_id)
    await main.run_log_storage.append_entries(
        [
            RunPlanUpdated(
                sequence=sequence,
                session_id=main.session_id,
                run_id=handle.run_id,
                agent_id=main.agent_id,
                milestones=[
                    Milestone(
                        id="m1",
                        description="finish wave c",
                        status="completed",
                    )
                ],
                revision=1,
                reason="declared",
            )
        ]
    )

    await main.wait_current_run()

    intent = await intent_store.get(main.session_id)
    assert intent is not None
    assert intent.last_run_plan is not None
    assert intent.last_run_plan.milestones[0].description == "finish wave c"


def test_session_intent_append_entry_only_called_from_main_agent() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    hits: list[str] = []
    for path in repo_root.rglob("*.py"):
        if "tests/" in str(path) or path.name == "base.py":
            continue
        text = path.read_text(encoding="utf-8")
        if ".append_entry(" in text and "session_intent" in text:
            hits.append(str(path.relative_to(repo_root)))
    assert hits == ["agiwo/agent/main_agent.py"]


@pytest.mark.asyncio
async def test_busy_accept_records_multiple_full_user_entries() -> None:
    intent_store = InMemorySessionIntentStore()
    start_event = asyncio.Event()
    main = _build_main_agent(
        intent_store=intent_store,
        model=FixedResponseModel(start_event=start_event),
    )

    await main.accept("first")
    await main.accept("second longer message")

    intent = await intent_store.get(main.session_id)
    assert intent is not None
    user_entries = [entry for entry in intent.entries if entry.kind == "user_input"]
    assert [entry.text for entry in user_entries] == [
        "first",
        "second longer message",
    ]

    start_event.set()
    await main.wait_current_run()
    assert main.state is MainAgentState.IDLE
