from collections.abc import AsyncIterator

import pytest

from agiwo.agent import Agent, AgentConfig, TerminationReason
from agiwo.agent.hooks import HookPhase, HookRegistry, observe, transform
from agiwo.agent.models.log import (
    AssistantStepCommitted,
    HookFailed,
    LLMCallStarted,
    MessagesRebuilt,
    RunFinished,
    RunLogEntryKind,
)
from agiwo.agent.models.config import AgentOptions
from agiwo.agent.models.run import RunIdentity
from agiwo.agent.models.stream import stream_items_from_entries
from agiwo.agent.models.step import MessageRole
from agiwo.agent.models.step import StepView
from agiwo.agent.run_loop import RunLoopOrchestrator
from agiwo.agent.runtime.context import RunContext, RunRuntime
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.storage.base import InMemoryRunLogStorage
from agiwo.llm.base import Model, StreamChunk


class _FixedResponseModel(Model):
    def __init__(self, response: str = "ok") -> None:
        super().__init__(id="fixed-model", name="fixed-model", temperature=0.0)
        self._response = response

    async def arun_stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        del messages, tools
        yield StreamChunk(content=self._response)
        yield StreamChunk(finish_reason="stop")


class _NeverCalledModel(Model):
    def __init__(self) -> None:
        super().__init__(id="never", name="never", temperature=0.0)

    async def arun_stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        del messages, tools
        raise AssertionError("llm should not run when prepare fails")


@pytest.mark.asyncio
async def test_orchestrator_commit_step_writes_projects_and_dispatches_hook() -> None:
    storage = InMemoryRunLogStorage()
    session_runtime = SessionRuntime(
        session_id="commit-session",
        run_log_storage=storage,
    )
    published = []

    async def publish(item):
        published.append(item)

    session_runtime.publish = publish  # type: ignore[method-assign]
    seen_steps = []

    async def capture_step(payload: dict) -> None:
        seen_steps.append(payload["step"].role.value)

    hooks = HookRegistry(
        [
            observe(
                HookPhase.AFTER_STEP_COMMIT,
                "capture_step",
                capture_step,
            )
        ]
    )
    context = RunContext(
        identity=RunIdentity(
            run_id="run-1",
            agent_id="agent-1",
            agent_name="agent",
        ),
        session_runtime=session_runtime,
    )
    context.hooks = hooks
    runtime = RunRuntime(
        session_runtime=session_runtime,
        config=AgentOptions(),
        hooks=hooks,
        model=_FixedResponseModel(),
        tools_map={},
        abort_signal=None,
        root_path=".",
        compact_start_seq=0,
        max_input_tokens_per_call=0,
        max_context_window=None,
        compact_prompt=None,
    )
    orchestrator = RunLoopOrchestrator(context, runtime)
    step = StepView.user(context, sequence=1, user_input="hello")

    committed = await orchestrator._commit_step(step)

    assert committed is step
    entries = await session_runtime.list_run_log_entries(run_id="run-1")
    assert [entry.kind.value for entry in entries] == ["user_step_committed"]
    assert [item.type for item in published] == ["step_completed"]
    assert seen_steps == ["user"]
    assert context.ledger.messages[-1]["role"] == "user"
    assert context.ledger.messages[-1]["content"] == "hello"


@pytest.mark.asyncio
async def test_agent_run_writes_basic_run_log_entries() -> None:
    agent = Agent(
        AgentConfig(name="run-engine-test", description="run engine test"),
        model=_FixedResponseModel(),
    )

    result = await agent.run("hello", session_id="sess-1")

    assert result.response == "ok"
    entries = await agent.run_log_storage.list_entries(session_id="sess-1")
    kinds = [entry.kind for entry in entries]
    assert RunLogEntryKind.RUN_STARTED in kinds
    assert RunLogEntryKind.CONTEXT_ASSEMBLED in kinds
    assert RunLogEntryKind.USER_STEP_COMMITTED in kinds
    assert RunLogEntryKind.LLM_CALL_STARTED in kinds
    assert RunLogEntryKind.LLM_CALL_COMPLETED in kinds
    assert RunLogEntryKind.TERMINATION_DECIDED in kinds
    assert RunLogEntryKind.RUN_FINISHED in kinds
    replayed = stream_items_from_entries(entries)
    assert [
        item.type
        for item in replayed
        if item.type in {"termination_decided", "run_completed"}
    ] == [
        "termination_decided",
        "run_completed",
    ]
    termination_event = next(
        item for item in replayed if item.type == "termination_decided"
    )
    assert termination_event.termination_reason == TerminationReason.COMPLETED


@pytest.mark.asyncio
async def test_prepare_failure_after_run_started_writes_run_failed() -> None:
    async def explode(payload: dict) -> dict:
        del payload
        raise RuntimeError("prepare boom")

    agent = Agent(
        AgentConfig(name="strict-run", description="strict run"),
        model=_NeverCalledModel(),
        hooks=HookRegistry(
            [
                transform(
                    HookPhase.PREPARE,
                    "explode",
                    explode,
                    critical=True,
                )
            ]
        ),
    )

    with pytest.raises(RuntimeError, match="prepare boom"):
        await agent.run("hello", session_id="strict-run-session")

    entries = await agent.run_log_storage.list_entries(session_id="strict-run-session")
    assert [entry.kind.value for entry in entries][-1] == "run_failed"


@pytest.mark.asyncio
async def test_before_llm_message_rewrite_becomes_messages_rebuilt_fact() -> None:
    async def rewrite_messages(payload: dict) -> dict:
        updated = dict(payload)
        messages = list(payload["messages"])
        messages.append({"role": "user", "content": "steered follow-up"})
        updated["messages"] = messages
        return updated

    agent = Agent(
        AgentConfig(name="rewrite-test", description="rewrite test"),
        model=_FixedResponseModel(),
        hooks=HookRegistry(
            [
                transform(
                    HookPhase.BEFORE_LLM,
                    "rewrite_messages",
                    rewrite_messages,
                )
            ]
        ),
    )

    result = await agent.run("hello", session_id="rewrite-session")

    assert result.response == "ok"
    entries = await agent.run_log_storage.list_entries(session_id="rewrite-session")
    rebuilt = next(entry for entry in entries if isinstance(entry, MessagesRebuilt))
    llm_started = next(entry for entry in entries if isinstance(entry, LLMCallStarted))

    assert rebuilt.reason == "before_llm"
    assert rebuilt.messages[-1] == {
        "role": "user",
        "content": "steered follow-up",
    }
    assert llm_started.messages == rebuilt.messages


@pytest.mark.asyncio
async def test_before_llm_failure_records_accepted_steer_input_before_run_failed() -> (
    None
):
    async def explode(payload: dict) -> dict:
        assert payload["messages"][-1] == {
            "role": "user",
            "content": "follow up",
        }
        raise RuntimeError("before llm boom")

    session_runtime = SessionRuntime(
        session_id="steer-failure-session",
        run_log_storage=InMemoryRunLogStorage(),
    )
    context = RunContext(
        identity=RunIdentity(
            run_id="run-1",
            agent_id="agent-1",
            agent_name="agent",
        ),
        session_runtime=session_runtime,
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ],
    )
    hooks = HookRegistry(
        [
            transform(
                HookPhase.BEFORE_LLM,
                "explode",
                explode,
                critical=True,
            )
        ]
    )
    context.hooks = hooks
    runtime = RunRuntime(
        session_runtime=session_runtime,
        config=AgentOptions(),
        hooks=hooks,
        model=_NeverCalledModel(),
        tools_map={},
        abort_signal=None,
        root_path=".",
        compact_start_seq=0,
        max_input_tokens_per_call=0,
        max_context_window=None,
        compact_prompt=None,
    )
    orchestrator = RunLoopOrchestrator(context, runtime)

    await orchestrator._start_run("hello")
    accepted = await session_runtime.enqueue_steer("follow up")

    assert accepted is True

    with pytest.raises(RuntimeError, match="before llm boom") as exc_info:
        await orchestrator._run_assistant_turn()
    await orchestrator._fail_run(exc_info.value)

    entries = await session_runtime.list_run_log_entries(run_id="run-1")
    rebuilt = next(entry for entry in entries if isinstance(entry, MessagesRebuilt))
    hook_failed = next(entry for entry in entries if isinstance(entry, HookFailed))

    assert rebuilt.reason == "before_llm"
    assert rebuilt.messages[-1] == {
        "role": "user",
        "content": "follow up",
    }
    assert hook_failed.phase == "before_llm"
    assert hook_failed.handler_name == "explode"
    assert rebuilt.sequence < hook_failed.sequence < entries[-1].sequence
    assert entries[-1].kind.value == "run_failed"


def test_stream_items_from_entries_reuses_nested_run_context_without_run_started() -> (
    None
):
    items = stream_items_from_entries(
        [
            AssistantStepCommitted(
                sequence=2,
                session_id="sess-1",
                run_id="child-run",
                agent_id="agent-1",
                step_id="step-2",
                role=MessageRole.ASSISTANT,
                content="done",
                parent_run_id="parent-run",
                depth=1,
            ),
            RunFinished(
                sequence=3,
                session_id="sess-1",
                run_id="child-run",
                agent_id="agent-1",
                response="done",
                termination_reason=TerminationReason.COMPLETED,
            ),
        ]
    )

    assert items[0].parent_run_id == "parent-run"
    assert items[0].depth == 1
    assert items[1].parent_run_id == "parent-run"
    assert items[1].depth == 1
