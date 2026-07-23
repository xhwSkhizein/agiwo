"""P2-02: preallocated Run identity and public Agent API stability."""

import inspect
from collections.abc import AsyncIterator

import pytest

from agiwo.agent import Agent, RunTreeRole, RunExecutionRequest, RunStatus
from agiwo.agent.models.config import AgentConfig
from agiwo.agent.models.input import ContentPart, ContentType, UserMessage
from agiwo.agent.models.log import RunStarted
from agiwo.agent.storage.serialization import (
    build_run_view_from_entries,
    deserialize_run_log_entry_from_storage,
    serialize_run_log_entry_for_storage,
)
from agiwo.llm.base import Model, StreamChunk
from agiwo.scheduler import Scheduler


class _FixedResponseModel(Model):
    def __init__(self, response: str = "ok") -> None:
        super().__init__(id="fixed", name="fixed", temperature=0.0)
        self._response = response

    async def arun_stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        del messages, tools
        yield StreamChunk(content=self._response)
        yield StreamChunk(finish_reason="stop")


def _user(text: str = "hi") -> UserMessage:
    return UserMessage(
        content=[ContentPart(type=ContentType.TEXT, text=text)],
        is_user_provided=True,
    )


def test_public_agent_start_signature_unchanged() -> None:
    sig = inspect.signature(Agent.start)
    assert "execution_request" not in sig.parameters
    assert "run_id" not in sig.parameters
    for name in ("run", "run_stream"):
        method = getattr(Agent, name)
        assert "execution_request" not in inspect.signature(method).parameters


def test_run_status_enum_shape() -> None:
    assert {s.value for s in RunStatus} == {
        "running",
        "paused",
        "completed",
        "interrupted",
        "failed",
    }


def test_run_execution_request_accepts_root() -> None:
    req = RunExecutionRequest(
        run_id="r1",
        run_tree_role=RunTreeRole.ROOT,
    )
    assert req.run_tree_role is RunTreeRole.ROOT


@pytest.mark.asyncio
async def test_start_prevalidated_uses_preallocated_run_id() -> None:
    agent = Agent(
        AgentConfig(name="t", description="t"),
        model=_FixedResponseModel(),
        id="agent-1",
    )
    request = RunExecutionRequest(
        run_id="run_fixed_1",
        run_tree_role=RunTreeRole.ROOT,
    )
    handle = agent.start_prevalidated(
        UserMessage.from_system("sys"),
        session_id="sess-1",
        execution_request=request,
    )
    assert handle.run_id == "run_fixed_1"
    result = await handle.wait()
    assert result.run_id == "run_fixed_1"
    view = await agent.run_log_storage.get_run_view("run_fixed_1")
    assert view is not None
    assert view.run_tree_role == RunTreeRole.ROOT
    await agent.close()


def test_run_started_round_trip_keeps_identity_fields() -> None:
    entry = RunStarted(
        sequence=1,
        session_id="s",
        run_id="r",
        agent_id="a",
        user_input=_user(),
        run_tree_role=RunTreeRole.ROOT.value,
    )
    stored = serialize_run_log_entry_for_storage(entry)
    restored = deserialize_run_log_entry_from_storage(stored)
    assert isinstance(restored, RunStarted)
    assert restored.run_tree_role == "root"
    view = build_run_view_from_entries([restored])
    assert view is not None
    assert view.run_tree_role == RunTreeRole.ROOT


def test_scheduler_dropped_session_facade_methods() -> None:
    for name in ("route_root_input", "dispatch_execution", "inject_user_message"):
        assert not hasattr(Scheduler, name), f"Scheduler must not expose {name!r}"
