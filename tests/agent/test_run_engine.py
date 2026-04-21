from collections.abc import AsyncIterator

import pytest

from agiwo.agent import Agent, AgentConfig
from agiwo.agent.models.log import RunLogEntryKind
from agiwo.llm.base import Model, StreamChunk


class _FixedResponseModel(Model):
    def __init__(self, response: str = "ok") -> None:
        super().__init__(id="fixed-model", name="fixed-model", temperature=0.0)
        self._response = response

    async def arun_stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        del messages, tools
        yield StreamChunk(content=self._response)
        yield StreamChunk(finish_reason="stop")


@pytest.mark.asyncio
async def test_agent_run_writes_basic_run_log_entries() -> None:
    agent = Agent(
        AgentConfig(name="run-engine-test", description="run engine test"),
        model=_FixedResponseModel(),
    )

    result = await agent.run("hello", session_id="sess-1")

    assert result.response == "ok"
    entries = await agent.run_step_storage.list_entries(session_id="sess-1")
    kinds = [entry.kind for entry in entries]
    assert RunLogEntryKind.RUN_STARTED in kinds
    assert RunLogEntryKind.CONTEXT_ASSEMBLED in kinds
    assert RunLogEntryKind.USER_STEP_COMMITTED in kinds
    assert RunLogEntryKind.LLM_CALL_STARTED in kinds
    assert RunLogEntryKind.LLM_CALL_COMPLETED in kinds
    assert RunLogEntryKind.RUN_FINISHED in kinds
