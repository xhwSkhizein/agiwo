import pytest

from agiwo.agent.models.log import RunStarted
from agiwo.agent.storage.base import InMemoryRunLogStorage


@pytest.mark.asyncio
async def test_in_memory_run_log_storage_appends_and_replays() -> None:
    storage = InMemoryRunLogStorage()
    entry = RunStarted(
        sequence=1,
        session_id="sess-1",
        run_id="run-1",
        agent_id="agent-1",
        user_input="hello",
    )

    await storage.append_entries([entry])

    replay = await storage.list_entries(session_id="sess-1")
    assert replay == [entry]
