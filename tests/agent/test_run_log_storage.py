import pytest

from agiwo.agent.models.log import RunStarted
from agiwo.agent.storage.base import InMemoryRunLogStorage


@pytest.mark.asyncio
async def test_in_memory_run_log_storage_appends_and_replays() -> None:
    storage = InMemoryRunLogStorage()
    entry1 = RunStarted(
        sequence=1,
        session_id="sess-1",
        run_id="run-1",
        agent_id="agent-1",
        user_input="hello",
    )
    entry2 = RunStarted(
        sequence=2,
        session_id="sess-1",
        run_id="run-2",
        agent_id="agent-1",
        user_input="world",
    )
    entry3 = RunStarted(
        sequence=3,
        session_id="sess-1",
        run_id="run-3",
        agent_id="agent-1",
        user_input="again",
    )
    other_session_entry = RunStarted(
        sequence=1,
        session_id="sess-2",
        run_id="run-4",
        agent_id="agent-2",
        user_input="elsewhere",
    )

    await storage.append_entries([entry1, other_session_entry, entry2, entry3])

    replay = await storage.list_entries(session_id="sess-1")
    assert replay == [entry1, entry2, entry3]


@pytest.mark.asyncio
async def test_in_memory_run_log_storage_rejects_duplicate_sequences() -> None:
    storage = InMemoryRunLogStorage()
    entry = RunStarted(
        sequence=1,
        session_id="sess-1",
        run_id="run-1",
        agent_id="agent-1",
        user_input="hello",
    )

    await storage.append_entries([entry])

    with pytest.raises(ValueError):
        await storage.append_entries([entry])
