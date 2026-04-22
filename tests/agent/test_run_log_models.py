from agiwo.agent import (
    AssistantStepCommitted,
    CompactionFailed,
    HookFailed,
    RunLogEntryKind,
    RunStarted,
    TerminationDecided,
)
from agiwo.agent.models.stream import stream_items_from_entries


def test_run_log_entries_are_public_and_typed() -> None:
    entry = RunStarted(
        sequence=1,
        session_id="sess-1",
        run_id="run-1",
        agent_id="agent-1",
        user_input="hello",
    )

    assert entry.kind is RunLogEntryKind.RUN_STARTED
    assert HookFailed.__name__ == "HookFailed"
    assert AssistantStepCommitted.__name__ == "AssistantStepCommitted"
    assert CompactionFailed.__name__ == "CompactionFailed"
    assert TerminationDecided.__name__ == "TerminationDecided"


def test_stream_items_from_entries_replays_compaction_failed_event() -> None:
    items = stream_items_from_entries(
        [
            CompactionFailed(
                sequence=4,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                error="compact boom",
                attempt=1,
                max_attempts=3,
                terminal=False,
            )
        ]
    )

    assert items[0].type == "compaction_failed"
    assert items[0].error == "compact boom"
    assert RunLogEntryKind.COMPACTION_FAILED.value == "compaction_failed"
