from agiwo.agent import (
    AssistantStepCommitted,
    HookFailed,
    RunLogEntryKind,
    RunStarted,
    TerminationDecided,
)


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
    assert TerminationDecided.__name__ == "TerminationDecided"
