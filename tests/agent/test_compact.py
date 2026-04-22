"""
Tests for Context Compact functionality.
"""

import pytest

from agiwo.agent.models.config import AgentOptions
from agiwo.agent.compaction import (
    CompactResult,
    build_compacted_messages,
    compact_if_needed,
)
from agiwo.agent.hooks import HookPhase, HookRegistry, observe
from agiwo.agent.models.log import CompactionFailed, RunLogEntryKind
from agiwo.agent.models.run import CompactMetadata, RunIdentity
from agiwo.agent.run_loop import RunLoopOrchestrator
from agiwo.agent.runtime.context import RunContext, RunRuntime
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.storage.base import InMemoryRunLogStorage
from agiwo.llm.base import Model, StreamChunk


class _CompactModel(Model):
    async def arun_stream(self, messages, tools=None):
        del messages, tools
        yield StreamChunk(content='{"summary": "compressed"}')
        yield StreamChunk(finish_reason="stop")


class TestCompactMetadata:
    def test_create_metadata(self):
        metadata = CompactMetadata(
            session_id="session-1",
            agent_id="agent-1",
            start_seq=0,
            end_seq=10,
            before_token_estimate=5000,
            after_token_estimate=500,
            message_count=10,
            transcript_path="/path/to/transcript.jsonl",
            analysis={"summary": "Test summary", "key_decisions": ["decision1"]},
            compact_model="gpt-4o-mini",
            compact_tokens=1000,
        )

        assert metadata.session_id == "session-1"
        assert metadata.agent_id == "agent-1"
        assert metadata.start_seq == 0
        assert metadata.end_seq == 10
        assert metadata.before_token_estimate == 5000
        assert metadata.after_token_estimate == 500
        assert metadata.get_summary() == "Test summary"

    def test_get_summary_missing(self):
        metadata = CompactMetadata(
            session_id="session-1",
            agent_id="agent-1",
            start_seq=0,
            end_seq=10,
            before_token_estimate=5000,
            after_token_estimate=500,
            message_count=10,
            transcript_path="/path/to/transcript.jsonl",
            analysis={},
        )
        assert metadata.get_summary() == ""


class TestDefaultPrompt:
    def test_build_compacted_messages_includes_summary_and_default_response(self):
        messages = build_compacted_messages(
            system_prompt="system prompt",
            summary="compressed summary",
            transcript_path="/tmp/transcript.jsonl",
        )

        assert messages[0] == {"role": "system", "content": "system prompt"}
        assert messages[1]["role"] == "user"
        assert "compressed summary" in messages[1]["content"]
        assert "/tmp/transcript.jsonl" in messages[1]["content"]
        assert messages[2]["role"] == "assistant"
        assert "Understood" in messages[2]["content"]


@pytest.mark.asyncio
async def test_compact_if_needed_uses_the_same_step_commit_pipeline_as_run_loop(
    tmp_path,
):
    step_storage = InMemoryRunLogStorage()
    session_runtime = SessionRuntime(
        session_id="sess-1",
        run_log_storage=step_storage,
    )
    published = []

    async def publish(item):
        published.append(item)

    session_runtime.publish = publish  # type: ignore[method-assign]

    seen_steps = []

    async def on_step(payload):
        step = payload["step"]
        seen_steps.append(step.name or step.role.value)

    state = RunContext(
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
    state.hooks = HookRegistry(
        [
            observe(
                HookPhase.AFTER_STEP_COMMIT,
                "capture_compaction_steps",
                on_step,
            )
        ]
    )

    result = await compact_if_needed(
        state=state,
        model=_CompactModel(
            id="compact-model", name="compact-model", provider="openai"
        ),
        abort_signal=None,
        max_context_window=1,
        compact_prompt=None,
        compact_start_seq=1,
        root_path=str(tmp_path),
    )

    steps = await step_storage.list_step_views(session_id="sess-1", run_id="run-1")

    assert [step.name for step in steps[-2:]] == ["compact_request", "compact"]
    assert seen_steps[-2:] == ["compact_request", "compact"]
    assert [item.type for item in published if item.type == "step_completed"][-2:] == [
        "step_completed",
        "step_completed",
    ]
    assert result.metadata is not None
    metadata = result.metadata
    assert metadata.get_summary() == "compressed"
    msgs = state.ledger.messages
    assert msgs[0] == {"role": "system", "content": "sys"}
    assert msgs[1]["role"] == "user"
    assert metadata.transcript_path in msgs[1]["content"]
    assert "# Summary\ncompressed" in msgs[1]["content"]
    assert msgs[2]["role"] == "assistant"
    assert "Understood" in msgs[2]["content"]
    assert msgs[3] == {"role": "user", "content": "hello"}
    assert all(message.get("name") != "compact_request" for message in msgs)
    assert state.ledger.compaction.last_metadata == metadata

    persisted = await step_storage.get_latest_compact_metadata("sess-1", "agent-1")
    assert persisted == metadata
    entries = await session_runtime.list_run_log_entries(run_id="run-1")
    kinds = [entry.kind for entry in entries]
    assert RunLogEntryKind.MESSAGES_REBUILT in kinds
    assert RunLogEntryKind.COMPACTION_APPLIED in kinds
    assert [
        item.type
        for item in published
        if item.type in {"messages_rebuilt", "compaction_applied"}
    ] == [
        "messages_rebuilt",
        "compaction_applied",
    ]


@pytest.mark.asyncio
async def test_run_loop_records_compaction_failed_entry_on_failed_attempt(
    monkeypatch,
) -> None:
    async def fake_compact_if_needed(**kwargs):
        del kwargs

        return CompactResult(failed=True, error="compact boom")

    monkeypatch.setattr(
        "agiwo.agent.run_loop.compact_if_needed", fake_compact_if_needed
    )

    session_runtime = SessionRuntime(
        session_id="sess-1",
        run_log_storage=InMemoryRunLogStorage(),
    )
    context = RunContext(
        identity=RunIdentity(
            run_id="run-1",
            agent_id="agent-1",
            agent_name="agent",
        ),
        session_runtime=session_runtime,
    )
    runtime = RunRuntime(
        session_runtime=session_runtime,
        config=AgentOptions(),
        hooks=HookRegistry(),
        model=_CompactModel(id="compact-model", name="compact-model"),
        tools_map={},
        abort_signal=None,
        root_path=".",
        compact_start_seq=0,
        max_input_tokens_per_call=0,
        max_context_window=1,
        compact_prompt=None,
    )

    orchestrator = RunLoopOrchestrator(context, runtime)
    await orchestrator._run_compaction_cycle()

    entries = await session_runtime.list_run_log_entries(run_id="run-1")
    failed = [entry for entry in entries if isinstance(entry, CompactionFailed)]
    assert len(failed) == 1
    assert failed[0].error == "compact boom"
    assert failed[0].attempt == 1
    assert failed[0].terminal is False
