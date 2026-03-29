"""
Tests for Context Compact functionality.
"""

import pytest

from agiwo.agent.compaction import build_compacted_messages, compact_if_needed
from agiwo.agent.models.run import CompactMetadata
from agiwo.agent.hooks import AgentHooks
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.storage.base import InMemoryRunStepStorage
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
    step_storage = InMemoryRunStepStorage()
    session_runtime = SessionRuntime(
        session_id="sess-1",
        run_step_storage=step_storage,
    )
    published = []

    async def publish(item):
        published.append(item)

    session_runtime.publish = publish  # type: ignore[method-assign]

    seen_steps = []

    async def on_step(step):
        seen_steps.append(step.name or step.role.value)

    state = RunContext(
        session_runtime=session_runtime,
        run_id="run-1",
        agent_id="agent-1",
        agent_name="agent",
        hooks=AgentHooks(on_step=on_step),
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ],
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

    steps = await step_storage.get_steps("sess-1", run_id="run-1")

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
    assert state.ledger.last_compact_metadata == metadata

    persisted = await step_storage.get_latest_compact_metadata("sess-1", "agent-1")
    assert persisted == metadata
