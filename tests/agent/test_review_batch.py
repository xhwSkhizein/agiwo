import pytest

from agiwo.agent.models.config import AgentOptions
from agiwo.agent.models.review import Milestone
from agiwo.agent.models.run import RunLedger
from agiwo.agent.review import ReviewBatch, _build_system_review_cleanup_updates
from agiwo.agent.storage.base import InMemoryRunLogStorage


class FakeTool:
    def __init__(self, name):
        self.name = name


class FakeToolResult:
    def __init__(
        self,
        tool_name,
        content,
        is_success=True,
        tool_call_id="tc_1",
        output=None,
    ):
        self.tool_name = tool_name
        self.content = content
        self.is_success = is_success
        self.tool_call_id = tool_call_id
        self.input_args = {}
        self.content_for_user = None
        self.termination_reason = None
        self.output = output or {}


class TestReviewBatch:
    def test_enabled_when_configured_and_tools_present(self):
        config = AgentOptions(enable_goal_directed_review=True)
        ledger = RunLedger()
        tools_map = {
            "review_trajectory": FakeTool("review_trajectory"),
            "declare_milestones": FakeTool("declare_milestones"),
        }
        batch = ReviewBatch(config, ledger, tools_map)
        assert batch.enabled is True

    def test_disabled_when_flag_off(self):
        config = AgentOptions(enable_goal_directed_review=False)
        ledger = RunLedger()
        tools_map = {
            "review_trajectory": FakeTool("review_trajectory"),
            "declare_milestones": FakeTool("declare_milestones"),
        }
        batch = ReviewBatch(config, ledger, tools_map)
        assert batch.enabled is False

    def test_disabled_when_review_tool_missing(self):
        config = AgentOptions(enable_goal_directed_review=True)
        ledger = RunLedger()
        tools_map = {"declare_milestones": FakeTool("declare_milestones")}
        batch = ReviewBatch(config, ledger, tools_map)
        assert batch.enabled is False

    def test_disabled_when_milestones_tool_missing(self):
        config = AgentOptions(enable_goal_directed_review=True)
        ledger = RunLedger()
        tools_map = {"review_trajectory": FakeTool("review_trajectory")}
        batch = ReviewBatch(config, ledger, tools_map)
        assert batch.enabled is False

    def test_process_result_records_aligned_checkpoint(self):
        config = AgentOptions(enable_goal_directed_review=True)
        ledger = RunLedger()
        ledger.review.consecutive_errors = 2
        ledger.messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc_review",
                        "type": "function",
                        "function": {
                            "name": "review_trajectory",
                            "arguments": "{}",
                        },
                    }
                ],
                "_sequence": 6,
            },
            {
                "role": "tool",
                "tool_call_id": "tc_review",
                "content": "Trajectory review: aligned=True.",
                "_sequence": 7,
            },
        ]
        tools_map = {
            "review_trajectory": FakeTool("review_trajectory"),
            "declare_milestones": FakeTool("declare_milestones"),
        }
        batch = ReviewBatch(config, ledger, tools_map)
        result = FakeToolResult(
            "review_trajectory",
            "Trajectory review: aligned=True.",
            is_success=True,
            tool_call_id="tc_review",
            output={"aligned": True, "experience": ""},
        )
        content = batch.process_result(
            result,
            current_seq=7,
            assistant_step_id="step_review_call",
            tool_step_id="step_review_result",
        )
        assert content == "Trajectory review: aligned=True."
        assert ledger.review.consecutive_errors == 0
        assert ledger.review.last_review_seq == 7
        assert ledger.review.latest_checkpoint is not None
        assert ledger.review.latest_checkpoint.seq == 7
        assert ledger.review.latest_checkpoint.milestone_id == ""
        assert ledger.messages

    @pytest.mark.asyncio
    async def test_finalize_returns_hidden_review_steps_for_aligned_review(self):
        config = AgentOptions(enable_goal_directed_review=True)
        ledger = RunLedger()
        tools_map = {
            "review_trajectory": FakeTool("review_trajectory"),
            "declare_milestones": FakeTool("declare_milestones"),
        }
        batch = ReviewBatch(config, ledger, tools_map)
        batch.process_result(
            FakeToolResult(
                "review_trajectory",
                "Trajectory review: aligned=True.",
                tool_call_id="tc_review",
                output={"aligned": True, "experience": ""},
            ),
            current_seq=7,
            assistant_step_id="step_review_call",
            tool_step_id="step_review_result",
        )

        outcome = await batch.finalize()
        assert outcome.mode == "metadata_only"
        assert outcome.review_tool_call_id == "tc_review"
        assert outcome.hidden_step_ids == ["step_review_call", "step_review_result"]

    @pytest.mark.asyncio
    async def test_process_result_step_back_only_when_not_aligned(self):
        config = AgentOptions(enable_goal_directed_review=True)
        ledger = RunLedger()
        ledger.messages = [
            {
                "role": "tool",
                "tool_call_id": "tc_search",
                "content": "Verbose JWT result",
                "_sequence": 3,
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "tc_review",
                        "type": "function",
                        "function": {"name": "review_trajectory", "arguments": "{}"},
                    }
                ],
                "_sequence": 4,
            },
            {
                "role": "tool",
                "tool_call_id": "tc_review",
                "content": "Trajectory review",
                "_sequence": 5,
            },
        ]
        tools_map = {
            "review_trajectory": FakeTool("review_trajectory"),
            "declare_milestones": FakeTool("declare_milestones"),
        }
        batch = ReviewBatch(config, ledger, tools_map)
        batch.register_step("tc_search", "step_search", 3)
        result = FakeToolResult(
            "review_trajectory",
            "Trajectory review: aligned=False. JWT was a dead end",
            is_success=True,
            tool_call_id="tc_review",
            output={"aligned": False, "experience": "JWT was a dead end"},
        )

        batch.process_result(
            result,
            current_seq=5,
            assistant_step_id="step_review_call",
            tool_step_id="step_review_result",
        )
        outcome = await batch.finalize(
            storage=InMemoryRunLogStorage(),
            session_id="s1",
            run_id="r1",
            agent_id="a1",
        )

        assert outcome.mode == "step_back"
        assert outcome.experience == "JWT was a dead end"
        assert outcome.affected_count == 1
        assert outcome.review_tool_call_id == "tc_review"
        assert outcome.hidden_step_ids == ["step_review_call", "step_review_result"]
        assert [
            (update.step_id, update.tool_call_id) for update in outcome.content_updates
        ] == [("step_search", "tc_search")]

    def test_process_result_injects_review_when_step_interval_triggered(self):
        config = AgentOptions(enable_goal_directed_review=True, review_step_interval=1)
        ledger = RunLedger()
        ledger.review.milestones = [
            Milestone(id="a", description="Find the bug", status="active")
        ]
        tools_map = {
            "review_trajectory": FakeTool("review_trajectory"),
            "declare_milestones": FakeTool("declare_milestones"),
        }
        batch = ReviewBatch(config, ledger, tools_map)
        result = FakeToolResult("search", "Found results", tool_call_id="tc_search")
        content = batch.process_result(result, current_seq=2)
        assert "<system-review>" in content
        assert "</system-review>" in content
        assert "Find the bug" in content

    def test_step_interval_uses_non_review_tool_count_not_sequence_delta(self):
        config = AgentOptions(enable_goal_directed_review=True, review_step_interval=2)
        ledger = RunLedger()
        ledger.review.last_review_seq = 0
        ledger.review.milestones = [
            Milestone(id="a", description="Find the bug", status="active")
        ]
        tools_map = {
            "review_trajectory": FakeTool("review_trajectory"),
            "declare_milestones": FakeTool("declare_milestones"),
        }
        batch = ReviewBatch(config, ledger, tools_map)

        first = batch.process_result(
            FakeToolResult("search", "first", tool_call_id="tc_1"),
            current_seq=100,
        )
        second = batch.process_result(
            FakeToolResult("read", "second", tool_call_id="tc_2"),
            current_seq=101,
        )

        assert "<system-review>" not in first
        assert "<system-review>" in second
        assert ledger.review.review_count_since_checkpoint == 2
        assert "Steps since last review: 2" in second

    def test_review_trajectory_resets_count_and_does_not_count_itself(self):
        config = AgentOptions(enable_goal_directed_review=True, review_step_interval=2)
        ledger = RunLedger()
        tools_map = {
            "review_trajectory": FakeTool("review_trajectory"),
            "declare_milestones": FakeTool("declare_milestones"),
        }
        batch = ReviewBatch(config, ledger, tools_map)
        batch.process_result(FakeToolResult("search", "first"), current_seq=1)

        batch.process_result(
            FakeToolResult(
                "review_trajectory",
                "Trajectory review: aligned=True.",
                tool_call_id="tc_review",
                output={"aligned": True, "experience": ""},
            ),
            current_seq=2,
        )
        content = batch.process_result(FakeToolResult("read", "second"), current_seq=50)

        assert ledger.review.review_count_since_checkpoint == 1
        assert "<system-review>" not in content

    def test_declare_milestones_counts_toward_step_interval(self):
        config = AgentOptions(enable_goal_directed_review=True, review_step_interval=1)
        ledger = RunLedger()
        tools_map = {
            "review_trajectory": FakeTool("review_trajectory"),
            "declare_milestones": FakeTool("declare_milestones"),
        }
        batch = ReviewBatch(config, ledger, tools_map)
        result = FakeToolResult(
            "declare_milestones",
            "Milestones declared: a",
            tool_call_id="tc_declare",
            output={"milestones": [{"id": "a", "description": "Find the bug"}]},
        )

        content = batch.process_result(result, current_seq=4)

        assert "<system-review>" in content
        assert ledger.review.review_count_since_checkpoint == 1
        assert "Find the bug" in content

    def test_process_result_does_not_set_pending_on_milestone_declaration(self):
        config = AgentOptions(enable_goal_directed_review=True)
        ledger = RunLedger()
        tools_map = {
            "review_trajectory": FakeTool("review_trajectory"),
            "declare_milestones": FakeTool("declare_milestones"),
        }
        batch = ReviewBatch(config, ledger, tools_map)
        result = FakeToolResult("declare_milestones", "ok", tool_call_id="tc_declare")
        batch.process_result(result)
        assert ledger.review.pending_review_reason is None

    def test_process_result_declares_milestones_from_tool_output(self):
        config = AgentOptions(enable_goal_directed_review=True)
        ledger = RunLedger()
        tools_map = {
            "review_trajectory": FakeTool("review_trajectory"),
            "declare_milestones": FakeTool("declare_milestones"),
        }
        batch = ReviewBatch(config, ledger, tools_map)
        result = FakeToolResult(
            "declare_milestones",
            "Milestones declared: a",
            tool_call_id="tc_declare",
            output={"milestones": [{"id": "a", "description": "Find the bug"}]},
        )
        batch.process_result(result, current_seq=4)
        assert [(m.id, m.description, m.status) for m in ledger.review.milestones] == [
            ("a", "Find the bug", "active")
        ]
        assert ledger.review.milestones[0].declared_at_seq == 4

    def test_process_result_tracks_consecutive_errors(self):
        config = AgentOptions(enable_goal_directed_review=True)
        ledger = RunLedger()
        tools_map = {
            "review_trajectory": FakeTool("review_trajectory"),
            "declare_milestones": FakeTool("declare_milestones"),
        }
        batch = ReviewBatch(config, ledger, tools_map)
        batch.process_result(FakeToolResult("search", "", is_success=False))
        assert ledger.review.consecutive_errors == 1
        batch.process_result(FakeToolResult("search", "", is_success=False))
        assert ledger.review.consecutive_errors == 2
        batch.process_result(FakeToolResult("search", "ok", is_success=True))
        assert ledger.review.consecutive_errors == 0

    def test_process_result_respects_review_on_error_disabled(self):
        config = AgentOptions(
            enable_goal_directed_review=True,
            review_on_error=False,
            review_step_interval=100,
        )
        ledger = RunLedger()
        tools_map = {
            "review_trajectory": FakeTool("review_trajectory"),
            "declare_milestones": FakeTool("declare_milestones"),
        }
        batch = ReviewBatch(config, ledger, tools_map)
        batch.process_result(
            FakeToolResult("search", "failed once", is_success=False),
            current_seq=1,
        )
        content = batch.process_result(
            FakeToolResult("search", "failed twice", is_success=False),
            current_seq=2,
        )
        assert ledger.review.consecutive_errors == 2
        assert "<system-review>" not in content

    @pytest.mark.asyncio
    async def test_register_step_supports_public_finalize_outcome(self):
        config = AgentOptions(enable_goal_directed_review=True)
        ledger = RunLedger()
        ledger.messages = [
            {
                "role": "tool",
                "tool_call_id": "tc_1",
                "content": "verbose",
                "_sequence": 1,
            }
        ]
        tools_map = {
            "review_trajectory": FakeTool("review_trajectory"),
            "declare_milestones": FakeTool("declare_milestones"),
        }
        batch = ReviewBatch(config, ledger, tools_map)
        batch.register_step("tc_1", "step_1", 5)
        batch.process_result(
            FakeToolResult(
                "review_trajectory",
                "Trajectory review: aligned=False. summarize",
                tool_call_id="tc_review",
                output={"aligned": False, "experience": "summarize"},
            ),
            current_seq=6,
        )
        outcome = await batch.finalize(
            storage=InMemoryRunLogStorage(),
            session_id="s1",
            run_id="r1",
            agent_id="a1",
        )
        assert outcome.mode == "step_back"
        assert [
            (update.step_id, update.content) for update in outcome.content_updates
        ] == [("step_1", "[EXPERIENCE] summarize")]

    @pytest.mark.asyncio
    async def test_finalize_returns_not_applied_when_no_feedback(self):
        config = AgentOptions()
        ledger = RunLedger()
        tools_map = {}
        batch = ReviewBatch(config, ledger, tools_map)
        outcome = await batch.finalize()
        assert outcome.mode == "none"

    @pytest.mark.asyncio
    async def test_malformed_review_result_still_advances_review_cursor(self):
        config = AgentOptions(enable_goal_directed_review=True, review_step_interval=1)
        ledger = RunLedger()
        ledger.review.pending_review_reason = "milestone_switch"
        tools_map = {
            "review_trajectory": FakeTool("review_trajectory"),
            "declare_milestones": FakeTool("declare_milestones"),
        }
        batch = ReviewBatch(config, ledger, tools_map)
        batch.process_result(
            FakeToolResult(
                "review_trajectory",
                "Trajectory review: missing aligned output",
                tool_call_id="tc_review",
                output={},
            ),
            current_seq=9,
            assistant_step_id="step_review_call",
            tool_step_id="step_review_result",
        )

        outcome = await batch.finalize()

        assert outcome.mode == "metadata_only"
        assert outcome.hidden_step_ids == ["step_review_call", "step_review_result"]
        assert ledger.review.last_review_seq == 9
        assert ledger.review.pending_review_reason is None

    @pytest.mark.asyncio
    async def test_finalize_requires_storage_when_feedback_present(self):
        config = AgentOptions(enable_goal_directed_review=True)
        ledger = RunLedger()
        tools_map = {
            "review_trajectory": FakeTool("review_trajectory"),
            "declare_milestones": FakeTool("declare_milestones"),
        }
        batch = ReviewBatch(config, ledger, tools_map)
        batch.process_result(
            FakeToolResult(
                "review_trajectory",
                "Trajectory review: aligned=False. summarize",
                output={"aligned": False, "experience": "summarize"},
            ),
            current_seq=6,
        )

        with pytest.raises(ValueError, match="requires a non-None storage"):
            await batch.finalize()

    @pytest.mark.asyncio
    async def test_system_review_cleanup_skips_updates_without_resolved_step_id(self):
        updates = await _build_system_review_cleanup_updates(
            [
                {
                    "role": "tool",
                    "tool_call_id": "tc_missing",
                    "content": "result\n<system-review>review</system-review>",
                }
            ],
            storage=InMemoryRunLogStorage(),
            session_id="s1",
            run_id="r1",
            agent_id="a1",
            review_tool_call_id=None,
        )

        assert updates == []

    def test_agent_options_rejects_invalid_review_step_interval(self):
        with pytest.raises(ValueError):
            AgentOptions(review_step_interval=0)
