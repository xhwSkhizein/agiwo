import pytest

from agiwo.agent.models.config import AgentOptions
from agiwo.agent.models.review import Milestone
from agiwo.agent.models.run import RunLedger
from agiwo.agent.review import ReviewBatch
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
        content = batch.process_result(result, current_seq=7)
        assert content == "Trajectory review: aligned=True."
        assert ledger.review.consecutive_errors == 0
        assert ledger.review.last_review_seq == 7
        assert ledger.review.last_checkpoint_seq == 7

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

        batch.process_result(result, current_seq=5)
        outcome = await batch.finalize(
            storage=InMemoryRunLogStorage(),
            session_id="s1",
            run_id="r1",
            agent_id="a1",
        )

        assert outcome.applied is True
        assert outcome.experience == "JWT was a dead end"
        assert outcome.affected_count == 1
        assert all(msg.get("tool_call_id") != "tc_review" for msg in outcome.messages)

    def test_process_result_injects_review_when_step_interval_triggered(self):
        config = AgentOptions(enable_goal_directed_review=True, review_step_interval=1)
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
        result = FakeToolResult("search", "Found results", tool_call_id="tc_search")
        content = batch.process_result(result, current_seq=2)
        assert "<system-review>" in content
        assert "</system-review>" in content
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
        assert ledger.review.is_review_pending is False

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
        assert outcome.applied is True
        assert outcome.messages[0]["content"] == "[EXPERIENCE] summarize"

    @pytest.mark.asyncio
    async def test_finalize_returns_not_applied_when_no_feedback(self):
        config = AgentOptions()
        ledger = RunLedger()
        tools_map = {}
        batch = ReviewBatch(config, ledger, tools_map)
        outcome = await batch.finalize()
        assert outcome.applied is False

    def test_agent_options_rejects_invalid_review_step_interval(self):
        with pytest.raises(ValueError):
            AgentOptions(review_step_interval=0)
