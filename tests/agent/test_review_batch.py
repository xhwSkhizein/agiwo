# tests/agent/test_review_batch.py
import pytest
from agiwo.agent.models.review import Milestone
from agiwo.agent.models.config import AgentOptions
from agiwo.agent.models.run import RunLedger
from agiwo.agent.review import ReviewBatch


class FakeTool:
    def __init__(self, name):
        self.name = name


class FakeToolResult:
    def __init__(self, tool_name, content, is_success=True, tool_call_id="tc_1"):
        self.tool_name = tool_name
        self.content = content
        self.is_success = is_success
        self.tool_call_id = tool_call_id
        self.input_args = {}
        self.content_for_user = None
        self.termination_reason = None
        self.output = {}


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

    def test_process_result_captures_review_feedback(self):
        config = AgentOptions(enable_goal_directed_review=True)
        ledger = RunLedger()
        tools_map = {
            "review_trajectory": FakeTool("review_trajectory"),
            "declare_milestones": FakeTool("declare_milestones"),
        }
        batch = ReviewBatch(config, ledger, tools_map)
        result = FakeToolResult(
            "review_trajectory",
            "JWT was a dead end",
            is_success=True,
            tool_call_id="tc_review",
        )
        content = batch.process_result(result)
        assert content == "JWT was a dead end"
        assert batch._feedback == "JWT was a dead end"

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
        assert "<system-notice>" in content
        assert "Find the bug" in content

    def test_process_result_sets_review_pending_on_milestone_declaration(self):
        config = AgentOptions(enable_goal_directed_review=True)
        ledger = RunLedger()
        tools_map = {
            "review_trajectory": FakeTool("review_trajectory"),
            "declare_milestones": FakeTool("declare_milestones"),
        }
        batch = ReviewBatch(config, ledger, tools_map)
        result = FakeToolResult("declare_milestones", "ok", tool_call_id="tc_declare")
        batch.process_result(result)
        assert ledger.review.is_review_pending is True

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

    def test_register_step_stores_entry(self):
        config = AgentOptions()
        ledger = RunLedger()
        tools_map = {}
        batch = ReviewBatch(config, ledger, tools_map)
        batch.register_step("tc_1", "step_1", 5)
        assert batch._step_lookup["tc_1"] == {"id": "step_1", "sequence": 5}

    @pytest.mark.asyncio
    async def test_finalize_returns_not_applied_when_no_feedback(self):
        config = AgentOptions()
        ledger = RunLedger()
        tools_map = {}
        batch = ReviewBatch(config, ledger, tools_map)
        outcome = await batch.finalize()
        assert outcome.applied is False
