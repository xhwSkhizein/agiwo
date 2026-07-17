# tests/agent/test_review_trajectory_tool.py
import pytest

from agiwo.agent.introspect.tool import ReviewTrajectoryTool
from agiwo.agent.plan import UpdatePlanTool
from agiwo.tool.base import ToolContext


class TestUpdatePlanTool:
    def test_name_and_description(self):
        tool = UpdatePlanTool()
        assert tool.name == "update_plan"
        assert "plan" in tool.description.lower()


class TestReviewTrajectoryTool:
    def test_name_and_description(self):
        tool = ReviewTrajectoryTool()
        assert tool.name == "review_trajectory"
        assert "system-review" in tool.description.lower()

    def test_parameters_schema(self):
        tool = ReviewTrajectoryTool()
        params = tool.get_parameters()
        assert params["type"] == "object"
        assert "aligned" in params["properties"]
        assert "experience" in params["properties"]
        assert "tool_usefulness" in params["properties"]
        assert "aligned" in params["required"]

    @pytest.mark.asyncio
    async def test_execute_aligned_true(self):
        tool = ReviewTrajectoryTool()
        result = await tool.execute(
            parameters={"aligned": True},
            context=ToolContext(
                session_id="s1",
                tool_call_id="tc_r",
                metadata={
                    "review_window": [
                        {"tool_call_id": "tc_search", "tool_name": "search"}
                    ]
                },
            ),
        )
        assert result.is_success
        assert result.output["aligned"] is True
        assert result.output["unknown_tool_call_ids"] == ["tc_search"]

    @pytest.mark.asyncio
    async def test_execute_scores_and_unknown(self):
        tool = ReviewTrajectoryTool()
        result = await tool.execute(
            parameters={
                "aligned": False,
                "experience": "Search was too broad.",
                "tool_usefulness": [
                    {"tool_call_id": "tc_search", "score": 2},
                    {"tool_call_id": "tc_dup", "score": 1},
                    {"tool_call_id": "tc_dup", "score": 3},
                    {"tool_call_id": "tc_out", "score": 2},
                    {"tool_call_id": "tc_bad", "score": 9},
                ],
            },
            context=ToolContext(
                session_id="s1",
                tool_call_id="tc_r",
                metadata={
                    "review_window": [
                        {"tool_call_id": "tc_search", "tool_name": "search"},
                        {"tool_call_id": "tc_dup", "tool_name": "bash"},
                        {"tool_call_id": "tc_missing", "tool_name": "read"},
                        {"tool_call_id": "tc_bad", "tool_name": "write"},
                    ]
                },
            ),
        )
        assert result.is_success
        usefulness = {
            item["tool_call_id"]: item["score"]
            for item in result.output["tool_usefulness"]
        }
        assert usefulness["tc_search"] == 2
        assert usefulness["tc_dup"] == 1
        assert usefulness["tc_missing"] is None
        assert "tc_out" not in usefulness
        assert "tc_bad" not in usefulness
        assert any("duplicate" in item for item in result.output["rejected_scores"])
        assert any("out-of-window" in item for item in result.output["rejected_scores"])
        assert any("illegal score" in item for item in result.output["rejected_scores"])
