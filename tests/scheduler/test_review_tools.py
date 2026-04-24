# tests/scheduler/test_review_tools.py
import pytest
from unittest.mock import MagicMock
from agiwo.scheduler.runtime_tools import (
    DeclareMilestonesTool,
    ReviewTrajectoryTool,
)
from agiwo.tool.base import ToolContext
from agiwo.scheduler.tool_control import SchedulerToolControl


class TestDeclareMilestonesTool:
    def test_name_and_description(self):
        tool = DeclareMilestonesTool(MagicMock(spec=SchedulerToolControl))
        assert tool.name == "declare_milestones"
        assert "milestones" in tool.description.lower()

    def test_parameters_schema(self):
        tool = DeclareMilestonesTool(MagicMock(spec=SchedulerToolControl))
        params = tool.get_parameters()
        assert params["type"] == "object"
        assert "milestones" in params["properties"]
        assert "milestones" in params["required"]

    @pytest.mark.asyncio
    async def test_execute_success(self):
        tool = DeclareMilestonesTool(MagicMock(spec=SchedulerToolControl))
        result = await tool.execute(
            parameters={
                "milestones": [
                    {"id": "a", "description": "Step A"},
                    {"id": "b", "description": "Step B"},
                ]
            },
            context=ToolContext(session_id="s1", tool_call_id="tc_1"),
        )
        assert result.is_success
        assert "a" in result.content
        assert "b" in result.content

    @pytest.mark.asyncio
    async def test_execute_empty_milestones(self):
        tool = DeclareMilestonesTool(MagicMock(spec=SchedulerToolControl))
        result = await tool.execute(
            parameters={"milestones": []},
            context=ToolContext(session_id="s1", tool_call_id="tc_1"),
        )
        assert not result.is_success


class TestReviewTrajectoryTool:
    def test_name_and_description(self):
        tool = ReviewTrajectoryTool(MagicMock(spec=SchedulerToolControl))
        assert tool.name == "review_trajectory"
        assert "system-review" in tool.description.lower()

    def test_parameters_schema(self):
        tool = ReviewTrajectoryTool(MagicMock(spec=SchedulerToolControl))
        params = tool.get_parameters()
        assert params["type"] == "object"
        assert "aligned" in params["properties"]
        assert "experience" in params["properties"]
        assert "aligned" in params["required"]

    @pytest.mark.asyncio
    async def test_execute_aligned_true(self):
        tool = ReviewTrajectoryTool(MagicMock(spec=SchedulerToolControl))
        result = await tool.execute(
            parameters={"aligned": True},
            context=ToolContext(session_id="s1", tool_call_id="tc_r"),
        )
        assert result.is_success

    @pytest.mark.asyncio
    async def test_execute_aligned_false_requires_experience(self):
        tool = ReviewTrajectoryTool(MagicMock(spec=SchedulerToolControl))
        result = await tool.execute(
            parameters={
                "aligned": False,
                "experience": "Tried X, learned Y, will do Z next.",
            },
            context=ToolContext(session_id="s1", tool_call_id="tc_r"),
        )
        assert result.is_success
        assert "Tried X" in result.content

    @pytest.mark.asyncio
    async def test_execute_aligned_false_without_experience(self):
        tool = ReviewTrajectoryTool(MagicMock(spec=SchedulerToolControl))
        result = await tool.execute(
            parameters={"aligned": False},
            context=ToolContext(session_id="s1", tool_call_id="tc_r"),
        )
        assert not result.is_success
