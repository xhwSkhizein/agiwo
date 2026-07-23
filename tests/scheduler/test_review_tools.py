# tests/scheduler/test_review_tools.py
"""Scheduler tests for plan tools; review tool moved to agent introspect."""

from agiwo.agent.plan import UpdatePlanTool


class TestUpdatePlanToolSchedulerSurface:
    def test_update_plan_available_from_agent_plan(self):
        tool = UpdatePlanTool()
        assert tool.name == "update_plan"
