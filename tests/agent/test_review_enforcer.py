# tests/agent/test_review_enforcer.py
import pytest
from agiwo.agent.models.review import Milestone, ReviewState
from agiwo.agent.review.review_enforcer import (
    ReviewTrigger,
    check_review_trigger,
    inject_system_review,
)


class TestCheckReviewTrigger:
    def test_disabled_returns_none(self):
        state = ReviewState()
        trigger = check_review_trigger(
            state=state,
            enabled=False,
            is_error=False,
            step_interval=8,
            error_threshold=2,
        )
        assert trigger == ReviewTrigger.NONE

    def test_error_consecutive_trigger(self):
        state = ReviewState(consecutive_errors=2)
        trigger = check_review_trigger(
            state=state,
            enabled=True,
            is_error=True,
            step_interval=8,
            error_threshold=2,
        )
        assert trigger == ReviewTrigger.CONSECUTIVE_ERRORS

    def test_step_interval_trigger(self):
        state = ReviewState(
            last_review_seq=5,
            last_checkpoint_seq=5,
            consecutive_errors=0,
        )
        # current_seq=14, last_review_seq=5, diff=9 >= interval=8
        trigger = check_review_trigger(
            state=state,
            enabled=True,
            is_error=False,
            step_interval=8,
            error_threshold=2,
            current_seq=14,
        )
        assert trigger == ReviewTrigger.STEP_INTERVAL

    def test_pending_review_trigger(self):
        state = ReviewState(is_review_pending=True)
        trigger = check_review_trigger(
            state=state,
            enabled=True,
            is_error=False,
            step_interval=8,
            error_threshold=2,
        )
        assert trigger == ReviewTrigger.MILESTONE_SWITCH

    def test_no_trigger_for_review_tool_itself(self):
        state = ReviewState(is_review_pending=True)
        trigger = check_review_trigger(
            state=state,
            enabled=True,
            is_error=False,
            step_interval=8,
            error_threshold=2,
            tool_name="review_trajectory",
        )
        assert trigger == ReviewTrigger.NONE

    def test_below_interval_no_trigger(self):
        state = ReviewState(last_review_seq=5, last_checkpoint_seq=5)
        trigger = check_review_trigger(
            state=state,
            enabled=True,
            is_error=False,
            step_interval=8,
            error_threshold=2,
            current_seq=7,
        )
        assert trigger == ReviewTrigger.NONE


class TestInjectSystemReview:
    def test_injects_review_with_milestone(self):
        content = "Tool result content"
        milestone = Milestone(id="locate", description="定位超时根因", status="active")
        result = inject_system_review(content, milestone, step_count=3)
        assert "<system-notice>" in result
        assert content in result
        assert "定位超时根因" in result
        assert "review_trajectory" in result

    def test_injects_review_without_milestone(self):
        content = "Tool result content"
        result = inject_system_review(content, None, step_count=5)
        assert "<system-notice>" in result
        assert "No active milestone" in result
