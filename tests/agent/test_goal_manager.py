from agiwo.agent.models.review import Milestone, ReviewState
from agiwo.agent.review.goal_manager import (
    declare_milestones,
    activate_next_milestone,
    complete_active_milestone,
    get_active_milestone,
)


class TestDeclareMilestones:
    def test_first_declaration_sets_first_active(self):
        state = ReviewState()
        milestones = [
            Milestone(id="a", description="Step A"),
            Milestone(id="b", description="Step B"),
        ]
        result = declare_milestones(state, milestones)
        assert len(state.milestones) == 2
        assert state.milestones[0].status == "active"
        assert state.milestones[1].status == "pending"
        assert result == ["a", "b"]

    def test_redeclare_preserves_previous_if_no_active_change(self):
        m = Milestone(id="a", description="Step A", status="completed")
        m2 = Milestone(id="b", description="Step B", status="active")
        state = ReviewState(milestones=[m, m2])
        result = declare_milestones(state, [
            Milestone(id="c", description="Step C"),
        ])
        # state unchanged because active milestone was already active
        assert len(state.milestones) == 3
        assert result == ["c"]


class TestCompleteActiveMilestone:
    def test_complete_active(self):
        m = Milestone(id="a", description="Step A", status="active")
        state = ReviewState(milestones=[m])
        result = complete_active_milestone(state, seq=10)
        assert result is True
        assert state.milestones[0].status == "completed"
        assert state.milestones[0].completed_at_seq == 10

    def test_complete_no_active_returns_false(self):
        state = ReviewState()
        result = complete_active_milestone(state, seq=10)
        assert result is False


class TestActivateNextMilestone:
    def test_activate_first_pending(self):
        m1 = Milestone(id="a", description="A", status="completed")
        m2 = Milestone(id="b", description="B", status="pending")
        state = ReviewState(milestones=[m1, m2])
        result = activate_next_milestone(state)
        assert result is not None
        assert result.id == "b"
        assert result.status == "active"

    def test_activate_none_pending_returns_none(self):
        m = Milestone(id="a", description="A", status="completed")
        state = ReviewState(milestones=[m])
        result = activate_next_milestone(state)
        assert result is None


class TestGetActiveMilestone:
    def test_returns_active(self):
        m = Milestone(id="a", description="A", status="active")
        state = ReviewState(milestones=[m])
        result = get_active_milestone(state)
        assert result is not None
        assert result.id == "a"

    def test_returns_none_when_none_active(self):
        m = Milestone(id="a", description="A", status="completed")
        state = ReviewState(milestones=[m])
        result = get_active_milestone(state)
        assert result is None
