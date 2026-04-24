# tests/agent/test_review_models.py
from datetime import datetime
from agiwo.agent.models.review import Milestone, ReviewCheckpoint, ReviewState


class TestMilestone:
    def test_milestone_creation(self):
        m = Milestone(
            id="understand",
            description="Understand session management",
            status="pending",
            declared_at_seq=5,
        )
        assert m.id == "understand"
        assert m.description == "Understand session management"
        assert m.status == "pending"
        assert m.declared_at_seq == 5
        assert m.completed_at_seq is None

    def test_milestone_defaults(self):
        m = Milestone(id="fix", description="Fix the bug", status="pending")
        assert m.declared_at_seq == 0
        assert m.completed_at_seq is None

    def test_milestone_equality(self):
        m1 = Milestone(id="a", description="desc a", status="pending")
        m2 = Milestone(id="a", description="desc a", status="pending")
        assert m1 == m2


class TestReviewCheckpoint:
    def test_checkpoint_creation(self):
        now = datetime.now()
        cp = ReviewCheckpoint(
            seq=10,
            milestone_id="understand",
            confirmed_at=now,
        )
        assert cp.seq == 10
        assert cp.milestone_id == "understand"
        assert cp.confirmed_at == now


class TestReviewState:
    def test_review_state_defaults(self):
        rs = ReviewState()
        assert rs.milestones == []
        assert rs.last_review_seq == 0
        assert rs.last_checkpoint_seq == 0
        assert rs.consecutive_errors == 0
        assert rs.is_review_pending is False

    def test_review_state_with_milestones(self):
        m = Milestone(id="a", description="desc", status="active")
        rs = ReviewState(
            milestones=[m],
            last_review_seq=5,
            last_checkpoint_seq=3,
            consecutive_errors=2,
            is_review_pending=True,
        )
        assert len(rs.milestones) == 1
        assert rs.last_review_seq == 5
        assert rs.last_checkpoint_seq == 3
        assert rs.consecutive_errors == 2
        assert rs.is_review_pending is True
