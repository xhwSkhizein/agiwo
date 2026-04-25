from datetime import datetime, timezone

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

        default_cp = ReviewCheckpoint(seq=11, milestone_id="diagnose")
        assert default_cp.confirmed_at.tzinfo is timezone.utc
        delta = datetime.now(timezone.utc) - default_cp.confirmed_at
        assert abs(delta.total_seconds()) < 1


class TestReviewState:
    def test_review_state_defaults(self):
        rs = ReviewState()
        assert rs.milestones == []
        assert rs.last_review_seq == 0
        assert rs.latest_checkpoint is None
        assert rs.consecutive_errors == 0
        assert rs.pending_review_reason is None

    def test_review_state_with_milestones(self):
        m = Milestone(id="a", description="desc", status="active")
        checkpoint = ReviewCheckpoint(seq=3, milestone_id="a")
        rs = ReviewState(
            milestones=[m],
            last_review_seq=5,
            latest_checkpoint=checkpoint,
            consecutive_errors=2,
            pending_review_reason="milestone_switch",
        )
        assert len(rs.milestones) == 1
        assert rs.last_review_seq == 5
        assert rs.latest_checkpoint == checkpoint
        assert rs.consecutive_errors == 2
        assert rs.pending_review_reason == "milestone_switch"
