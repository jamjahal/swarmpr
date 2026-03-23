"""Tests for GitHub PR creation."""



from swarmpr.github.pr import PRCreator
from swarmpr.orchestrator.state import (
    FileChange,
    FileChangeType,
    PipelineState,
    ReviewVerdict,
    TaskPlan,
)


def _make_completed_state() -> PipelineState:
    """Create a pipeline state representing a completed run.

    Returns:
        A PipelineState with all fields populated.
    """
    return PipelineState(
        task_description="Fix null check in payment validation",
        repo_path="/tmp/repo",
        branch_name="swarmpr/fix-null-check",
        diff="--- a/payments/validator.py\n+++ b/payments/validator.py",
        tests_passed=True,
        test_output="3 passed in 0.5s",
        plan=TaskPlan(
            task_description="Fix null check",
            files=[
                FileChange(
                    path="payments/validator.py",
                    change_type=FileChangeType.MODIFY,
                    description="Add null check",
                ),
            ],
            risk_tier=3,
            risk_justification="Payment path",
            estimated_complexity="low",
        ),
        review=ReviewVerdict(
            approved=False,
            risk_score=7.5,
            escalation_action="block",
            summary="Payment logic modified. Human review required.",
            findings=["Payment validation changed"],
        ),
    )


class TestPRCreator:
    """Tests for the PRCreator class."""

    def test_build_pr_title(self):
        """Test generating a PR title from the task description."""
        creator = PRCreator(token="fake-token")
        state = _make_completed_state()
        title = creator.build_title(state)
        assert len(title) <= 70
        assert "null check" in title.lower() or "fix" in title.lower()

    def test_build_pr_body(self):
        """Test generating a PR body with all sections."""
        creator = PRCreator(token="fake-token")
        state = _make_completed_state()
        body = creator.build_body(state)

        # Should contain key sections.
        assert "## Summary" in body or "## Changes" in body
        assert "Risk" in body
        assert "Test" in body
        assert "payments/validator.py" in body

    def test_build_pr_body_includes_risk_tier(self):
        """Test that the PR body shows the risk tier."""
        creator = PRCreator(token="fake-token")
        state = _make_completed_state()
        body = creator.build_body(state)
        assert "Tier 3" in body or "tier 3" in body or "tier_3" in body

    def test_build_pr_body_includes_review_verdict(self):
        """Test that the PR body shows the review verdict."""
        creator = PRCreator(token="fake-token")
        state = _make_completed_state()
        body = creator.build_body(state)
        assert "block" in body.lower() or "human review" in body.lower()

    def test_build_pr_body_includes_test_results(self):
        """Test that the PR body shows test results."""
        creator = PRCreator(token="fake-token")
        state = _make_completed_state()
        body = creator.build_body(state)
        assert "passed" in body.lower()

    def test_get_labels_tier_3(self):
        """Test that tier 3 gets appropriate labels."""
        creator = PRCreator(token="fake-token")
        state = _make_completed_state()
        labels = creator.get_labels(state)
        assert "risk:high" in labels or "tier-3" in labels
        assert "swarmpr" in labels

    def test_get_labels_tier_1(self):
        """Test that tier 1 gets auto-approve label."""
        creator = PRCreator(token="fake-token")
        state = _make_completed_state()
        state.plan.risk_tier = 1
        state.review.escalation_action = "approve"
        labels = creator.get_labels(state)
        assert "risk:low" in labels or "tier-1" in labels

    def test_is_draft_for_blocked_pr(self):
        """Test that blocked PRs are created as drafts."""
        creator = PRCreator(token="fake-token")
        state = _make_completed_state()
        assert creator.should_be_draft(state) is True

    def test_is_not_draft_for_approved_pr(self):
        """Test that approved PRs are not drafts."""
        creator = PRCreator(token="fake-token")
        state = _make_completed_state()
        state.review.approved = True
        state.review.escalation_action = "approve"
        assert creator.should_be_draft(state) is False

    def test_build_pr_body_includes_metrics(self):
        """Test that the PR body includes pipeline metrics."""
        creator = PRCreator(token="fake-token")
        state = _make_completed_state()
        body = creator.build_body(state)
        assert "SwarmPR" in body or "swarmpr" in body.lower()
