"""Tests for the metrics collector."""


from swarmpr.metrics.collector import (
    MetricsCollector,
)
from swarmpr.orchestrator.state import (
    AgentResult,
    AgentStatus,
    FileChange,
    FileChangeType,
    PipelineStage,
    PipelineState,
    PipelineStatus,
    ReviewVerdict,
    TaskPlan,
)


def _make_completed_state() -> PipelineState:
    """Create a state with agent results for metrics testing.

    Returns:
        A PipelineState with populated agent results.
    """
    state = PipelineState(
        task_description="Fix null check",
        repo_path="/tmp/repo",
        status=PipelineStatus.COMPLETED,
        tests_passed=True,
        plan=TaskPlan(
            task_description="Fix null check",
            files=[
                FileChange(
                    path="payments/validator.py",
                    change_type=FileChangeType.MODIFY,
                    description="Fix",
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
            summary="Payment logic modified.",
            findings=["Payment validation changed"],
        ),
    )

    state.agent_results = [
        AgentResult(
            agent=PipelineStage.PLANNER,
            status=AgentStatus.COMPLETED,
            duration_ms=1500,
            tokens_in=200,
            tokens_out=150,
        ),
        AgentResult(
            agent=PipelineStage.CODER,
            status=AgentStatus.COMPLETED,
            duration_ms=3000,
            tokens_in=500,
            tokens_out=800,
        ),
        AgentResult(
            agent=PipelineStage.TESTER,
            status=AgentStatus.COMPLETED,
            duration_ms=500,
            tokens_in=50,
            tokens_out=30,
        ),
        AgentResult(
            agent=PipelineStage.REVIEWER,
            status=AgentStatus.COMPLETED,
            duration_ms=2000,
            tokens_in=400,
            tokens_out=200,
        ),
    ]

    return state


class TestMetricsCollector:
    """Tests for the MetricsCollector."""

    def test_collect_totals(self):
        """Test that metrics totals are correctly aggregated."""
        state = _make_completed_state()
        collector = MetricsCollector()
        metrics = collector.collect(state)

        assert metrics.total_duration_ms == 7000
        assert metrics.total_tokens_in == 1150
        assert metrics.total_tokens_out == 1180
        assert metrics.total_tokens == 2330

    def test_collect_per_agent(self):
        """Test that per-agent metrics are recorded."""
        state = _make_completed_state()
        collector = MetricsCollector()
        metrics = collector.collect(state)

        assert len(metrics.agents) == 4
        assert metrics.agents[0].agent == "planner"
        assert metrics.agents[1].agent == "coder"
        assert metrics.agents[2].agent == "tester"
        assert metrics.agents[3].agent == "reviewer"

    def test_collect_risk_info(self):
        """Test that risk information is captured."""
        state = _make_completed_state()
        collector = MetricsCollector()
        metrics = collector.collect(state)

        assert metrics.risk_tier == 3
        assert metrics.risk_score == 7.5
        assert metrics.escalation_action == "block"
        assert metrics.tests_passed is True

    def test_collect_cost_estimate(self):
        """Test that cost estimates are non-zero."""
        state = _make_completed_state()
        collector = MetricsCollector()
        metrics = collector.collect(state)

        assert metrics.estimated_cost_usd > 0

    def test_collect_empty_state(self):
        """Test collecting metrics from a state with no results."""
        state = PipelineState(
            task_description="Test",
            repo_path="/tmp",
        )
        collector = MetricsCollector()
        metrics = collector.collect(state)

        assert metrics.total_duration_ms == 0
        assert metrics.total_tokens == 0
        assert len(metrics.agents) == 0

    def test_format_summary(self):
        """Test that format_summary produces readable output."""
        state = _make_completed_state()
        collector = MetricsCollector()
        metrics = collector.collect(state)
        summary = collector.format_summary(metrics)

        assert "Pipeline Metrics" in summary
        assert "Total Duration" in summary
        assert "planner" in summary
        assert "coder" in summary
