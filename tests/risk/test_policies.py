"""Tests for risk escalation policies."""

import pytest

from swarmpr.config import PipelineConfig, RiskTierConfig
from swarmpr.risk.classifier import ClassificationResult
from swarmpr.risk.policies import EscalationDecision, EscalationPolicy


@pytest.fixture
def risk_tiers() -> dict[str, RiskTierConfig]:
    """Return standard risk tier configurations.

    Returns:
        A dictionary of tier name to RiskTierConfig.
    """
    return {
        "tier_3": RiskTierConfig(
            description="Critical path",
            paths=["payments/", "auth/"],
            keywords=["secret"],
            action="block",
        ),
        "tier_2": RiskTierConfig(
            description="Business logic",
            paths=["api/"],
            keywords=[],
            action="flag",
        ),
        "tier_1": RiskTierConfig(
            description="Low risk",
            paths=["config/", "tests/"],
            keywords=[],
            action="approve",
        ),
    }


@pytest.fixture
def pipeline_config() -> PipelineConfig:
    """Return a standard pipeline configuration.

    Returns:
        A PipelineConfig with default forbidden paths.
    """
    return PipelineConfig(
        max_diff_lines=500,
        test_timeout_seconds=120,
        forbidden_paths=[".env", "secrets/", "credentials/"],
    )


class TestEscalationDecision:
    """Tests for the EscalationDecision model."""

    def test_create_approve_decision(self):
        """Test creating an approve decision."""
        decision = EscalationDecision(
            action="approve",
            tier=1,
            can_auto_merge=True,
            requires_human_review=False,
            reason="Config-only change, low risk.",
        )
        assert decision.can_auto_merge is True
        assert decision.requires_human_review is False

    def test_create_block_decision(self):
        """Test creating a block decision."""
        decision = EscalationDecision(
            action="block",
            tier=3,
            can_auto_merge=False,
            requires_human_review=True,
            reason="Touches payment processing logic.",
        )
        assert decision.can_auto_merge is False
        assert decision.requires_human_review is True


class TestEscalationPolicy:
    """Tests for the EscalationPolicy engine."""

    def test_tier_1_auto_approves(self, risk_tiers, pipeline_config):
        """Test that tier 1 classification results in auto-approval."""
        policy = EscalationPolicy(risk_tiers, pipeline_config)
        classification = ClassificationResult(
            tier=1,
            action="approve",
            justification="Config change only",
            matched_rules=["path: config/"],
        )
        decision = policy.evaluate(classification, diff_lines=10)
        assert decision.action == "approve"
        assert decision.can_auto_merge is True
        assert decision.requires_human_review is False

    def test_tier_2_flags_for_review(self, risk_tiers, pipeline_config):
        """Test that tier 2 classification flags for optional review."""
        policy = EscalationPolicy(risk_tiers, pipeline_config)
        classification = ClassificationResult(
            tier=2,
            action="flag",
            justification="API endpoint change",
            matched_rules=["path: api/"],
        )
        decision = policy.evaluate(classification, diff_lines=50)
        assert decision.action == "flag"
        assert decision.can_auto_merge is False
        assert decision.requires_human_review is False

    def test_tier_3_blocks_for_human_review(self, risk_tiers, pipeline_config):
        """Test that tier 3 classification requires human review."""
        policy = EscalationPolicy(risk_tiers, pipeline_config)
        classification = ClassificationResult(
            tier=3,
            action="block",
            justification="Payment logic modified",
            matched_rules=["path: payments/"],
        )
        decision = policy.evaluate(classification, diff_lines=30)
        assert decision.action == "block"
        assert decision.can_auto_merge is False
        assert decision.requires_human_review is True

    def test_large_diff_escalates(self, risk_tiers, pipeline_config):
        """Test that exceeding max diff lines escalates the decision."""
        policy = EscalationPolicy(risk_tiers, pipeline_config)
        classification = ClassificationResult(
            tier=1,
            action="approve",
            justification="Config change",
            matched_rules=["path: config/"],
        )
        # Diff exceeds max_diff_lines (500)
        decision = policy.evaluate(classification, diff_lines=600)
        assert decision.requires_human_review is True
        assert "diff size" in decision.reason.lower()

    def test_forbidden_path_blocks(self, risk_tiers, pipeline_config):
        """Test that touching a forbidden path always blocks."""
        policy = EscalationPolicy(risk_tiers, pipeline_config)
        classification = ClassificationResult(
            tier=1,
            action="approve",
            justification="Config change",
            matched_rules=["path: config/"],
        )
        decision = policy.evaluate(
            classification,
            diff_lines=10,
            changed_paths=[".env", "config/settings.py"],
        )
        assert decision.action == "block"
        assert decision.requires_human_review is True
        assert "forbidden" in decision.reason.lower()

    def test_decision_includes_reason(self, risk_tiers, pipeline_config):
        """Test that all decisions include a human-readable reason."""
        policy = EscalationPolicy(risk_tiers, pipeline_config)
        classification = ClassificationResult(
            tier=2,
            action="flag",
            justification="API change",
            matched_rules=["path: api/"],
        )
        decision = policy.evaluate(classification, diff_lines=20)
        assert decision.reason != ""
