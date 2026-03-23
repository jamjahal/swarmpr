"""Tests for the blast radius classification engine."""

import pytest

from swarmpr.config import RiskTierConfig
from swarmpr.risk.classifier import RiskClassifier


@pytest.fixture
def risk_tiers() -> dict[str, RiskTierConfig]:
    """Return a standard set of risk tier configurations for testing.

    Returns:
        A dictionary of tier name to RiskTierConfig.
    """
    return {
        "tier_3": RiskTierConfig(
            description="Critical path",
            paths=["payments/", "auth/", "crypto/"],
            keywords=["api_key", "secret", "password", "pci"],
            action="block",
        ),
        "tier_2": RiskTierConfig(
            description="Business logic",
            paths=["api/", "services/", "models/"],
            keywords=[],
            action="flag",
        ),
        "tier_1": RiskTierConfig(
            description="Low risk",
            paths=["config/", "docs/", "tests/", ".github/"],
            keywords=[],
            action="approve",
        ),
    }


class TestRiskClassifierPathMatching:
    """Tests for path-based risk classification."""

    def test_tier_3_payment_path(self, risk_tiers):
        """Test that payment paths are classified as tier 3."""
        classifier = RiskClassifier(risk_tiers)
        result = classifier.classify_paths(["payments/processor.py"])
        assert result.tier == 3
        assert result.action == "block"

    def test_tier_3_auth_path(self, risk_tiers):
        """Test that auth paths are classified as tier 3."""
        classifier = RiskClassifier(risk_tiers)
        result = classifier.classify_paths(["auth/token_manager.py"])
        assert result.tier == 3

    def test_tier_2_api_path(self, risk_tiers):
        """Test that API paths are classified as tier 2."""
        classifier = RiskClassifier(risk_tiers)
        result = classifier.classify_paths(["api/endpoints.py"])
        assert result.tier == 2
        assert result.action == "flag"

    def test_tier_1_config_path(self, risk_tiers):
        """Test that config paths are classified as tier 1."""
        classifier = RiskClassifier(risk_tiers)
        result = classifier.classify_paths(["config/settings.py"])
        assert result.tier == 1
        assert result.action == "approve"

    def test_tier_1_test_path(self, risk_tiers):
        """Test that test paths are classified as tier 1."""
        classifier = RiskClassifier(risk_tiers)
        result = classifier.classify_paths(["tests/test_payments.py"])
        assert result.tier == 1

    def test_highest_tier_wins_with_mixed_paths(self, risk_tiers):
        """Test that the highest tier is returned when paths span tiers."""
        classifier = RiskClassifier(risk_tiers)
        result = classifier.classify_paths([
            "config/settings.py",
            "payments/processor.py",
        ])
        assert result.tier == 3
        assert result.action == "block"

    def test_unknown_path_defaults_to_tier_2(self, risk_tiers):
        """Test that paths not matching any tier default to tier 2."""
        classifier = RiskClassifier(risk_tiers)
        result = classifier.classify_paths(["utils/helpers.py"])
        assert result.tier == 2
        assert result.action == "flag"

    def test_empty_paths_defaults_to_tier_1(self, risk_tiers):
        """Test that empty path list defaults to tier 1."""
        classifier = RiskClassifier(risk_tiers)
        result = classifier.classify_paths([])
        assert result.tier == 1

    def test_nested_path_matches_prefix(self, risk_tiers):
        """Test that deeply nested paths match their prefix tier."""
        classifier = RiskClassifier(risk_tiers)
        result = classifier.classify_paths(
            ["payments/stripe/webhooks/handler.py"]
        )
        assert result.tier == 3


class TestRiskClassifierKeywordMatching:
    """Tests for keyword-based risk classification."""

    def test_keyword_escalates_tier(self, risk_tiers):
        """Test that a tier 3 keyword in content escalates to tier 3."""
        classifier = RiskClassifier(risk_tiers)
        result = classifier.classify_content(
            "config/settings.py",
            'API_KEY = "sk-1234567890"',
        )
        assert result.tier == 3

    def test_password_keyword_triggers_tier_3(self, risk_tiers):
        """Test that password keyword in any file triggers tier 3."""
        classifier = RiskClassifier(risk_tiers)
        result = classifier.classify_content(
            "utils/helpers.py",
            "def hash_password(raw_password: str) -> str:",
        )
        assert result.tier == 3

    def test_no_keywords_no_escalation(self, risk_tiers):
        """Test that content without keywords doesn't trigger escalation."""
        classifier = RiskClassifier(risk_tiers)
        result = classifier.classify_content(
            "config/settings.py",
            "DEBUG = True\nLOG_LEVEL = 'info'",
        )
        assert result.tier == 1

    def test_keyword_matching_is_case_insensitive(self, risk_tiers):
        """Test that keyword matching ignores case."""
        classifier = RiskClassifier(risk_tiers)
        result = classifier.classify_content(
            "config/settings.py",
            'API_KEY = "something"',
        )
        assert result.tier == 3


class TestRiskClassifierCombined:
    """Tests for combined path + keyword classification."""

    def test_classify_changes_combines_path_and_content(self, risk_tiers):
        """Test full classification with both paths and content."""
        classifier = RiskClassifier(risk_tiers)
        changes = [
            {"path": "config/settings.py", "content": "DEBUG = True"},
            {"path": "api/endpoints.py", "content": "def get_users():"},
        ]
        result = classifier.classify_changes(changes)
        assert result.tier == 2
        assert result.action == "flag"
        assert len(result.matched_rules) >= 1

    def test_keyword_in_low_tier_path_escalates(self, risk_tiers):
        """Test that a keyword in a low-tier file escalates the result."""
        classifier = RiskClassifier(risk_tiers)
        changes = [
            {
                "path": "config/settings.py",
                "content": 'SECRET_KEY = "my-secret"',
            },
        ]
        result = classifier.classify_changes(changes)
        assert result.tier == 3
        assert result.action == "block"

    def test_result_includes_justification(self, risk_tiers):
        """Test that the classification result includes justification."""
        classifier = RiskClassifier(risk_tiers)
        changes = [
            {"path": "payments/processor.py", "content": "amount = 100"},
        ]
        result = classifier.classify_changes(changes)
        assert result.justification != ""
        assert "payments/" in result.justification

    def test_result_includes_matched_rules(self, risk_tiers):
        """Test that matched rules are recorded in the result."""
        classifier = RiskClassifier(risk_tiers)
        changes = [
            {
                "path": "auth/login.py",
                "content": 'password = input("Enter password")',
            },
        ]
        result = classifier.classify_changes(changes)
        assert len(result.matched_rules) >= 1
