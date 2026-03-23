"""Tests for the reviewer agent."""

import json

import pytest

from swarmpr.agents.reviewer import ReviewerAgent
from swarmpr.config import PipelineConfig, RiskTierConfig
from swarmpr.orchestrator.state import (
    FileChange,
    FileChangeType,
    PipelineStage,
    PipelineState,
    ReviewVerdict,
    TaskPlan,
)
from swarmpr.providers.base import AgentProvider, Message
from swarmpr.risk.classifier import RiskClassifier
from swarmpr.risk.policies import EscalationPolicy


class FakeProvider(AgentProvider):
    """A fake provider that returns configurable JSON responses."""

    def __init__(self, response: str = "{}") -> None:
        """Initialize with a canned response.

        Args:
            response: JSON string to return from complete().
        """
        self.response = response
        self.last_usage: dict | None = None
        self.calls: list[list[Message]] = []

    async def complete(self, messages: list[Message], **kwargs: object) -> str:
        """Record the call and return the canned response.

        Args:
            messages: The messages sent to the provider.
            **kwargs: Additional parameters.

        Returns:
            The canned response string.
        """
        self.calls.append(messages)
        self.last_usage = {
            "prompt_tokens": 150,
            "completion_tokens": 100,
            "total_tokens": 250,
        }
        return self.response


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
            keywords=["api_key", "secret"],
            action="block",
        ),
        "tier_2": RiskTierConfig(
            description="Business logic",
            paths=["api/", "services/"],
            keywords=[],
            action="flag",
        ),
        "tier_1": RiskTierConfig(
            description="Low risk",
            paths=["config/", "docs/", "tests/"],
            keywords=[],
            action="approve",
        ),
    }


@pytest.fixture
def pipeline_config() -> PipelineConfig:
    """Return a standard pipeline configuration.

    Returns:
        A PipelineConfig with defaults.
    """
    return PipelineConfig(
        max_diff_lines=500,
        test_timeout_seconds=120,
        forbidden_paths=[".env", "secrets/"],
    )


@pytest.fixture
def sample_review_response() -> str:
    """Return a sample LLM review response.

    Returns:
        A JSON string representing a review result.
    """
    return json.dumps({
        "risk_score": 7.5,
        "summary": "Changes touch payment validation logic. "
        "Null check added correctly but needs test coverage.",
        "findings": [
            "Payment validation logic modified",
            "Test added for null case",
            "No changes to error handling paths",
        ],
        "code_quality": "good",
    })


def _make_state_with_plan(
    task: str = "Fix null check",
    diff: str = "--- a/payments/validator.py\n+++ b/payments/validator.py",
    plan_files: list[dict] | None = None,
    risk_tier: int = 3,
    tests_passed: bool = True,
) -> PipelineState:
    """Create a pipeline state with a plan and diff for review.

    Args:
        task: The task description.
        diff: The code diff.
        plan_files: File changes for the plan.
        risk_tier: The plan's risk tier.
        tests_passed: Whether tests passed.

    Returns:
        A PipelineState ready for the reviewer agent.
    """
    if plan_files is None:
        plan_files = [
            {
                "path": "payments/validator.py",
                "change_type": "modify",
                "description": "Fix null check",
            },
        ]

    files = [
        FileChange(
            path=f["path"],
            change_type=FileChangeType(f["change_type"]),
            description=f["description"],
        )
        for f in plan_files
    ]

    return PipelineState(
        task_description=task,
        repo_path="/tmp/repo",
        diff=diff,
        tests_passed=tests_passed,
        test_output="3 passed, 0 failed",
        plan=TaskPlan(
            task_description=task,
            files=files,
            risk_tier=risk_tier,
            risk_justification="Payment path",
            estimated_complexity="low",
        ),
    )


class TestReviewerAgent:
    """Tests for the ReviewerAgent."""

    def test_reviewer_stage(self, risk_tiers, pipeline_config):
        """Test that reviewer reports the correct pipeline stage."""
        provider = FakeProvider()
        classifier = RiskClassifier(risk_tiers)
        policy = EscalationPolicy(risk_tiers, pipeline_config)
        agent = ReviewerAgent(
            provider=provider,
            classifier=classifier,
            policy=policy,
        )
        assert agent.stage == PipelineStage.REVIEWER

    @pytest.mark.asyncio
    async def test_reviewer_produces_verdict(
        self, risk_tiers, pipeline_config, sample_review_response
    ):
        """Test that the reviewer produces a ReviewVerdict on the state."""
        provider = FakeProvider(response=sample_review_response)
        classifier = RiskClassifier(risk_tiers)
        policy = EscalationPolicy(risk_tiers, pipeline_config)
        agent = ReviewerAgent(
            provider=provider,
            classifier=classifier,
            policy=policy,
        )

        state = _make_state_with_plan()
        result = await agent.execute(state)
        assert result.review is not None
        assert isinstance(result.review, ReviewVerdict)

    @pytest.mark.asyncio
    async def test_reviewer_verdict_includes_risk_score(
        self, risk_tiers, pipeline_config, sample_review_response
    ):
        """Test that the verdict includes the LLM's risk score."""
        provider = FakeProvider(response=sample_review_response)
        classifier = RiskClassifier(risk_tiers)
        policy = EscalationPolicy(risk_tiers, pipeline_config)
        agent = ReviewerAgent(
            provider=provider,
            classifier=classifier,
            policy=policy,
        )

        state = _make_state_with_plan()
        result = await agent.execute(state)
        assert result.review.risk_score == 7.5

    @pytest.mark.asyncio
    async def test_reviewer_blocks_tier_3(
        self, risk_tiers, pipeline_config, sample_review_response
    ):
        """Test that tier 3 changes result in a block decision."""
        provider = FakeProvider(response=sample_review_response)
        classifier = RiskClassifier(risk_tiers)
        policy = EscalationPolicy(risk_tiers, pipeline_config)
        agent = ReviewerAgent(
            provider=provider,
            classifier=classifier,
            policy=policy,
        )

        state = _make_state_with_plan(risk_tier=3)
        result = await agent.execute(state)
        assert result.review.escalation_action == "block"
        assert result.review.approved is False

    @pytest.mark.asyncio
    async def test_reviewer_approves_tier_1(
        self, risk_tiers, pipeline_config
    ):
        """Test that tier 1 changes can be auto-approved."""
        review_response = json.dumps({
            "risk_score": 1.0,
            "summary": "Config change only, low risk.",
            "findings": [],
            "code_quality": "good",
        })
        provider = FakeProvider(response=review_response)
        classifier = RiskClassifier(risk_tiers)
        policy = EscalationPolicy(risk_tiers, pipeline_config)
        agent = ReviewerAgent(
            provider=provider,
            classifier=classifier,
            policy=policy,
        )

        state = _make_state_with_plan(
            task="Update config",
            diff="--- a/config/settings.py\n+++ b/config/settings.py",
            plan_files=[{
                "path": "config/settings.py",
                "change_type": "modify",
                "description": "Update log level",
            }],
            risk_tier=1,
        )
        result = await agent.execute(state)
        assert result.review.escalation_action == "approve"
        assert result.review.approved is True

    @pytest.mark.asyncio
    async def test_reviewer_includes_findings(
        self, risk_tiers, pipeline_config, sample_review_response
    ):
        """Test that the verdict includes specific findings."""
        provider = FakeProvider(response=sample_review_response)
        classifier = RiskClassifier(risk_tiers)
        policy = EscalationPolicy(risk_tiers, pipeline_config)
        agent = ReviewerAgent(
            provider=provider,
            classifier=classifier,
            policy=policy,
        )

        state = _make_state_with_plan()
        result = await agent.execute(state)
        assert len(result.review.findings) > 0

    @pytest.mark.asyncio
    async def test_reviewer_sends_diff_in_prompt(
        self, risk_tiers, pipeline_config, sample_review_response
    ):
        """Test that the reviewer includes the diff in its prompt."""
        provider = FakeProvider(response=sample_review_response)
        classifier = RiskClassifier(risk_tiers)
        policy = EscalationPolicy(risk_tiers, pipeline_config)
        agent = ReviewerAgent(
            provider=provider,
            classifier=classifier,
            policy=policy,
        )

        diff_text = (
            "--- a/payments/validator.py\n"
            "+++ b/payments/validator.py\n"
            "+if amount is None:"
        )
        state = _make_state_with_plan(diff=diff_text)
        await agent.execute(state)
        user_msg = provider.calls[0][-1]
        assert "payments/validator.py" in user_msg.content

    @pytest.mark.asyncio
    async def test_reviewer_handles_malformed_response(
        self, risk_tiers, pipeline_config
    ):
        """Test that the reviewer raises on unparseable LLM response."""
        provider = FakeProvider(response="Not valid JSON")
        classifier = RiskClassifier(risk_tiers)
        policy = EscalationPolicy(risk_tiers, pipeline_config)
        agent = ReviewerAgent(
            provider=provider,
            classifier=classifier,
            policy=policy,
        )

        state = _make_state_with_plan()
        with pytest.raises(ValueError, match="parse"):
            await agent.execute(state)

    @pytest.mark.asyncio
    async def test_reviewer_fails_tests_affects_approval(
        self, risk_tiers, pipeline_config
    ):
        """Test that failing tests prevent auto-approval."""
        review_response = json.dumps({
            "risk_score": 2.0,
            "summary": "Simple config change.",
            "findings": [],
            "code_quality": "good",
        })
        provider = FakeProvider(response=review_response)
        classifier = RiskClassifier(risk_tiers)
        policy = EscalationPolicy(risk_tiers, pipeline_config)
        agent = ReviewerAgent(
            provider=provider,
            classifier=classifier,
            policy=policy,
        )

        state = _make_state_with_plan(
            risk_tier=1,
            tests_passed=False,
            plan_files=[{
                "path": "config/settings.py",
                "change_type": "modify",
                "description": "Change setting",
            }],
        )
        result = await agent.execute(state)
        assert result.review.approved is False
