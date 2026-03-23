"""Tests for the planner agent."""

import json

import pytest

from swarmpr.agents.planner import PlannerAgent
from swarmpr.config import RiskTierConfig
from swarmpr.orchestrator.state import (
    PipelineStage,
    PipelineState,
    TaskPlan,
)
from swarmpr.providers.base import AgentProvider, Message
from swarmpr.risk.classifier import RiskClassifier


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
            "prompt_tokens": 100,
            "completion_tokens": 200,
            "total_tokens": 300,
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
def sample_plan_response() -> str:
    """Return a sample LLM response for plan generation.

    Returns:
        A JSON string representing a valid plan.
    """
    return json.dumps({
        "task_description": "Fix null check in payment validation",
        "files": [
            {
                "path": "payments/validator.py",
                "change_type": "modify",
                "description": "Add null check for amount parameter",
            },
            {
                "path": "tests/test_validator.py",
                "change_type": "modify",
                "description": "Add test for null amount",
            },
        ],
        "estimated_complexity": "low",
    })


class TestPlannerAgent:
    """Tests for the PlannerAgent."""

    def test_planner_stage(self, risk_tiers):
        """Test that planner reports the correct pipeline stage."""
        provider = FakeProvider()
        classifier = RiskClassifier(risk_tiers)
        agent = PlannerAgent(provider=provider, classifier=classifier)
        assert agent.stage == PipelineStage.PLANNER

    @pytest.mark.asyncio
    async def test_planner_produces_plan(
        self, risk_tiers, sample_plan_response
    ):
        """Test that the planner produces a TaskPlan on the state."""
        provider = FakeProvider(response=sample_plan_response)
        classifier = RiskClassifier(risk_tiers)
        agent = PlannerAgent(provider=provider, classifier=classifier)

        state = PipelineState(
            task_description="Fix null check in payment validation",
            repo_path="/tmp/repo",
        )

        result = await agent.execute(state)
        assert result.plan is not None
        assert isinstance(result.plan, TaskPlan)
        assert len(result.plan.files) == 2

    @pytest.mark.asyncio
    async def test_planner_assigns_risk_tier(
        self, risk_tiers, sample_plan_response
    ):
        """Test that the planner assigns a risk tier via the classifier."""
        provider = FakeProvider(response=sample_plan_response)
        classifier = RiskClassifier(risk_tiers)
        agent = PlannerAgent(provider=provider, classifier=classifier)

        state = PipelineState(
            task_description="Fix null check in payment validation",
            repo_path="/tmp/repo",
        )

        result = await agent.execute(state)
        # payments/validator.py should trigger tier 3
        assert result.plan.risk_tier == 3

    @pytest.mark.asyncio
    async def test_planner_assigns_tier_1_for_config(self, risk_tiers):
        """Test that config-only changes get tier 1."""
        response = json.dumps({
            "task_description": "Update log level",
            "files": [
                {
                    "path": "config/settings.py",
                    "change_type": "modify",
                    "description": "Change log level to DEBUG",
                },
            ],
            "estimated_complexity": "low",
        })
        provider = FakeProvider(response=response)
        classifier = RiskClassifier(risk_tiers)
        agent = PlannerAgent(provider=provider, classifier=classifier)

        state = PipelineState(
            task_description="Update log level",
            repo_path="/tmp/repo",
        )

        result = await agent.execute(state)
        assert result.plan.risk_tier == 1

    @pytest.mark.asyncio
    async def test_planner_sends_task_in_prompt(
        self, risk_tiers, sample_plan_response
    ):
        """Test that the planner includes the task description in its prompt."""
        provider = FakeProvider(response=sample_plan_response)
        classifier = RiskClassifier(risk_tiers)
        agent = PlannerAgent(provider=provider, classifier=classifier)

        state = PipelineState(
            task_description="Fix null check in payment validation",
            repo_path="/tmp/repo",
        )

        await agent.execute(state)
        assert len(provider.calls) == 1
        user_msg = provider.calls[0][-1]
        assert "Fix null check" in user_msg.content

    @pytest.mark.asyncio
    async def test_planner_generates_branch_name(
        self, risk_tiers, sample_plan_response
    ):
        """Test that the planner generates a branch name on the state."""
        provider = FakeProvider(response=sample_plan_response)
        classifier = RiskClassifier(risk_tiers)
        agent = PlannerAgent(provider=provider, classifier=classifier)

        state = PipelineState(
            task_description="Fix null check in payment validation",
            repo_path="/tmp/repo",
        )

        result = await agent.execute(state)
        assert result.branch_name is not None
        assert result.branch_name.startswith("swarmpr/")

    @pytest.mark.asyncio
    async def test_planner_handles_malformed_response(self, risk_tiers):
        """Test that the planner raises on unparseable LLM response."""
        provider = FakeProvider(response="This is not JSON at all")
        classifier = RiskClassifier(risk_tiers)
        agent = PlannerAgent(provider=provider, classifier=classifier)

        state = PipelineState(
            task_description="Some task",
            repo_path="/tmp/repo",
        )

        with pytest.raises(ValueError, match="parse"):
            await agent.execute(state)
