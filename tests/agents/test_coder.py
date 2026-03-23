"""Tests for the coder agent."""

import json

import pytest

from swarmpr.agents.coder import CoderAgent
from swarmpr.config import PipelineConfig
from swarmpr.orchestrator.state import (
    FileChange,
    FileChangeType,
    PipelineStage,
    PipelineState,
    TaskPlan,
)
from swarmpr.providers.base import AgentProvider, Message


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
            "prompt_tokens": 200,
            "completion_tokens": 300,
            "total_tokens": 500,
        }
        return self.response


@pytest.fixture
def pipeline_config() -> PipelineConfig:
    """Return a standard pipeline configuration.

    Returns:
        A PipelineConfig with defaults.
    """
    return PipelineConfig(
        max_diff_lines=500,
        test_timeout_seconds=120,
        forbidden_paths=[".env", "secrets/", "credentials/"],
    )


@pytest.fixture
def sample_coder_response() -> str:
    """Return a sample LLM response for code generation.

    Returns:
        A JSON string representing file changes.
    """
    return json.dumps({
        "files": [
            {
                "path": "payments/validator.py",
                "action": "modify",
                "content": (
                    "def validate_amount(amount):\n"
                    "    if amount is None:\n"
                    '        raise ValueError("Amount cannot be None")\n'
                    "    if amount <= 0:\n"
                    '        raise ValueError("Amount must be positive")\n'
                    "    return True\n"
                ),
            },
        ],
    })


def _make_state_with_plan(
    plan_files: list[dict] | None = None,
    risk_tier: int = 2,
) -> PipelineState:
    """Create a pipeline state with a plan ready for the coder.

    Args:
        plan_files: File changes for the plan.
        risk_tier: The plan's risk tier.

    Returns:
        A PipelineState with plan populated.
    """
    if plan_files is None:
        plan_files = [
            {
                "path": "payments/validator.py",
                "change_type": "modify",
                "description": "Add null check for amount parameter",
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
        task_description="Fix null check in payment validation",
        repo_path="/tmp/repo",
        branch_name="swarmpr/fix-null-check",
        plan=TaskPlan(
            task_description="Fix null check in payment validation",
            files=files,
            risk_tier=risk_tier,
            risk_justification="Payment path",
            estimated_complexity="low",
        ),
    )


class TestCoderAgent:
    """Tests for the CoderAgent."""

    def test_coder_stage(self, pipeline_config):
        """Test that coder reports the correct pipeline stage."""
        provider = FakeProvider()
        agent = CoderAgent(
            provider=provider, pipeline_config=pipeline_config
        )
        assert agent.stage == PipelineStage.CODER

    @pytest.mark.asyncio
    async def test_coder_produces_diff(
        self, pipeline_config, sample_coder_response
    ):
        """Test that the coder produces a diff on the state."""
        provider = FakeProvider(response=sample_coder_response)
        agent = CoderAgent(
            provider=provider, pipeline_config=pipeline_config
        )

        state = _make_state_with_plan()
        result = await agent.execute(state)
        assert result.diff is not None
        assert len(result.diff) > 0

    @pytest.mark.asyncio
    async def test_coder_diff_contains_file_changes(
        self, pipeline_config, sample_coder_response
    ):
        """Test that the diff references the planned files."""
        provider = FakeProvider(response=sample_coder_response)
        agent = CoderAgent(
            provider=provider, pipeline_config=pipeline_config
        )

        state = _make_state_with_plan()
        result = await agent.execute(state)
        assert "payments/validator.py" in result.diff

    @pytest.mark.asyncio
    async def test_coder_sends_plan_in_prompt(
        self, pipeline_config, sample_coder_response
    ):
        """Test that the coder includes the plan in its prompt."""
        provider = FakeProvider(response=sample_coder_response)
        agent = CoderAgent(
            provider=provider, pipeline_config=pipeline_config
        )

        state = _make_state_with_plan()
        await agent.execute(state)
        user_msg = provider.calls[0][-1]
        assert "payments/validator.py" in user_msg.content
        assert "null check" in user_msg.content.lower()

    @pytest.mark.asyncio
    async def test_coder_rejects_forbidden_paths(self, pipeline_config):
        """Test that the coder rejects changes to forbidden paths."""
        response = json.dumps({
            "files": [
                {
                    "path": ".env",
                    "action": "modify",
                    "content": "SECRET=leaked",
                },
            ],
        })
        provider = FakeProvider(response=response)
        agent = CoderAgent(
            provider=provider, pipeline_config=pipeline_config
        )

        state = _make_state_with_plan(
            plan_files=[{
                "path": ".env",
                "change_type": "modify",
                "description": "Update secret",
            }]
        )

        with pytest.raises(ValueError, match="forbidden"):
            await agent.execute(state)

    @pytest.mark.asyncio
    async def test_coder_handles_malformed_response(self, pipeline_config):
        """Test that the coder raises on unparseable LLM response."""
        provider = FakeProvider(response="Not valid JSON at all")
        agent = CoderAgent(
            provider=provider, pipeline_config=pipeline_config
        )

        state = _make_state_with_plan()
        with pytest.raises(ValueError, match="parse"):
            await agent.execute(state)

    @pytest.mark.asyncio
    async def test_coder_stores_generated_files(
        self, pipeline_config, sample_coder_response
    ):
        """Test that generated file contents are stored on the state."""
        provider = FakeProvider(response=sample_coder_response)
        agent = CoderAgent(
            provider=provider, pipeline_config=pipeline_config
        )

        state = _make_state_with_plan()
        result = await agent.execute(state)
        assert result.generated_files is not None
        assert "payments/validator.py" in result.generated_files

    @pytest.mark.asyncio
    async def test_coder_multiple_files(self, pipeline_config):
        """Test that the coder handles multiple file changes."""
        response = json.dumps({
            "files": [
                {
                    "path": "payments/validator.py",
                    "action": "modify",
                    "content": "def validate(): pass\n",
                },
                {
                    "path": "tests/test_validator.py",
                    "action": "create",
                    "content": "def test_validate(): pass\n",
                },
            ],
        })
        provider = FakeProvider(response=response)
        agent = CoderAgent(
            provider=provider, pipeline_config=pipeline_config
        )

        state = _make_state_with_plan(
            plan_files=[
                {
                    "path": "payments/validator.py",
                    "change_type": "modify",
                    "description": "Fix validate",
                },
                {
                    "path": "tests/test_validator.py",
                    "change_type": "create",
                    "description": "Add test",
                },
            ]
        )

        result = await agent.execute(state)
        assert len(result.generated_files) == 2
