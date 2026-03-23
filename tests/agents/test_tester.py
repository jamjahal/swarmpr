"""Tests for the tester agent."""

from unittest.mock import AsyncMock, patch

import pytest

from swarmpr.agents.tester import TesterAgent
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
    """A fake provider for testing."""

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
            "prompt_tokens": 50,
            "completion_tokens": 30,
            "total_tokens": 80,
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
        test_timeout_seconds=5,
        forbidden_paths=[".env"],
    )


def _make_state_with_diff() -> PipelineState:
    """Create a pipeline state with diff ready for testing.

    Returns:
        A PipelineState with plan and diff populated.
    """
    return PipelineState(
        task_description="Fix null check",
        repo_path="/tmp/repo",
        branch_name="swarmpr/fix-null-check",
        diff="--- a/payments/validator.py\n+++ b/payments/validator.py",
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
    )


class TestTesterAgent:
    """Tests for the TesterAgent."""

    def test_tester_stage(self, pipeline_config):
        """Test that tester reports the correct pipeline stage."""
        provider = FakeProvider()
        agent = TesterAgent(
            provider=provider, pipeline_config=pipeline_config
        )
        assert agent.stage == PipelineStage.TESTER

    @pytest.mark.asyncio
    async def test_tester_runs_tests_passing(self, pipeline_config):
        """Test tester with passing subprocess result."""
        provider = FakeProvider()
        agent = TesterAgent(
            provider=provider, pipeline_config=pipeline_config
        )
        state = _make_state_with_diff()

        mock_result = AsyncMock()
        mock_result.returncode = 0
        mock_result.stdout = "3 passed in 0.5s"
        mock_result.stderr = ""

        with patch(
            "swarmpr.agents.tester.asyncio.create_subprocess_exec",
            return_value=mock_result,
        ):
            mock_result.communicate = AsyncMock(
                return_value=(b"3 passed in 0.5s", b"")
            )
            result = await agent.execute(state)

        assert result.tests_passed is True
        assert "3 passed" in result.test_output

    @pytest.mark.asyncio
    async def test_tester_runs_tests_failing(self, pipeline_config):
        """Test tester with failing subprocess result."""
        provider = FakeProvider()
        agent = TesterAgent(
            provider=provider, pipeline_config=pipeline_config
        )
        state = _make_state_with_diff()

        mock_result = AsyncMock()
        mock_result.returncode = 1
        mock_result.communicate = AsyncMock(
            return_value=(b"1 failed, 2 passed", b"FAILED test_x")
        )

        with patch(
            "swarmpr.agents.tester.asyncio.create_subprocess_exec",
            return_value=mock_result,
        ):
            result = await agent.execute(state)

        assert result.tests_passed is False
        assert "failed" in result.test_output.lower()

    @pytest.mark.asyncio
    async def test_tester_handles_no_test_suite(self, pipeline_config):
        """Test tester when no test runner is found."""
        provider = FakeProvider()
        agent = TesterAgent(
            provider=provider, pipeline_config=pipeline_config
        )
        state = _make_state_with_diff()

        mock_result = AsyncMock()
        mock_result.returncode = 5  # pytest exit code for no tests collected
        mock_result.communicate = AsyncMock(
            return_value=(b"no tests ran", b"")
        )

        with patch(
            "swarmpr.agents.tester.asyncio.create_subprocess_exec",
            return_value=mock_result,
        ):
            result = await agent.execute(state)

        # No tests is not a failure — it's a skip
        assert result.tests_passed is None or result.tests_passed is True

    @pytest.mark.asyncio
    async def test_tester_handles_timeout(self, pipeline_config):
        """Test tester handles subprocess timeout gracefully."""
        import asyncio

        provider = FakeProvider()
        agent = TesterAgent(
            provider=provider, pipeline_config=pipeline_config
        )
        state = _make_state_with_diff()

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )
        mock_proc.kill = AsyncMock()
        mock_proc.wait = AsyncMock()

        with patch(
            "swarmpr.agents.tester.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            result = await agent.execute(state)

        assert result.tests_passed is False
        assert "timed out" in result.test_output.lower()

    @pytest.mark.asyncio
    async def test_tester_uses_configured_timeout(self, pipeline_config):
        """Test that the tester uses the configured timeout value."""
        provider = FakeProvider()
        agent = TesterAgent(
            provider=provider, pipeline_config=pipeline_config
        )
        assert agent.timeout_seconds == pipeline_config.test_timeout_seconds

    @pytest.mark.asyncio
    async def test_tester_records_raw_output(self, pipeline_config):
        """Test that raw test output is stored on the state."""
        provider = FakeProvider()
        agent = TesterAgent(
            provider=provider, pipeline_config=pipeline_config
        )
        state = _make_state_with_diff()

        output_text = "===== 5 passed, 1 warning in 1.2s ====="
        mock_result = AsyncMock()
        mock_result.returncode = 0
        mock_result.communicate = AsyncMock(
            return_value=(output_text.encode(), b"")
        )

        with patch(
            "swarmpr.agents.tester.asyncio.create_subprocess_exec",
            return_value=mock_result,
        ):
            result = await agent.execute(state)

        assert output_text in result.test_output
