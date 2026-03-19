"""Tests for the BaseAgent abstract class."""

import pytest

from swarmpr.agents.base import BaseAgent
from swarmpr.orchestrator.state import PipelineStage, PipelineState
from swarmpr.providers.base import AgentProvider, Message


class FakeProvider(AgentProvider):
    """A fake provider for testing."""

    def __init__(self, response: str = "fake") -> None:
        """Initialize with a canned response.

        Args:
            response: The string to return from complete().
        """
        self.response = response
        self.last_usage = None
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
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        return self.response


class ConcreteAgent(BaseAgent):
    """A minimal concrete agent for testing the base class."""

    @property
    def stage(self) -> PipelineStage:
        """Return the planner stage.

        Returns:
            PipelineStage.PLANNER.
        """
        return PipelineStage.PLANNER

    async def execute(self, state: PipelineState) -> PipelineState:
        """Call the provider and store the response on the state branch name.

        Args:
            state: The current pipeline state.

        Returns:
            The modified pipeline state.
        """
        response = await self.call_llm(
            system_prompt="You are a test agent.",
            user_prompt=state.task_description,
        )
        state.branch_name = response
        return state


class TestBaseAgent:
    """Tests for the BaseAgent abstract class."""

    def test_base_agent_requires_provider(self):
        """Test that the base agent stores a provider reference."""
        provider = FakeProvider()
        agent = ConcreteAgent(provider=provider)
        assert agent.provider is provider

    def test_base_agent_has_stage_property(self):
        """Test that concrete agents expose their pipeline stage."""
        provider = FakeProvider()
        agent = ConcreteAgent(provider=provider)
        assert agent.stage == PipelineStage.PLANNER

    @pytest.mark.asyncio
    async def test_call_llm_sends_messages_to_provider(self):
        """Test that call_llm constructs and sends messages correctly."""
        provider = FakeProvider(response="test-branch-name")
        agent = ConcreteAgent(provider=provider)

        result = await agent.call_llm(
            system_prompt="You are a planner.",
            user_prompt="Create a branch for fixing payments.",
        )

        assert result == "test-branch-name"
        assert len(provider.calls) == 1
        messages = provider.calls[0]
        assert len(messages) == 2
        assert messages[0].role.value == "system"
        assert messages[0].content == "You are a planner."
        assert messages[1].role.value == "user"

    @pytest.mark.asyncio
    async def test_call_llm_with_context_messages(self):
        """Test that call_llm can include additional context messages."""
        provider = FakeProvider(response="done")
        agent = ConcreteAgent(provider=provider)

        context = [
            Message(role="user", content="Previous question"),
            Message(role="assistant", content="Previous answer"),
        ]

        await agent.call_llm(
            system_prompt="System",
            user_prompt="New question",
            context_messages=context,
        )

        messages = provider.calls[0]
        # system + 2 context + user = 4 messages
        assert len(messages) == 4
        assert messages[1].content == "Previous question"
        assert messages[2].content == "Previous answer"

    @pytest.mark.asyncio
    async def test_execute_modifies_state(self):
        """Test that a concrete agent can modify pipeline state."""
        provider = FakeProvider(response="swarmpr/fix-payment-null")
        agent = ConcreteAgent(provider=provider)

        state = PipelineState(
            task_description="Fix null check",
            repo_path="/tmp/repo",
        )

        result = await agent.execute(state)
        assert result.branch_name == "swarmpr/fix-payment-null"

    @pytest.mark.asyncio
    async def test_get_last_usage_after_call(self):
        """Test that token usage is accessible after an LLM call."""
        provider = FakeProvider(response="test")
        agent = ConcreteAgent(provider=provider)

        await agent.call_llm(system_prompt="Sys", user_prompt="User")

        usage = agent.get_last_usage()
        assert usage is not None
        assert usage["total_tokens"] == 15
