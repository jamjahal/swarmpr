"""Tests for the AgentProvider protocol and LiteLLM implementation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from swarmpr.config import ProviderConfig
from swarmpr.providers.base import AgentProvider, Message, Role
from swarmpr.providers.litellm_provider import LiteLLMProvider


class TestMessage:
    """Tests for the Message data class."""

    def test_create_user_message(self):
        """Test creating a user message."""
        msg = Message(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"

    def test_create_system_message(self):
        """Test creating a system message."""
        msg = Message(role=Role.SYSTEM, content="You are a planner agent.")
        assert msg.role == Role.SYSTEM

    def test_create_assistant_message(self):
        """Test creating an assistant message."""
        msg = Message(role=Role.ASSISTANT, content="I will analyze the repo.")
        assert msg.role == Role.ASSISTANT

    def test_message_to_dict(self):
        """Test converting a message to an API-compatible dictionary."""
        msg = Message(role=Role.USER, content="Hello")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "Hello"}


class TestAgentProviderProtocol:
    """Tests for the AgentProvider protocol definition."""

    def test_litellm_provider_satisfies_protocol(self):
        """Test that LiteLLMProvider implements the AgentProvider protocol."""
        config = ProviderConfig(model="anthropic/claude-sonnet-4-20250514")
        provider = LiteLLMProvider(config=config)
        assert isinstance(provider, AgentProvider)


class TestLiteLLMProvider:
    """Tests for the LiteLLM provider implementation."""

    def test_provider_initialization(self):
        """Test that the provider initializes with a config."""
        config = ProviderConfig(
            model="anthropic/claude-sonnet-4-20250514",
            api_base="http://localhost:11434",
            temperature=0.5,
            max_tokens=2048,
        )
        provider = LiteLLMProvider(config=config)
        assert provider.model == "anthropic/claude-sonnet-4-20250514"
        assert provider.api_base == "http://localhost:11434"
        assert provider.temperature == 0.5
        assert provider.max_tokens == 2048

    def test_provider_default_temperature(self):
        """Test that provider defaults to temperature 0 (deterministic)."""
        config = ProviderConfig(model="openai/gpt-4o")
        provider = LiteLLMProvider(config=config)
        assert provider.temperature == 0.0

    @pytest.mark.asyncio
    async def test_complete_returns_string(self):
        """Test that complete() returns a string response."""
        config = ProviderConfig(model="anthropic/claude-sonnet-4-20250514")
        provider = LiteLLMProvider(config=config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )

        with patch(
            "swarmpr.providers.litellm_provider.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            messages = [Message(role=Role.USER, content="Hello")]
            result = await provider.complete(messages)
            assert result == "Test response"
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_complete_with_usage_tracking(self):
        """Test that complete() tracks token usage when available."""
        config = ProviderConfig(model="anthropic/claude-sonnet-4-20250514")
        provider = LiteLLMProvider(config=config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.usage = MagicMock(
            prompt_tokens=100, completion_tokens=50, total_tokens=150
        )

        with patch(
            "swarmpr.providers.litellm_provider.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            messages = [Message(role=Role.USER, content="Hello")]
            await provider.complete(messages)
            assert provider.last_usage is not None
            assert provider.last_usage["prompt_tokens"] == 100
            assert provider.last_usage["completion_tokens"] == 50
            assert provider.last_usage["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_complete_passes_correct_params_to_litellm(self):
        """Test that complete() forwards config params to litellm."""
        config = ProviderConfig(
            model="ollama/llama3",
            api_base="http://192.168.1.100:11434",
            temperature=0.7,
            max_tokens=1024,
        )
        provider = LiteLLMProvider(config=config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Local response"
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )

        with patch(
            "swarmpr.providers.litellm_provider.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_acompletion:
            messages = [
                Message(role=Role.SYSTEM, content="You are helpful."),
                Message(role=Role.USER, content="Hello"),
            ]
            await provider.complete(messages)

            mock_acompletion.assert_called_once_with(
                model="ollama/llama3",
                messages=[
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"},
                ],
                api_base="http://192.168.1.100:11434",
                temperature=0.7,
                max_tokens=1024,
            )
