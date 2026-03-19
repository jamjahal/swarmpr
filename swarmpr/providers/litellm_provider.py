"""LiteLLM provider implementation for SwarmPR.

Wraps LiteLLM's async completion API to satisfy the AgentProvider
protocol. Supports any model provider that LiteLLM supports (100+),
including local models via Ollama.
"""


from litellm import acompletion

from swarmpr.config import ProviderConfig
from swarmpr.providers.base import AgentProvider, Message


class LiteLLMProvider(AgentProvider):
    """LLM provider backed by LiteLLM for multi-provider support.

    Translates SwarmPR's provider config into LiteLLM API calls,
    supporting cloud APIs (Anthropic, OpenAI, Google) and local
    models (Ollama, vLLM) through a single interface.

    Attributes:
        model: LiteLLM model identifier.
        api_base: Optional custom API endpoint.
        api_key: Optional API key override.
        temperature: Sampling temperature.
        max_tokens: Maximum response tokens.
        last_usage: Token usage from the most recent call.
    """

    def __init__(self, config: ProviderConfig) -> None:
        """Initialize the LiteLLM provider from a ProviderConfig.

        Args:
            config: Provider configuration with model, API base, and params.
        """
        self.model = config.model
        self.api_base = config.api_base
        self.api_key = config.api_key
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.last_usage: dict | None = None

    async def complete(self, messages: list[Message], **kwargs: object) -> str:
        """Send messages to the configured LLM and return the response.

        Args:
            messages: List of conversation messages.
            **kwargs: Additional parameters forwarded to LiteLLM.

        Returns:
            The model's response text.

        Raises:
            litellm.exceptions.APIError: If the API call fails.
        """
        call_kwargs: dict = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if self.api_base:
            call_kwargs["api_base"] = self.api_base

        if self.api_key:
            call_kwargs["api_key"] = self.api_key

        call_kwargs.update(kwargs)

        response = await acompletion(**call_kwargs)

        if response.usage:
            self.last_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return response.choices[0].message.content
