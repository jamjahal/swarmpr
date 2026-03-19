"""Base agent class for SwarmPR pipeline agents.

Defines the abstract interface that all agents (planner, coder, tester,
reviewer) must implement, plus shared utilities for LLM interaction.
"""

from abc import ABC, abstractmethod

from swarmpr.orchestrator.state import PipelineStage, PipelineState
from swarmpr.providers.base import AgentProvider, Message, Role


class BaseAgent(ABC):
    """Abstract base class for all SwarmPR pipeline agents.

    Provides common LLM interaction patterns and defines the interface
    that the orchestration engine uses to run each agent.

    Attributes:
        provider: The LLM provider this agent uses for completions.
    """

    def __init__(self, provider: AgentProvider) -> None:
        """Initialize the agent with an LLM provider.

        Args:
            provider: The LLM provider to use for completions.
        """
        self.provider = provider

    @property
    @abstractmethod
    def stage(self) -> PipelineStage:
        """Return the pipeline stage this agent occupies.

        Returns:
            The PipelineStage enum value for this agent.
        """
        ...

    @abstractmethod
    async def execute(self, state: PipelineState) -> PipelineState:
        """Execute the agent's work and return updated pipeline state.

        This is the main entry point called by the orchestration engine.
        Implementations should read from the state, perform their work
        (typically involving LLM calls), and return the modified state.

        Args:
            state: The current pipeline state.

        Returns:
            The updated pipeline state after this agent's work.
        """
        ...

    async def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        context_messages: list[Message] | None = None,
        **kwargs: object,
    ) -> str:
        """Send a structured prompt to the LLM provider.

        Builds a message list from the system prompt, optional context,
        and user prompt, then sends it to the provider.

        Args:
            system_prompt: The system-level instructions.
            user_prompt: The user-level prompt or question.
            context_messages: Optional additional messages inserted between
                the system prompt and user prompt.
            **kwargs: Additional parameters forwarded to the provider.

        Returns:
            The LLM's response as a string.
        """
        messages: list[Message] = [
            Message(role=Role.SYSTEM, content=system_prompt),
        ]

        if context_messages:
            messages.extend(context_messages)

        messages.append(Message(role=Role.USER, content=user_prompt))

        return await self.provider.complete(messages, **kwargs)

    def get_last_usage(self) -> dict | None:
        """Return token usage stats from the most recent LLM call.

        Returns:
            A dictionary with prompt_tokens, completion_tokens, and
            total_tokens, or None if no call has been made yet.
        """
        return self.provider.last_usage
