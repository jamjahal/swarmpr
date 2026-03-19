"""Base provider protocol and shared types for LLM agent providers.

Defines the AgentProvider abstract base class that all provider
implementations must satisfy, plus shared data types for messages.
"""

from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel


class Role(str, Enum):
    """Message roles for LLM conversations.

    Attributes:
        SYSTEM: System-level instructions for the model.
        USER: User/human messages.
        ASSISTANT: Model/assistant responses.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """A single message in an LLM conversation.

    Attributes:
        role: The role of the message sender.
        content: The text content of the message.
    """

    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert the message to an API-compatible dictionary.

        Returns:
            A dictionary with 'role' and 'content' keys.
        """
        return {"role": self.role.value, "content": self.content}


class AgentProvider(ABC):
    """Abstract base class for LLM provider implementations.

    All provider implementations (LiteLLM, direct API clients, local models)
    must implement this interface. This keeps the orchestration layer
    decoupled from any specific LLM provider.

    Attributes:
        last_usage: Token usage stats from the most recent completion call.
    """

    last_usage: dict | None = None

    @abstractmethod
    async def complete(self, messages: list[Message], **kwargs: object) -> str:
        """Send messages to the LLM and return the response text.

        Args:
            messages: List of conversation messages.
            **kwargs: Additional provider-specific parameters.

        Returns:
            The model's response as a string.
        """
        ...
