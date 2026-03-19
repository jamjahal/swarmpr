"""Event bus for SwarmPR pipeline observability.

Provides a publish-subscribe event system that agents and the orchestrator
use to report activity. Subscribers can be terminal loggers (now),
WebSocket streamers (future React UI), or metrics collectors.
"""

from collections.abc import Awaitable, Callable
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from swarmpr.orchestrator.state import PipelineStage


class EventType(str, Enum):
    """Types of events emitted during pipeline execution.

    Attributes:
        PIPELINE_STARTED: Pipeline run has begun.
        PIPELINE_COMPLETED: Pipeline run finished successfully.
        PIPELINE_FAILED: Pipeline run encountered a fatal error.
        AGENT_STARTED: An agent has started its work.
        AGENT_COMPLETED: An agent has finished successfully.
        AGENT_FAILED: An agent has encountered an error.
        AGENT_MESSAGE: An agent is reporting intermediate progress.
    """

    PIPELINE_STARTED = "pipeline_started"
    PIPELINE_COMPLETED = "pipeline_completed"
    PIPELINE_FAILED = "pipeline_failed"
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    AGENT_MESSAGE = "agent_message"


class Event(BaseModel):
    """A single event emitted by the pipeline or an agent.

    Attributes:
        event_type: The category of this event.
        message: Human-readable event description.
        stage: The pipeline stage that emitted the event, if applicable.
        data: Additional structured data associated with the event.
        timestamp: When the event was created.
    """

    event_type: EventType
    message: str = ""
    stage: PipelineStage | None = None
    data: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Type alias for event handler callbacks.
EventHandler = Callable[[Event], Awaitable[None]]


class _Subscription:
    """Internal subscription record pairing a handler with optional filters.

    Attributes:
        handler: The async callback function.
        event_types: If set, only events matching these types are delivered.
    """

    def __init__(
        self,
        handler: EventHandler,
        event_types: list[EventType] | None = None,
    ) -> None:
        """Initialize a subscription.

        Args:
            handler: The async callback to invoke on matching events.
            event_types: Optional filter; if None, all events are delivered.
        """
        self.handler = handler
        self.event_types = event_types


class EventBus:
    """Publish-subscribe event bus for pipeline observability.

    Agents and the orchestrator emit events through the bus. Subscribers
    (terminal logger, WebSocket streamer, metrics collector) receive
    events asynchronously. Supports filtered subscriptions by event type.

    Attributes:
        history: Chronological list of all emitted events.
    """

    def __init__(self) -> None:
        """Initialize an empty event bus."""
        self._subscriptions: list[_Subscription] = []
        self.history: list[Event] = []

    def subscribe(
        self,
        handler: EventHandler,
        event_types: list[EventType] | None = None,
    ) -> None:
        """Register a handler to receive events.

        Args:
            handler: Async function called with each matching event.
            event_types: If provided, only deliver events of these types.
                If None, the handler receives all events.
        """
        self._subscriptions.append(
            _Subscription(handler=handler, event_types=event_types)
        )

    def unsubscribe(self, handler: EventHandler) -> None:
        """Remove a previously registered handler.

        Args:
            handler: The handler function to remove. If the handler was
                registered multiple times, all registrations are removed.
        """
        self._subscriptions = [
            sub for sub in self._subscriptions if sub.handler != handler
        ]

    async def emit(self, event: Event) -> None:
        """Emit an event to all matching subscribers.

        The event is added to the history and delivered to every subscriber
        whose filter (if any) matches the event type.

        Args:
            event: The event to broadcast.
        """
        self.history.append(event)
        for sub in self._subscriptions:
            if sub.event_types is None or event.event_type in sub.event_types:
                await sub.handler(event)

    def clear_history(self) -> None:
        """Clear the event history."""
        self.history.clear()
