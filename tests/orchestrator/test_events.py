"""Tests for the SwarmPR event bus system."""

from datetime import datetime

import pytest

from swarmpr.orchestrator.events import (
    Event,
    EventBus,
    EventType,
)
from swarmpr.orchestrator.state import PipelineStage


class TestEventType:
    """Tests for the EventType enum."""

    def test_event_types_exist(self):
        """Test that all expected event types are defined."""
        assert EventType.PIPELINE_STARTED.value == "pipeline_started"
        assert EventType.PIPELINE_COMPLETED.value == "pipeline_completed"
        assert EventType.PIPELINE_FAILED.value == "pipeline_failed"
        assert EventType.AGENT_STARTED.value == "agent_started"
        assert EventType.AGENT_COMPLETED.value == "agent_completed"
        assert EventType.AGENT_FAILED.value == "agent_failed"
        assert EventType.AGENT_MESSAGE.value == "agent_message"


class TestEvent:
    """Tests for the Event model."""

    def test_create_event(self):
        """Test creating a basic event."""
        event = Event(
            event_type=EventType.PIPELINE_STARTED,
            message="Pipeline started for task: Fix null check",
        )
        assert event.event_type == EventType.PIPELINE_STARTED
        assert "Fix null check" in event.message
        assert isinstance(event.timestamp, datetime)

    def test_event_with_stage(self):
        """Test creating an event associated with a pipeline stage."""
        event = Event(
            event_type=EventType.AGENT_STARTED,
            stage=PipelineStage.PLANNER,
            message="Planner agent starting analysis",
        )
        assert event.stage == PipelineStage.PLANNER

    def test_event_with_data(self):
        """Test creating an event with additional structured data."""
        event = Event(
            event_type=EventType.AGENT_COMPLETED,
            stage=PipelineStage.REVIEWER,
            message="Review complete",
            data={"risk_score": 3.5, "approved": True},
        )
        assert event.data["risk_score"] == 3.5


class TestEventBus:
    """Tests for the EventBus subscription and emission system."""

    @pytest.mark.asyncio
    async def test_subscribe_and_emit(self):
        """Test that a subscriber receives emitted events."""
        bus = EventBus()
        received = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe(handler)
        event = Event(
            event_type=EventType.PIPELINE_STARTED,
            message="Test pipeline started",
        )
        await bus.emit(event)
        assert len(received) == 1
        assert received[0].event_type == EventType.PIPELINE_STARTED

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self):
        """Test that multiple subscribers all receive the same event."""
        bus = EventBus()
        received_a = []
        received_b = []

        async def handler_a(event: Event) -> None:
            received_a.append(event)

        async def handler_b(event: Event) -> None:
            received_b.append(event)

        bus.subscribe(handler_a)
        bus.subscribe(handler_b)
        await bus.emit(
            Event(event_type=EventType.AGENT_STARTED, message="Start")
        )
        assert len(received_a) == 1
        assert len(received_b) == 1

    @pytest.mark.asyncio
    async def test_subscribe_to_specific_event_type(self):
        """Test that a filtered subscriber only receives matching events."""
        bus = EventBus()
        received = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe(handler, event_types=[EventType.AGENT_COMPLETED])
        await bus.emit(
            Event(event_type=EventType.AGENT_STARTED, message="Start")
        )
        await bus.emit(
            Event(event_type=EventType.AGENT_COMPLETED, message="Done")
        )
        assert len(received) == 1
        assert received[0].event_type == EventType.AGENT_COMPLETED

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """Test that an unsubscribed handler stops receiving events."""
        bus = EventBus()
        received = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe(handler)
        await bus.emit(
            Event(event_type=EventType.PIPELINE_STARTED, message="First")
        )
        bus.unsubscribe(handler)
        await bus.emit(
            Event(event_type=EventType.PIPELINE_COMPLETED, message="Second")
        )
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_emit_with_no_subscribers(self):
        """Test that emitting with no subscribers does not raise."""
        bus = EventBus()
        await bus.emit(
            Event(event_type=EventType.PIPELINE_STARTED, message="No one listening")
        )

    @pytest.mark.asyncio
    async def test_event_history(self):
        """Test that the event bus records event history."""
        bus = EventBus()
        await bus.emit(
            Event(event_type=EventType.PIPELINE_STARTED, message="Start")
        )
        await bus.emit(
            Event(event_type=EventType.PIPELINE_COMPLETED, message="Done")
        )
        assert len(bus.history) == 2
        assert bus.history[0].event_type == EventType.PIPELINE_STARTED
        assert bus.history[1].event_type == EventType.PIPELINE_COMPLETED

    @pytest.mark.asyncio
    async def test_clear_history(self):
        """Test that event history can be cleared."""
        bus = EventBus()
        await bus.emit(
            Event(event_type=EventType.PIPELINE_STARTED, message="Start")
        )
        bus.clear_history()
        assert len(bus.history) == 0
