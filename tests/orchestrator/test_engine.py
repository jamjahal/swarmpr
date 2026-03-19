"""Tests for the SwarmPR orchestration engine."""

import pytest

from swarmpr.agents.base import BaseAgent
from swarmpr.orchestrator.engine import PipelineEngine
from swarmpr.orchestrator.events import Event, EventBus, EventType
from swarmpr.orchestrator.state import (
    AgentStatus,
    PipelineStage,
    PipelineState,
    PipelineStatus,
)
from swarmpr.providers.base import AgentProvider, Message


class FakeProvider(AgentProvider):
    """A fake provider for testing that returns canned responses."""

    def __init__(self, response: str = "fake response") -> None:
        """Initialize with a canned response.

        Args:
            response: The string to return from complete().
        """
        self.response = response
        self.last_usage = None

    async def complete(self, messages: list[Message], **kwargs: object) -> str:
        """Return the canned response.

        Args:
            messages: Ignored in fake provider.
            **kwargs: Ignored in fake provider.

        Returns:
            The canned response string.
        """
        self.last_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        return self.response


class PassthroughAgent(BaseAgent):
    """A test agent that simply marks itself complete without modifying state."""

    @property
    def stage(self) -> PipelineStage:
        """Return the pipeline stage for this agent.

        Returns:
            The pipeline stage this agent occupies.
        """
        return self._stage

    def __init__(
        self, stage: PipelineStage, provider: AgentProvider | None = None
    ) -> None:
        """Initialize with a stage and optional provider.

        Args:
            stage: The pipeline stage this agent represents.
            provider: Optional LLM provider.
        """
        self._stage = stage
        super().__init__(provider=provider or FakeProvider())

    async def execute(self, state: PipelineState) -> PipelineState:
        """Pass state through without modification.

        Args:
            state: The current pipeline state.

        Returns:
            The unmodified pipeline state.
        """
        return state


class FailingAgent(BaseAgent):
    """A test agent that always raises an exception."""

    @property
    def stage(self) -> PipelineStage:
        """Return the pipeline stage for this agent.

        Returns:
            The pipeline stage this agent occupies.
        """
        return self._stage

    def __init__(
        self, stage: PipelineStage, provider: AgentProvider | None = None
    ) -> None:
        """Initialize with a stage and optional provider.

        Args:
            stage: The pipeline stage this agent represents.
            provider: Optional LLM provider.
        """
        self._stage = stage
        super().__init__(provider=provider or FakeProvider())

    async def execute(self, state: PipelineState) -> PipelineState:
        """Raise an error to simulate agent failure.

        Args:
            state: The current pipeline state.

        Raises:
            RuntimeError: Always raised to simulate failure.
        """
        raise RuntimeError("Simulated agent failure")


class StateModifyingAgent(BaseAgent):
    """A test agent that modifies state to prove state flows between agents."""

    @property
    def stage(self) -> PipelineStage:
        """Return the pipeline stage for this agent.

        Returns:
            The pipeline stage this agent occupies.
        """
        return self._stage

    def __init__(
        self,
        stage: PipelineStage,
        field: str,
        value: object,
        provider: AgentProvider | None = None,
    ) -> None:
        """Initialize with a stage, field to modify, and value to set.

        Args:
            stage: The pipeline stage this agent represents.
            field: The PipelineState field to modify.
            value: The value to set on the field.
            provider: Optional LLM provider.
        """
        self._stage = stage
        self._field = field
        self._value = value
        super().__init__(provider=provider or FakeProvider())

    async def execute(self, state: PipelineState) -> PipelineState:
        """Modify a field on the pipeline state.

        Args:
            state: The current pipeline state.

        Returns:
            The modified pipeline state.
        """
        setattr(state, self._field, self._value)
        return state


class TestPipelineEngine:
    """Tests for the PipelineEngine orchestrator."""

    @pytest.mark.asyncio
    async def test_engine_runs_single_agent(self):
        """Test running a pipeline with a single agent."""
        bus = EventBus()
        engine = PipelineEngine(event_bus=bus)
        agent = PassthroughAgent(stage=PipelineStage.PLANNER)

        state = PipelineState(
            task_description="Test task",
            repo_path="/tmp/repo",
        )

        result = await engine.run(state, agents=[agent])
        assert result.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_engine_runs_full_pipeline(self):
        """Test running a pipeline with all four agents."""
        bus = EventBus()
        engine = PipelineEngine(event_bus=bus)

        agents = [
            PassthroughAgent(stage=PipelineStage.PLANNER),
            PassthroughAgent(stage=PipelineStage.CODER),
            PassthroughAgent(stage=PipelineStage.TESTER),
            PassthroughAgent(stage=PipelineStage.REVIEWER),
        ]

        state = PipelineState(
            task_description="Test task",
            repo_path="/tmp/repo",
        )

        result = await engine.run(state, agents=agents)
        assert result.status == PipelineStatus.COMPLETED
        assert len(result.agent_results) == 4

    @pytest.mark.asyncio
    async def test_engine_state_flows_between_agents(self):
        """Test that state modifications persist across agent boundaries."""
        bus = EventBus()
        engine = PipelineEngine(event_bus=bus)

        agents = [
            StateModifyingAgent(
                stage=PipelineStage.PLANNER,
                field="branch_name",
                value="swarmpr/test-branch",
            ),
            StateModifyingAgent(
                stage=PipelineStage.CODER,
                field="diff",
                value="--- a/file.py\n+++ b/file.py",
            ),
        ]

        state = PipelineState(
            task_description="Test task",
            repo_path="/tmp/repo",
        )

        result = await engine.run(state, agents=agents)
        assert result.branch_name == "swarmpr/test-branch"
        assert result.diff == "--- a/file.py\n+++ b/file.py"

    @pytest.mark.asyncio
    async def test_engine_handles_agent_failure(self):
        """Test that the pipeline fails gracefully when an agent errors."""
        bus = EventBus()
        engine = PipelineEngine(event_bus=bus)

        agents = [
            PassthroughAgent(stage=PipelineStage.PLANNER),
            FailingAgent(stage=PipelineStage.CODER),
            PassthroughAgent(stage=PipelineStage.TESTER),
        ]

        state = PipelineState(
            task_description="Test task",
            repo_path="/tmp/repo",
        )

        result = await engine.run(state, agents=agents)
        assert result.status == PipelineStatus.FAILED
        # Only 2 agent results: planner succeeded, coder failed, tester never ran
        assert len(result.agent_results) == 2
        assert result.agent_results[0].status == AgentStatus.COMPLETED
        assert result.agent_results[1].status == AgentStatus.FAILED
        assert "Simulated agent failure" in result.agent_results[1].error

    @pytest.mark.asyncio
    async def test_engine_emits_pipeline_started_event(self):
        """Test that the engine emits a pipeline_started event."""
        bus = EventBus()
        engine = PipelineEngine(event_bus=bus)

        state = PipelineState(
            task_description="Test task",
            repo_path="/tmp/repo",
        )

        await engine.run(state, agents=[PassthroughAgent(stage=PipelineStage.PLANNER)])

        started_events = [
            e for e in bus.history if e.event_type == EventType.PIPELINE_STARTED
        ]
        assert len(started_events) == 1

    @pytest.mark.asyncio
    async def test_engine_emits_pipeline_completed_event(self):
        """Test that the engine emits a pipeline_completed event on success."""
        bus = EventBus()
        engine = PipelineEngine(event_bus=bus)

        state = PipelineState(
            task_description="Test task",
            repo_path="/tmp/repo",
        )

        await engine.run(state, agents=[PassthroughAgent(stage=PipelineStage.PLANNER)])

        completed_events = [
            e for e in bus.history if e.event_type == EventType.PIPELINE_COMPLETED
        ]
        assert len(completed_events) == 1

    @pytest.mark.asyncio
    async def test_engine_emits_pipeline_failed_event(self):
        """Test that the engine emits a pipeline_failed event on error."""
        bus = EventBus()
        engine = PipelineEngine(event_bus=bus)

        state = PipelineState(
            task_description="Test task",
            repo_path="/tmp/repo",
        )

        await engine.run(state, agents=[FailingAgent(stage=PipelineStage.PLANNER)])

        failed_events = [
            e for e in bus.history if e.event_type == EventType.PIPELINE_FAILED
        ]
        assert len(failed_events) == 1

    @pytest.mark.asyncio
    async def test_engine_emits_agent_started_and_completed_events(self):
        """Test that agent lifecycle events are emitted for each agent."""
        bus = EventBus()
        engine = PipelineEngine(event_bus=bus)

        agents = [
            PassthroughAgent(stage=PipelineStage.PLANNER),
            PassthroughAgent(stage=PipelineStage.CODER),
        ]

        state = PipelineState(
            task_description="Test task",
            repo_path="/tmp/repo",
        )

        await engine.run(state, agents=agents)

        agent_started = [
            e for e in bus.history if e.event_type == EventType.AGENT_STARTED
        ]
        agent_completed = [
            e for e in bus.history if e.event_type == EventType.AGENT_COMPLETED
        ]
        assert len(agent_started) == 2
        assert len(agent_completed) == 2

    @pytest.mark.asyncio
    async def test_engine_tracks_agent_duration(self):
        """Test that agent results include execution duration."""
        bus = EventBus()
        engine = PipelineEngine(event_bus=bus)

        state = PipelineState(
            task_description="Test task",
            repo_path="/tmp/repo",
        )

        result = await engine.run(
            state, agents=[PassthroughAgent(stage=PipelineStage.PLANNER)]
        )
        assert result.agent_results[0].duration_ms >= 0

    @pytest.mark.asyncio
    async def test_engine_updates_current_stage(self):
        """Test that the engine updates current_stage as it progresses."""
        bus = EventBus()
        stages_seen = []

        async def track_stage(event: Event) -> None:
            if event.event_type == EventType.AGENT_STARTED:
                stages_seen.append(event.stage)

        bus.subscribe(track_stage, event_types=[EventType.AGENT_STARTED])
        engine = PipelineEngine(event_bus=bus)

        agents = [
            PassthroughAgent(stage=PipelineStage.PLANNER),
            PassthroughAgent(stage=PipelineStage.CODER),
            PassthroughAgent(stage=PipelineStage.TESTER),
        ]

        state = PipelineState(
            task_description="Test task",
            repo_path="/tmp/repo",
        )

        await engine.run(state, agents=agents)
        assert stages_seen == [
            PipelineStage.PLANNER,
            PipelineStage.CODER,
            PipelineStage.TESTER,
        ]

    @pytest.mark.asyncio
    async def test_engine_with_empty_agent_list(self):
        """Test that running with no agents completes immediately."""
        bus = EventBus()
        engine = PipelineEngine(event_bus=bus)

        state = PipelineState(
            task_description="Test task",
            repo_path="/tmp/repo",
        )

        result = await engine.run(state, agents=[])
        assert result.status == PipelineStatus.COMPLETED
        assert len(result.agent_results) == 0
