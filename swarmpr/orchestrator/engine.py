"""Pipeline orchestration engine for SwarmPR.

Implements the state machine that runs agents in sequence, manages
state flow between them, handles errors, and emits events for
observability. This is the core coordinator of the multi-agent pipeline.
"""

import time

from swarmpr.agents.base import BaseAgent
from swarmpr.orchestrator.events import Event, EventBus, EventType
from swarmpr.orchestrator.state import (
    AgentResult,
    AgentStatus,
    PipelineState,
    PipelineStatus,
)


class PipelineEngine:
    """Orchestrates the sequential execution of SwarmPR agents.

    Runs a list of agents in order, passing shared state between them.
    Emits events at each lifecycle point for terminal logging, WebSocket
    streaming, and metrics collection.

    Attributes:
        event_bus: The event bus for emitting pipeline and agent events.
    """

    def __init__(self, event_bus: EventBus) -> None:
        """Initialize the engine with an event bus.

        Args:
            event_bus: The event bus to emit lifecycle events to.
        """
        self.event_bus = event_bus

    async def run(
        self,
        state: PipelineState,
        agents: list[BaseAgent],
    ) -> PipelineState:
        """Execute the full agent pipeline.

        Runs each agent in sequence. If an agent fails, the pipeline
        halts and records the failure. State is passed between agents
        so each can build on the previous agent's work.

        Args:
            state: The initial pipeline state with task description.
            agents: Ordered list of agents to execute.

        Returns:
            The final pipeline state after all agents have run
            (or after failure).
        """
        state.status = PipelineStatus.RUNNING

        await self.event_bus.emit(
            Event(
                event_type=EventType.PIPELINE_STARTED,
                message=f"Pipeline started for task: {state.task_description}",
            )
        )

        for agent in agents:
            state.current_stage = agent.stage

            await self.event_bus.emit(
                Event(
                    event_type=EventType.AGENT_STARTED,
                    stage=agent.stage,
                    message=f"Agent {agent.stage.value} starting",
                )
            )

            start_time = time.monotonic()

            try:
                state = await agent.execute(state)

                duration_ms = int((time.monotonic() - start_time) * 1000)

                usage = agent.get_last_usage()
                tokens_in = usage.get("prompt_tokens", 0) if usage else 0
                tokens_out = usage.get("completion_tokens", 0) if usage else 0

                result = AgentResult(
                    agent=agent.stage,
                    status=AgentStatus.COMPLETED,
                    duration_ms=duration_ms,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                )
                state.agent_results.append(result)

                await self.event_bus.emit(
                    Event(
                        event_type=EventType.AGENT_COMPLETED,
                        stage=agent.stage,
                        message=f"{agent.stage.value} completed in {duration_ms}ms",
                        data={
                            "duration_ms": duration_ms,
                            "tokens_in": tokens_in,
                            "tokens_out": tokens_out,
                        },
                    )
                )

            except Exception as exc:
                duration_ms = int((time.monotonic() - start_time) * 1000)

                result = AgentResult(
                    agent=agent.stage,
                    status=AgentStatus.FAILED,
                    duration_ms=duration_ms,
                    error=str(exc),
                )
                state.agent_results.append(result)
                state.status = PipelineStatus.FAILED

                await self.event_bus.emit(
                    Event(
                        event_type=EventType.AGENT_FAILED,
                        stage=agent.stage,
                        message=f"Agent {agent.stage.value} failed: {exc}",
                        data={"error": str(exc)},
                    )
                )

                await self.event_bus.emit(
                    Event(
                        event_type=EventType.PIPELINE_FAILED,
                        message=f"Pipeline failed at {agent.stage.value}: {exc}",
                        data={"failed_stage": agent.stage.value, "error": str(exc)},
                    )
                )

                return state

        state.status = PipelineStatus.COMPLETED
        state.current_stage = None

        await self.event_bus.emit(
            Event(
                event_type=EventType.PIPELINE_COMPLETED,
                message="Pipeline completed successfully",
                data={
                    "total_agents": len(agents),
                    "total_duration_ms": sum(
                        r.duration_ms for r in state.agent_results
                    ),
                },
            )
        )

        return state
