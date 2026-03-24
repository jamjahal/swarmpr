"""Metrics collection and reporting for SwarmPR pipeline runs.

Aggregates timing, token usage, cost estimates, and quality signals
from pipeline state into structured reports for CLI output, PR bodies,
and API endpoints.
"""

from pydantic import BaseModel, Field

from swarmpr.orchestrator.state import PipelineState

# Rough cost estimates per 1M tokens (USD) for common models.
_COST_PER_1M_INPUT: dict[str, float] = {
    "anthropic/claude-sonnet-4-20250514": 3.0,
    "anthropic/claude-haiku-4-5-20251001": 0.80,
    "openai/gpt-4o": 2.50,
    "openai/gpt-4o-mini": 0.15,
    "default": 2.0,
}

_COST_PER_1M_OUTPUT: dict[str, float] = {
    "anthropic/claude-sonnet-4-20250514": 15.0,
    "anthropic/claude-haiku-4-5-20251001": 4.0,
    "openai/gpt-4o": 10.0,
    "openai/gpt-4o-mini": 0.60,
    "default": 8.0,
}


class AgentMetrics(BaseModel):
    """Metrics for a single agent execution.

    Attributes:
        agent: The agent stage name.
        duration_ms: Execution time in milliseconds.
        tokens_in: Input tokens consumed.
        tokens_out: Output tokens generated.
        estimated_cost_usd: Estimated cost in USD.
    """

    agent: str
    duration_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    estimated_cost_usd: float = 0.0


class PipelineMetrics(BaseModel):
    """Aggregated metrics for a complete pipeline run.

    Attributes:
        total_duration_ms: End-to-end pipeline time.
        total_tokens_in: Total input tokens across all agents.
        total_tokens_out: Total output tokens across all agents.
        total_tokens: Combined input + output tokens.
        estimated_cost_usd: Total estimated cost in USD.
        risk_tier: Assigned blast radius tier.
        risk_score: Reviewer's composite risk score.
        tests_passed: Whether tests passed.
        escalation_action: Final escalation decision.
        agents: Per-agent metrics breakdown.
    """

    total_duration_ms: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    risk_tier: int | None = None
    risk_score: float | None = None
    tests_passed: bool | None = None
    escalation_action: str | None = None
    agents: list[AgentMetrics] = Field(default_factory=list)


class MetricsCollector:
    """Collects and aggregates pipeline metrics from state.

    Reads agent results and review data from the pipeline state
    to produce a structured metrics report.
    """

    def collect(self, state: PipelineState) -> PipelineMetrics:
        """Collect metrics from a completed pipeline state.

        Args:
            state: The pipeline state after execution.

        Returns:
            Aggregated pipeline metrics.
        """
        agent_metrics = []
        total_duration = 0
        total_in = 0
        total_out = 0
        total_cost = 0.0

        for result in state.agent_results:
            cost = self._estimate_cost(
                result.tokens_in, result.tokens_out
            )
            am = AgentMetrics(
                agent=result.agent.value,
                duration_ms=result.duration_ms,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                estimated_cost_usd=round(cost, 6),
            )
            agent_metrics.append(am)
            total_duration += result.duration_ms
            total_in += result.tokens_in
            total_out += result.tokens_out
            total_cost += cost

        return PipelineMetrics(
            total_duration_ms=total_duration,
            total_tokens_in=total_in,
            total_tokens_out=total_out,
            total_tokens=total_in + total_out,
            estimated_cost_usd=round(total_cost, 6),
            risk_tier=state.plan.risk_tier if state.plan else None,
            risk_score=(
                state.review.risk_score if state.review else None
            ),
            tests_passed=state.tests_passed,
            escalation_action=(
                state.review.escalation_action
                if state.review
                else None
            ),
            agents=agent_metrics,
        )

    def _estimate_cost(
        self, tokens_in: int, tokens_out: int, model: str = "default"
    ) -> float:
        """Estimate the USD cost for a given token count.

        Args:
            tokens_in: Number of input tokens.
            tokens_out: Number of output tokens.
            model: The model identifier for cost lookup.

        Returns:
            Estimated cost in USD.
        """
        input_rate = _COST_PER_1M_INPUT.get(
            model, _COST_PER_1M_INPUT["default"]
        )
        output_rate = _COST_PER_1M_OUTPUT.get(
            model, _COST_PER_1M_OUTPUT["default"]
        )
        input_cost = (tokens_in / 1_000_000) * input_rate
        output_cost = (tokens_out / 1_000_000) * output_rate
        return input_cost + output_cost

    def format_summary(self, metrics: PipelineMetrics) -> str:
        """Format metrics as a human-readable summary for CLI output.

        Args:
            metrics: The collected pipeline metrics.

        Returns:
            A formatted multi-line string.
        """
        lines = [
            "Pipeline Metrics",
            "=" * 40,
            f"Total Duration:    {metrics.total_duration_ms}ms",
            f"Total Tokens:      {metrics.total_tokens}",
            f"  Input:           {metrics.total_tokens_in}",
            f"  Output:          {metrics.total_tokens_out}",
            f"Est. Cost:         ${metrics.estimated_cost_usd:.4f}",
            f"Risk Tier:         {metrics.risk_tier or 'N/A'}",
            f"Risk Score:        {metrics.risk_score or 'N/A'}",
            f"Tests Passed:      {metrics.tests_passed}",
            f"Escalation:        {metrics.escalation_action or 'N/A'}",
            "",
            "Per-Agent Breakdown:",
            "-" * 40,
        ]

        for am in metrics.agents:
            lines.append(
                f"  {am.agent:12s} "
                f"{am.duration_ms:6d}ms  "
                f"{am.tokens_in + am.tokens_out:6d} tokens  "
                f"${am.estimated_cost_usd:.4f}"
            )

        return "\n".join(lines)
