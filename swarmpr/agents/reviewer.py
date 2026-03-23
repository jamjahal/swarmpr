"""Reviewer agent for SwarmPR.

Performs a risk-tiered review of code changes, combining LLM-based
code analysis with the escalation policy engine to produce a final
verdict: approve, flag, or block.
"""

import json
import re

from swarmpr.agents.base import BaseAgent
from swarmpr.orchestrator.state import (
    PipelineStage,
    PipelineState,
    ReviewVerdict,
)
from swarmpr.providers.base import AgentProvider
from swarmpr.risk.classifier import RiskClassifier
from swarmpr.risk.policies import EscalationPolicy

_SYSTEM_PROMPT = """\
You are a senior code reviewer acting as a review agent. Your job is to \
review a code diff and provide a structured assessment.

You MUST respond with valid JSON only — no markdown, no explanation, no \
code fences. The JSON schema is:

{
  "risk_score": float (0.0 to 10.0, higher = more risky),
  "summary": "string — 1-3 sentence review summary",
  "findings": ["string — specific finding or concern", ...],
  "code_quality": "good | acceptable | poor"
}

Review criteria:
- Correctness: Does the code do what the task requires?
- Safety: Are there security concerns (hardcoded secrets, SQL injection, etc.)?
- Style: Does it follow established patterns in the codebase?
- Testing: Are changes covered by tests?
- Blast radius: How much of the system could be affected by a bug here?

Be concise and specific in findings. Focus on what matters.
"""


class ReviewerAgent(BaseAgent):
    """Reviews code changes and produces an escalation verdict.

    Combines LLM-based code review with the risk classifier and
    escalation policy to determine whether changes can be auto-approved,
    flagged for optional human review, or blocked for mandatory review.

    Attributes:
        classifier: The risk classifier for independent tier assessment.
        policy: The escalation policy engine for final decisions.
    """

    def __init__(
        self,
        provider: AgentProvider,
        classifier: RiskClassifier,
        policy: EscalationPolicy,
    ) -> None:
        """Initialize the reviewer with provider, classifier, and policy.

        Args:
            provider: The LLM provider for review generation.
            classifier: The risk classifier for independent assessment.
            policy: The escalation policy for final decisions.
        """
        super().__init__(provider=provider)
        self.classifier = classifier
        self.policy = policy

    @property
    def stage(self) -> PipelineStage:
        """Return the reviewer pipeline stage.

        Returns:
            PipelineStage.REVIEWER.
        """
        return PipelineStage.REVIEWER

    async def execute(self, state: PipelineState) -> PipelineState:
        """Review the code changes and produce an escalation verdict.

        Args:
            state: The pipeline state with diff, plan, and test results.

        Returns:
            The state with review verdict populated.

        Raises:
            ValueError: If the LLM response cannot be parsed as JSON.
        """
        diff = state.diff or ""
        test_info = self._format_test_info(state)
        plan_info = self._format_plan_info(state)

        user_prompt = (
            f"## Task\n{state.task_description}\n\n"
            f"## Plan\n{plan_info}\n\n"
            f"## Diff\n```\n{diff}\n```\n\n"
            f"## Test Results\n{test_info}\n\n"
            f"Review this change and respond with JSON."
        )

        response = await self.call_llm(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        review_data = self._parse_response(response)

        # Get escalation decision from policy engine.
        changed_paths = []
        if state.plan:
            changed_paths = [f.path for f in state.plan.files]

        classification = self.classifier.classify_paths(changed_paths)
        diff_lines = len(diff.splitlines()) if diff else 0

        escalation = self.policy.evaluate(
            classification,
            diff_lines=diff_lines,
            changed_paths=changed_paths,
        )

        # Determine approval: auto-approve only if tier allows it
        # AND tests passed.
        approved = (
            escalation.can_auto_merge
            and state.tests_passed is not False
        )

        verdict = ReviewVerdict(
            approved=approved,
            risk_score=review_data.get("risk_score", 5.0),
            escalation_action=escalation.action,
            summary=review_data.get("summary", ""),
            findings=review_data.get("findings", []),
        )

        state.review = verdict
        return state

    def _parse_response(self, response: str) -> dict:
        """Parse the LLM's JSON response into a dictionary.

        Args:
            response: The raw LLM response string.

        Returns:
            The parsed dictionary.

        Raises:
            ValueError: If the response cannot be parsed as valid JSON.
        """
        cleaned = response.strip()

        fence_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
        fence_match = re.search(fence_pattern, cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Could not parse reviewer response as JSON: {exc}. "
                f"Raw response: {response[:200]}"
            ) from exc

    def _format_test_info(self, state: PipelineState) -> str:
        """Format test results for the review prompt.

        Args:
            state: The pipeline state with test results.

        Returns:
            A human-readable test summary string.
        """
        if state.tests_passed is None:
            return "Tests were not run."
        status = "PASSED" if state.tests_passed else "FAILED"
        output = state.test_output or "No output"
        return f"Status: {status}\nOutput: {output}"

    def _format_plan_info(self, state: PipelineState) -> str:
        """Format the plan for the review prompt.

        Args:
            state: The pipeline state with the plan.

        Returns:
            A human-readable plan summary string.
        """
        if not state.plan:
            return "No plan available."

        lines = [
            f"Risk tier: {state.plan.risk_tier}",
            f"Complexity: {state.plan.estimated_complexity}",
            "Files:",
        ]
        for f in state.plan.files:
            lines.append(
                f"  - {f.change_type.value} {f.path}: {f.description}"
            )
        return "\n".join(lines)
