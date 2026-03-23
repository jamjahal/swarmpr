"""Planner agent for SwarmPR.

Analyzes a task description and target repository to produce a structured
plan: which files to change, what to do in each, and the blast radius
tier. This drives the rest of the pipeline.
"""

import json
import re

from swarmpr.agents.base import BaseAgent
from swarmpr.orchestrator.state import (
    FileChange,
    FileChangeType,
    PipelineStage,
    PipelineState,
    TaskPlan,
)
from swarmpr.providers.base import AgentProvider
from swarmpr.risk.classifier import RiskClassifier

_SYSTEM_PROMPT = """\
You are a senior software engineer acting as a planning agent. Your job is to \
analyze a task description and produce a structured implementation plan.

You MUST respond with valid JSON only — no markdown, no explanation, no \
code fences. The JSON schema is:

{
  "task_description": "string — restate the task clearly",
  "files": [
    {
      "path": "string — relative file path",
      "change_type": "create | modify | delete",
      "description": "string — what to change in this file"
    }
  ],
  "estimated_complexity": "low | medium | high"
}

Guidelines:
- Be specific about file paths and change descriptions.
- Prefer modifying existing files over creating new ones.
- Include test files if the change warrants new tests.
- Keep the plan minimal — only the files needed for the task.
- Do NOT include risk_tier — that will be assigned separately.
"""


class PlannerAgent(BaseAgent):
    """Plans task implementation and assigns blast radius tier.

    Takes a task description, sends it to the LLM for decomposition,
    then uses the RiskClassifier to assign a risk tier based on the
    planned file changes.

    Attributes:
        classifier: The risk classifier for tier assignment.
    """

    def __init__(
        self,
        provider: AgentProvider,
        classifier: RiskClassifier,
    ) -> None:
        """Initialize the planner with a provider and risk classifier.

        Args:
            provider: The LLM provider for plan generation.
            classifier: The risk classifier for tier assignment.
        """
        super().__init__(provider=provider)
        self.classifier = classifier

    @property
    def stage(self) -> PipelineStage:
        """Return the planner pipeline stage.

        Returns:
            PipelineStage.PLANNER.
        """
        return PipelineStage.PLANNER

    async def execute(self, state: PipelineState) -> PipelineState:
        """Generate a task plan and assign a risk tier.

        Args:
            state: The current pipeline state with task_description.

        Returns:
            The state with plan, branch_name, and risk tier populated.

        Raises:
            ValueError: If the LLM response cannot be parsed as JSON.
        """
        user_prompt = (
            f"Task: {state.task_description}\n\n"
            f"Repository path: {state.repo_path}\n\n"
            f"Produce a JSON implementation plan."
        )

        response = await self.call_llm(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        plan_data = self._parse_response(response)

        # Classify risk based on planned file paths.
        paths = [f["path"] for f in plan_data.get("files", [])]
        classification = self.classifier.classify_paths(paths)

        files = [
            FileChange(
                path=f["path"],
                change_type=FileChangeType(f["change_type"]),
                description=f.get("description", ""),
            )
            for f in plan_data.get("files", [])
        ]

        plan = TaskPlan(
            task_description=plan_data.get(
                "task_description", state.task_description
            ),
            files=files,
            risk_tier=classification.tier,
            risk_justification=classification.justification,
            estimated_complexity=plan_data.get(
                "estimated_complexity", "medium"
            ),
        )

        state.plan = plan
        state.branch_name = self._generate_branch_name(
            state.task_description
        )

        return state

    def _parse_response(self, response: str) -> dict:
        """Parse the LLM's JSON response into a dictionary.

        Handles common LLM quirks like markdown code fences around JSON.

        Args:
            response: The raw LLM response string.

        Returns:
            The parsed dictionary.

        Raises:
            ValueError: If the response cannot be parsed as valid JSON.
        """
        cleaned = response.strip()

        # Strip markdown code fences if present.
        fence_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
        fence_match = re.search(fence_pattern, cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Could not parse planner response as JSON: {exc}. "
                f"Raw response: {response[:200]}"
            ) from exc

    def _generate_branch_name(self, task_description: str) -> str:
        """Generate a git branch name from the task description.

        Args:
            task_description: The original task text.

        Returns:
            A sanitized branch name like 'swarmpr/fix-null-check'.
        """
        slug = task_description.lower()
        slug = re.sub(r"[^a-z0-9\s-]", "", slug)
        slug = re.sub(r"\s+", "-", slug).strip("-")
        slug = slug[:50]
        return f"swarmpr/{slug}"
