"""Coder agent for SwarmPR.

Generates code changes based on the planner's structured plan.
Produces file contents and a unified diff, respecting forbidden
path constraints and diff size limits.
"""

import json
import re

from swarmpr.agents.base import BaseAgent
from swarmpr.config import PipelineConfig
from swarmpr.orchestrator.state import PipelineStage, PipelineState
from swarmpr.providers.base import AgentProvider

_SYSTEM_PROMPT = """\
You are a senior software engineer acting as a coding agent. Your job is to \
implement code changes according to a structured plan.

You MUST respond with valid JSON only — no markdown, no explanation, no \
code fences. The JSON schema is:

{
  "files": [
    {
      "path": "string — relative file path (must match the plan)",
      "action": "create | modify | delete",
      "content": "string — the complete file content after changes"
    }
  ]
}

Guidelines:
- Implement EXACTLY what the plan describes, nothing more.
- For 'modify' actions, return the COMPLETE file content, not just the diff.
- For 'delete' actions, set content to an empty string.
- Write clean, idiomatic code following the project's existing patterns.
- Include docstrings and type hints where appropriate.
- Do NOT modify files outside the plan.
"""


class CoderAgent(BaseAgent):
    """Generates code changes from the planner's structured plan.

    Takes the plan's file list, asks the LLM to implement each change,
    validates against forbidden paths, and stores results as a diff
    and generated file map on the pipeline state.

    Attributes:
        pipeline_config: Pipeline constraints (forbidden paths, etc.).
    """

    def __init__(
        self,
        provider: AgentProvider,
        pipeline_config: PipelineConfig,
    ) -> None:
        """Initialize the coder with a provider and pipeline config.

        Args:
            provider: The LLM provider for code generation.
            pipeline_config: Pipeline constraints.
        """
        super().__init__(provider=provider)
        self.pipeline_config = pipeline_config

    @property
    def stage(self) -> PipelineStage:
        """Return the coder pipeline stage.

        Returns:
            PipelineStage.CODER.
        """
        return PipelineStage.CODER

    async def execute(self, state: PipelineState) -> PipelineState:
        """Generate code changes and store them on the state.

        Args:
            state: The pipeline state with plan populated.

        Returns:
            The state with generated_files and diff populated.

        Raises:
            ValueError: If the response is unparseable or touches
                forbidden paths.
        """
        if not state.plan:
            raise ValueError("Coder requires a plan on the state.")

        # Validate plan paths against forbidden list.
        self._check_forbidden_paths(state)

        plan_description = self._format_plan(state)

        user_prompt = (
            f"## Task\n{state.task_description}\n\n"
            f"## Implementation Plan\n{plan_description}\n\n"
            f"Implement the changes and respond with JSON."
        )

        response = await self.call_llm(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        file_data = self._parse_response(response)

        # Validate response paths against forbidden list.
        for file_entry in file_data.get("files", []):
            path = file_entry.get("path", "")
            self._validate_path(path)

        # Store generated files and build diff.
        generated_files: dict[str, str] = {}
        diff_parts: list[str] = []

        for file_entry in file_data.get("files", []):
            path = file_entry["path"]
            content = file_entry.get("content", "")
            action = file_entry.get("action", "modify")

            generated_files[path] = content
            diff_parts.append(
                self._build_diff_entry(path, action, content)
            )

        state.generated_files = generated_files
        state.diff = "\n".join(diff_parts)

        return state

    def _check_forbidden_paths(self, state: PipelineState) -> None:
        """Validate that the plan doesn't touch forbidden paths.

        Args:
            state: The pipeline state with plan.

        Raises:
            ValueError: If any planned file is in a forbidden path.
        """
        for file_change in state.plan.files:
            self._validate_path(file_change.path)

    def _validate_path(self, path: str) -> None:
        """Check a single path against the forbidden list.

        Args:
            path: The file path to validate.

        Raises:
            ValueError: If the path matches a forbidden pattern.
        """
        for forbidden in self.pipeline_config.forbidden_paths:
            if path == forbidden or path.startswith(forbidden):
                raise ValueError(
                    f"Path '{path}' is forbidden by pipeline config. "
                    f"Matched forbidden pattern: '{forbidden}'"
                )

    def _format_plan(self, state: PipelineState) -> str:
        """Format the plan for the coder prompt.

        Args:
            state: The pipeline state with plan.

        Returns:
            A human-readable plan description.
        """
        lines = []
        for f in state.plan.files:
            lines.append(
                f"- {f.change_type.value} `{f.path}`: {f.description}"
            )
        return "\n".join(lines)

    def _parse_response(self, response: str) -> dict:
        """Parse the LLM's JSON response.

        Args:
            response: The raw LLM response string.

        Returns:
            The parsed dictionary.

        Raises:
            ValueError: If the response cannot be parsed as JSON.
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
                f"Could not parse coder response as JSON: {exc}. "
                f"Raw response: {response[:200]}"
            ) from exc

    def _build_diff_entry(
        self, path: str, action: str, content: str
    ) -> str:
        """Build a unified diff entry for a single file.

        Args:
            path: The file path.
            action: The change action (create/modify/delete).
            content: The new file content.

        Returns:
            A unified diff string for the file.
        """
        if action == "create":
            header = f"--- /dev/null\n+++ b/{path}"
            lines = "\n".join(
                f"+{line}" for line in content.splitlines()
            )
        elif action == "delete":
            header = f"--- a/{path}\n+++ /dev/null"
            lines = ""
        else:
            header = f"--- a/{path}\n+++ b/{path}"
            lines = "\n".join(
                f"+{line}" for line in content.splitlines()
            )

        return f"{header}\n{lines}"
