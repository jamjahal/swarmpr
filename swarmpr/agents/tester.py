"""Tester agent for SwarmPR.

Executes the project's test suite against code changes to verify
correctness. Runs tests as a subprocess with configurable timeout
to prevent hangs.
"""

import asyncio

from swarmpr.agents.base import BaseAgent
from swarmpr.config import PipelineConfig
from swarmpr.orchestrator.state import PipelineStage, PipelineState
from swarmpr.providers.base import AgentProvider


class TesterAgent(BaseAgent):
    """Runs tests against the coder's changes.

    Executes pytest as a subprocess with a timeout. Records pass/fail
    status and raw output on the pipeline state.

    Note: Uses create_subprocess_exec (not shell=True) to avoid
    command injection. Arguments are passed as a list, not a string.

    Attributes:
        pipeline_config: Pipeline constraints (timeout, etc.).
        timeout_seconds: Maximum seconds to wait for test completion.
    """

    def __init__(
        self,
        provider: AgentProvider,
        pipeline_config: PipelineConfig,
    ) -> None:
        """Initialize the tester with a provider and pipeline config.

        Args:
            provider: The LLM provider (used for test analysis if needed).
            pipeline_config: Pipeline constraints including timeout.
        """
        super().__init__(provider=provider)
        self.pipeline_config = pipeline_config
        self.timeout_seconds = pipeline_config.test_timeout_seconds

    @property
    def stage(self) -> PipelineStage:
        """Return the tester pipeline stage.

        Returns:
            PipelineStage.TESTER.
        """
        return PipelineStage.TESTER

    async def execute(self, state: PipelineState) -> PipelineState:
        """Run the test suite and record results on the state.

        Args:
            state: The pipeline state with code changes applied.

        Returns:
            The state with test_output and tests_passed populated.
        """
        try:
            stdout, stderr, return_code = await self._run_tests(
                state.repo_path
            )

            output_parts = []
            if stdout:
                output_parts.append(stdout)
            if stderr:
                output_parts.append(stderr)

            state.test_output = "\n".join(output_parts) or "No output"

            if return_code == 0:
                state.tests_passed = True
            elif return_code == 5:
                # pytest exit code 5 = no tests collected
                state.tests_passed = True
                state.test_output = (
                    f"No tests collected.\n{state.test_output}"
                )
            else:
                state.tests_passed = False

        except asyncio.TimeoutError:
            state.tests_passed = False
            state.test_output = (
                f"Test execution timed out after "
                f"{self.timeout_seconds} seconds."
            )

        return state

    async def _run_tests(
        self, repo_path: str
    ) -> tuple[str, str, int]:
        """Run the test suite as a subprocess.

        Uses create_subprocess_exec (not shell) to prevent command
        injection. All arguments are passed as separate list elements.

        Args:
            repo_path: Path to the repository root.

        Returns:
            A tuple of (stdout, stderr, return_code).

        Raises:
            asyncio.TimeoutError: If tests exceed the configured timeout.
        """
        proc = await asyncio.create_subprocess_exec(
            "python",
            "-m",
            "pytest",
            "--tb=short",
            "-q",
            cwd=repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout_seconds,
            )
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            return stdout, stderr, proc.returncode or 0

        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise
