"""Pipeline state models for SwarmPR orchestration.

Defines the shared state that flows between agents in the pipeline,
plus supporting models for plans, reviews, file changes, and results.
All models are Pydantic v2 for validation and serialization.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class PipelineStage(str, Enum):
    """Stages in the SwarmPR agent pipeline.

    Attributes:
        PLANNER: Task decomposition and blast radius assessment.
        CODER: Code generation based on the plan.
        TESTER: Test execution and generation.
        REVIEWER: Risk-tiered review and escalation decision.
    """

    PLANNER = "planner"
    CODER = "coder"
    TESTER = "tester"
    REVIEWER = "reviewer"


class PipelineStatus(str, Enum):
    """Overall pipeline execution status.

    Attributes:
        PENDING: Pipeline created but not yet started.
        RUNNING: Pipeline is actively executing.
        COMPLETED: Pipeline finished successfully.
        FAILED: Pipeline encountered an unrecoverable error.
        AWAITING_HUMAN: Pipeline paused for human review.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    AWAITING_HUMAN = "awaiting_human"


class AgentStatus(str, Enum):
    """Individual agent execution status.

    Attributes:
        PENDING: Agent has not started.
        RUNNING: Agent is currently executing.
        COMPLETED: Agent finished successfully.
        FAILED: Agent encountered an error.
        SKIPPED: Agent was skipped (e.g., tester skipped if no tests).
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class FileChangeType(str, Enum):
    """Types of file changes in a task plan.

    Attributes:
        CREATE: New file to be created.
        MODIFY: Existing file to be modified.
        DELETE: Existing file to be deleted.
    """

    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"


class FileChange(BaseModel):
    """A single file change in a task plan.

    Attributes:
        path: Relative path to the file within the repo.
        change_type: Whether the file is created, modified, or deleted.
        description: Human-readable description of the change.
    """

    path: str
    change_type: FileChangeType
    description: str = ""


class TaskPlan(BaseModel):
    """Structured output from the planner agent.

    Attributes:
        task_description: The original task description.
        files: List of file changes to implement.
        risk_tier: Assigned blast radius tier (1-3).
        risk_justification: Explanation for the tier assignment.
        estimated_complexity: Rough complexity estimate.
    """

    task_description: str
    files: list[FileChange] = Field(default_factory=list)
    risk_tier: int = Field(ge=1, le=3)
    risk_justification: str = ""
    estimated_complexity: str = "low"

    @field_validator("risk_tier")
    @classmethod
    def validate_risk_tier(cls, v: int) -> int:
        """Validate that risk_tier is between 1 and 3.

        Args:
            v: The risk tier value to validate.

        Returns:
            The validated risk tier value.

        Raises:
            ValueError: If the tier is not between 1 and 3.
        """
        if not 1 <= v <= 3:
            raise ValueError("risk_tier must be between 1 and 3")
        return v


class ReviewVerdict(BaseModel):
    """Structured output from the reviewer agent.

    Attributes:
        approved: Whether the PR is approved by the AI reviewer.
        risk_score: Composite risk score (0-10 scale).
        escalation_action: The escalation decision (approve/flag/block).
        summary: Human-readable review summary.
        findings: List of specific findings or concerns.
    """

    approved: bool
    risk_score: float
    escalation_action: str
    summary: str
    findings: list[str] = Field(default_factory=list)


class AgentResult(BaseModel):
    """Result metadata from a single agent execution.

    Attributes:
        agent: Which pipeline stage this result is from.
        status: Execution status of the agent.
        duration_ms: Execution time in milliseconds.
        tokens_in: Number of input tokens consumed.
        tokens_out: Number of output tokens generated.
        error: Error message if the agent failed.
    """

    agent: PipelineStage
    status: AgentStatus
    duration_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    error: str | None = None


class PipelineState(BaseModel):
    """Shared state that flows through the entire SwarmPR pipeline.

    Each agent reads from and writes to this state object. The orchestrator
    manages the state lifecycle and passes it between agents.

    Attributes:
        task_description: The original task or PRD text.
        repo_path: Path to the target repository.
        status: Overall pipeline status.
        current_stage: Which agent is currently executing.
        branch_name: Git branch created for this pipeline run.
        plan: Structured plan from the planner agent.
        diff: Generated code diff from the coder agent.
        test_output: Raw test execution output from the tester agent.
        tests_passed: Whether all tests passed.
        review: Review verdict from the reviewer agent.
        agent_results: Execution metadata from each agent.
        pr_url: URL of the created GitHub PR.
        created_at: When the pipeline run was created.
    """

    task_description: str
    repo_path: str
    status: PipelineStatus = PipelineStatus.PENDING
    current_stage: PipelineStage | None = None
    branch_name: str | None = None
    plan: TaskPlan | None = None
    diff: str | None = None
    test_output: str | None = None
    tests_passed: bool | None = None
    review: ReviewVerdict | None = None
    agent_results: list[AgentResult] = Field(default_factory=list)
    pr_url: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
