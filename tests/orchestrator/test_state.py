"""Tests for pipeline state models."""


import pytest

from swarmpr.orchestrator.state import (
    AgentResult,
    AgentStatus,
    FileChange,
    FileChangeType,
    PipelineStage,
    PipelineState,
    PipelineStatus,
    ReviewVerdict,
    TaskPlan,
)


class TestPipelineStage:
    """Tests for the PipelineStage enum."""

    def test_stages_exist(self):
        """Test that all four pipeline stages are defined."""
        assert PipelineStage.PLANNER.value == "planner"
        assert PipelineStage.CODER.value == "coder"
        assert PipelineStage.TESTER.value == "tester"
        assert PipelineStage.REVIEWER.value == "reviewer"


class TestPipelineStatus:
    """Tests for the PipelineStatus enum."""

    def test_statuses_exist(self):
        """Test that all pipeline statuses are defined."""
        assert PipelineStatus.PENDING.value == "pending"
        assert PipelineStatus.RUNNING.value == "running"
        assert PipelineStatus.COMPLETED.value == "completed"
        assert PipelineStatus.FAILED.value == "failed"
        assert PipelineStatus.AWAITING_HUMAN.value == "awaiting_human"


class TestFileChange:
    """Tests for the FileChange model."""

    def test_create_file_change(self):
        """Test creating a file change entry."""
        change = FileChange(
            path="payments/processor.py",
            change_type=FileChangeType.MODIFY,
            description="Fix null check in validate_amount",
        )
        assert change.path == "payments/processor.py"
        assert change.change_type == FileChangeType.MODIFY

    def test_file_change_types(self):
        """Test all file change types are defined."""
        assert FileChangeType.CREATE.value == "create"
        assert FileChangeType.MODIFY.value == "modify"
        assert FileChangeType.DELETE.value == "delete"


class TestTaskPlan:
    """Tests for the TaskPlan model produced by the planner agent."""

    def test_create_task_plan(self):
        """Test creating a valid task plan."""
        plan = TaskPlan(
            task_description="Fix null check in payment validation",
            files=[
                FileChange(
                    path="payments/validator.py",
                    change_type=FileChangeType.MODIFY,
                    description="Add null check",
                ),
            ],
            risk_tier=3,
            risk_justification="Touches payment validation logic",
            estimated_complexity="low",
        )
        assert plan.risk_tier == 3
        assert len(plan.files) == 1

    def test_task_plan_risk_tier_must_be_1_to_3(self):
        """Test that risk tier must be between 1 and 3."""
        with pytest.raises(Exception):
            TaskPlan(
                task_description="Test",
                files=[],
                risk_tier=5,
                risk_justification="Invalid",
                estimated_complexity="low",
            )

    def test_task_plan_risk_tier_minimum(self):
        """Test that risk tier cannot be less than 1."""
        with pytest.raises(Exception):
            TaskPlan(
                task_description="Test",
                files=[],
                risk_tier=0,
                risk_justification="Invalid",
                estimated_complexity="low",
            )


class TestReviewVerdict:
    """Tests for the ReviewVerdict model."""

    def test_create_review_verdict(self):
        """Test creating a review verdict."""
        verdict = ReviewVerdict(
            approved=False,
            risk_score=8.5,
            escalation_action="block",
            summary="Changes touch payment processing logic.",
            findings=["Modifies payment validation without test coverage"],
        )
        assert verdict.approved is False
        assert verdict.risk_score == 8.5
        assert verdict.escalation_action == "block"

    def test_review_verdict_approved(self):
        """Test an approved review verdict."""
        verdict = ReviewVerdict(
            approved=True,
            risk_score=1.0,
            escalation_action="approve",
            summary="Config-only change, low risk.",
            findings=[],
        )
        assert verdict.approved is True


class TestAgentResult:
    """Tests for the AgentResult model."""

    def test_create_agent_result(self):
        """Test creating an agent result."""
        result = AgentResult(
            agent=PipelineStage.PLANNER,
            status=AgentStatus.COMPLETED,
            duration_ms=1500,
            tokens_in=200,
            tokens_out=150,
        )
        assert result.agent == PipelineStage.PLANNER
        assert result.status == AgentStatus.COMPLETED
        assert result.duration_ms == 1500

    def test_agent_result_with_error(self):
        """Test an agent result that records a failure."""
        result = AgentResult(
            agent=PipelineStage.CODER,
            status=AgentStatus.FAILED,
            error="API rate limit exceeded",
        )
        assert result.status == AgentStatus.FAILED
        assert "rate limit" in result.error


class TestPipelineState:
    """Tests for the top-level PipelineState model."""

    def test_create_initial_state(self):
        """Test creating a fresh pipeline state."""
        state = PipelineState(
            task_description="Fix the null check in payment validation",
            repo_path="/path/to/repo",
        )
        assert state.task_description == "Fix the null check in payment validation"
        assert state.status == PipelineStatus.PENDING
        assert state.current_stage is None
        assert state.plan is None
        assert state.review is None
        assert state.agent_results == []

    def test_state_tracks_current_stage(self):
        """Test updating the current pipeline stage."""
        state = PipelineState(
            task_description="Add endpoint",
            repo_path="/path/to/repo",
        )
        state.current_stage = PipelineStage.PLANNER
        state.status = PipelineStatus.RUNNING
        assert state.current_stage == PipelineStage.PLANNER
        assert state.status == PipelineStatus.RUNNING

    def test_state_accumulates_agent_results(self):
        """Test that agent results accumulate as the pipeline progresses."""
        state = PipelineState(
            task_description="Test task",
            repo_path="/path/to/repo",
        )
        result = AgentResult(
            agent=PipelineStage.PLANNER,
            status=AgentStatus.COMPLETED,
            duration_ms=1000,
        )
        state.agent_results.append(result)
        assert len(state.agent_results) == 1
        assert state.agent_results[0].agent == PipelineStage.PLANNER

    def test_state_serialization_roundtrip(self):
        """Test that state can be serialized and deserialized."""
        state = PipelineState(
            task_description="Test task",
            repo_path="/path/to/repo",
            status=PipelineStatus.RUNNING,
            current_stage=PipelineStage.CODER,
        )
        json_str = state.model_dump_json()
        restored = PipelineState.model_validate_json(json_str)
        assert restored.task_description == state.task_description
        assert restored.status == state.status
        assert restored.current_stage == state.current_stage

    def test_state_holds_branch_name(self):
        """Test that state tracks the git branch name."""
        state = PipelineState(
            task_description="Test task",
            repo_path="/path/to/repo",
            branch_name="swarmpr/fix-null-check",
        )
        assert state.branch_name == "swarmpr/fix-null-check"

    def test_state_holds_diff(self):
        """Test that state can hold the generated diff."""
        state = PipelineState(
            task_description="Test task",
            repo_path="/path/to/repo",
        )
        state.diff = "--- a/payments/validator.py\n+++ b/payments/validator.py"
        assert "payments/validator.py" in state.diff

    def test_state_holds_test_results(self):
        """Test that state can hold test execution results."""
        state = PipelineState(
            task_description="Test task",
            repo_path="/path/to/repo",
        )
        state.test_output = "5 passed, 0 failed"
        state.tests_passed = True
        assert state.tests_passed is True
