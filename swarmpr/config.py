"""Configuration loading and validation for SwarmPR.

Handles YAML config parsing and validation using Pydantic models.
Each agent, risk tier, and pipeline setting is defined as a typed model
with sensible defaults where appropriate.
"""

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class RiskAction(str, Enum):
    """Valid actions for risk tier escalation decisions.

    Attributes:
        APPROVE: Auto-approve eligible, merge-ready.
        FLAG: AI-approved, human notified for optional review.
        BLOCK: Human review mandatory, PR created as draft.
    """

    APPROVE = "approve"
    FLAG = "flag"
    BLOCK = "block"


class ProjectConfig(BaseModel):
    """Project-level configuration.

    Attributes:
        name: Human-readable project name.
        repo: GitHub repository URL or local path.
    """

    name: str
    repo: str


class ProviderConfig(BaseModel):
    """LLM provider configuration for a single agent.

    Attributes:
        model: LiteLLM model identifier (e.g., 'anthropic/claude-sonnet-4-20250514').
        api_base: Optional custom API base URL for local/self-hosted models.
        api_key: Optional API key override (defaults to environment variable).
        temperature: Sampling temperature for the model.
        max_tokens: Maximum tokens in the model response.
    """

    model: str
    api_base: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096


class RiskTierConfig(BaseModel):
    """Configuration for a single risk tier.

    Attributes:
        description: Human-readable description of what this tier covers.
        paths: File path prefixes that trigger this tier.
        keywords: Content keywords that trigger this tier.
        action: Escalation action when this tier is matched.
    """

    description: str
    paths: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    action: RiskAction


class PipelineConfig(BaseModel):
    """Pipeline execution configuration.

    Attributes:
        max_diff_lines: Maximum allowed diff size in lines.
        test_timeout_seconds: Timeout for test execution subprocess.
        forbidden_paths: Paths the coder agent must never modify.
    """

    max_diff_lines: int = 500
    test_timeout_seconds: int = 120
    forbidden_paths: list[str] = Field(default_factory=list)


class MetricsConfig(BaseModel):
    """Metrics collection configuration.

    Attributes:
        track_tokens: Whether to track token usage per agent.
        track_timing: Whether to track execution time per agent.
        track_cost: Whether to estimate cost per run.
    """

    track_tokens: bool = True
    track_timing: bool = True
    track_cost: bool = True


class SwarmPRConfig(BaseModel):
    """Top-level SwarmPR configuration.

    Attributes:
        project: Project identification and repository info.
        providers: Per-agent LLM provider configurations.
        risk_tiers: Risk tier definitions with escalation rules.
        pipeline: Pipeline execution settings.
        metrics: Metrics collection settings.
    """

    project: ProjectConfig
    providers: dict[str, ProviderConfig]
    risk_tiers: dict[str, RiskTierConfig] = Field(default_factory=dict)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)


def load_config(path: Path) -> SwarmPRConfig:
    """Load and validate a SwarmPR configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A validated SwarmPRConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
        pydantic.ValidationError: If the config fails validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got {type(raw)}")

    return SwarmPRConfig(**raw)
