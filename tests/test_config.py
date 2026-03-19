"""Tests for SwarmPR configuration loading and validation."""

from pathlib import Path

import pytest

from swarmpr.config import (
    MetricsConfig,
    PipelineConfig,
    ProjectConfig,
    ProviderConfig,
    RiskTierConfig,
    SwarmPRConfig,
    load_config,
)


class TestProjectConfig:
    """Tests for the ProjectConfig model."""

    def test_valid_project_config(self):
        """Test creating a valid project configuration."""
        config = ProjectConfig(
            name="test-project",
            repo="https://github.com/test/repo",
        )
        assert config.name == "test-project"
        assert config.repo == "https://github.com/test/repo"

    def test_project_config_requires_name(self):
        """Test that project config requires a name field."""
        with pytest.raises(Exception):
            ProjectConfig(repo="https://github.com/test/repo")

    def test_project_config_requires_repo(self):
        """Test that project config requires a repo field."""
        with pytest.raises(Exception):
            ProjectConfig(name="test-project")


class TestProviderConfig:
    """Tests for the ProviderConfig model."""

    def test_valid_provider_config(self):
        """Test creating a valid provider configuration."""
        config = ProviderConfig(model="anthropic/claude-sonnet-4-20250514")
        assert config.model == "anthropic/claude-sonnet-4-20250514"
        assert config.api_base is None

    def test_provider_config_with_api_base(self):
        """Test provider config with a custom API base URL."""
        config = ProviderConfig(
            model="ollama/llama3",
            api_base="http://192.168.1.100:11434",
        )
        assert config.api_base == "http://192.168.1.100:11434"

    def test_provider_config_requires_model(self):
        """Test that provider config requires a model field."""
        with pytest.raises(Exception):
            ProviderConfig()


class TestRiskTierConfig:
    """Tests for the RiskTierConfig model."""

    def test_valid_risk_tier(self):
        """Test creating a valid risk tier configuration."""
        config = RiskTierConfig(
            description="Critical path",
            paths=["payments/", "auth/"],
            keywords=["api_key", "secret"],
            action="block",
        )
        assert config.action == "block"
        assert "payments/" in config.paths
        assert "api_key" in config.keywords

    def test_risk_tier_action_validation(self):
        """Test that risk tier action must be approve, flag, or block."""
        with pytest.raises(Exception):
            RiskTierConfig(
                description="Invalid",
                paths=[],
                keywords=[],
                action="invalid_action",
            )

    def test_risk_tier_defaults_keywords_to_empty(self):
        """Test that keywords default to an empty list."""
        config = RiskTierConfig(
            description="Low risk",
            paths=["config/"],
            action="approve",
        )
        assert config.keywords == []


class TestPipelineConfig:
    """Tests for the PipelineConfig model."""

    def test_valid_pipeline_config(self):
        """Test creating a valid pipeline configuration."""
        config = PipelineConfig(
            max_diff_lines=500,
            test_timeout_seconds=120,
            forbidden_paths=[".env", "secrets/"],
        )
        assert config.max_diff_lines == 500
        assert config.test_timeout_seconds == 120

    def test_pipeline_config_defaults(self):
        """Test that pipeline config has sensible defaults."""
        config = PipelineConfig()
        assert config.max_diff_lines > 0
        assert config.test_timeout_seconds > 0
        assert isinstance(config.forbidden_paths, list)


class TestMetricsConfig:
    """Tests for the MetricsConfig model."""

    def test_valid_metrics_config(self):
        """Test creating a valid metrics configuration."""
        config = MetricsConfig(
            track_tokens=True,
            track_timing=True,
            track_cost=False,
        )
        assert config.track_tokens is True
        assert config.track_cost is False

    def test_metrics_config_defaults_to_enabled(self):
        """Test that metrics tracking defaults to enabled."""
        config = MetricsConfig()
        assert config.track_tokens is True
        assert config.track_timing is True
        assert config.track_cost is True


class TestSwarmPRConfig:
    """Tests for the top-level SwarmPRConfig model."""

    def test_full_config_from_dict(self, sample_config_dict: dict):
        """Test loading a full configuration from a dictionary."""
        config = SwarmPRConfig(**sample_config_dict)
        assert config.project.name == "test-project"
        assert config.providers["planner"].model == "anthropic/claude-sonnet-4-20250514"
        assert config.risk_tiers["tier_3"].action == "block"
        assert config.risk_tiers["tier_2"].action == "flag"
        assert config.risk_tiers["tier_1"].action == "approve"
        assert config.pipeline.max_diff_lines == 500
        assert config.metrics.track_tokens is True

    def test_config_requires_project(self):
        """Test that config requires a project section."""
        with pytest.raises(Exception):
            SwarmPRConfig(
                providers={"planner": ProviderConfig(model="test/model")},
                risk_tiers={},
            )

    def test_config_requires_providers(self, sample_config_dict: dict):
        """Test that config requires a providers section."""
        del sample_config_dict["providers"]
        with pytest.raises(Exception):
            SwarmPRConfig(**sample_config_dict)


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_config_from_yaml(
        self, sample_config_yaml: Path, sample_config_dict: dict
    ):
        """Test loading config from a YAML file."""
        config = load_config(sample_config_yaml)
        assert isinstance(config, SwarmPRConfig)
        assert config.project.name == "test-project"

    def test_load_config_file_not_found(self, tmp_path: Path):
        """Test that loading a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_load_config_invalid_yaml(self, tmp_path: Path):
        """Test that loading invalid YAML raises an appropriate error."""
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text(":::invalid yaml:::")
        with pytest.raises(Exception):
            load_config(bad_file)

    def test_load_example_config(self, sample_config_path: Path):
        """Test that the shipped config.example.yaml is valid."""
        if sample_config_path.exists():
            config = load_config(sample_config_path)
            assert isinstance(config, SwarmPRConfig)
