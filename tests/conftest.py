"""Shared test fixtures for SwarmPR test suite."""

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_config_path(project_root: Path) -> Path:
    """Return the path to the example config file."""
    return project_root / "config.example.yaml"


@pytest.fixture
def sample_config_dict() -> dict:
    """Return a minimal valid configuration dictionary.

    Returns:
        A dictionary representing a valid SwarmPR configuration.
    """
    return {
        "project": {
            "name": "test-project",
            "repo": "https://github.com/test/repo",
        },
        "providers": {
            "planner": {"model": "anthropic/claude-sonnet-4-20250514"},
            "coder": {"model": "anthropic/claude-sonnet-4-20250514"},
            "tester": {"model": "anthropic/claude-haiku-4-5-20251001"},
            "reviewer": {"model": "openai/gpt-4o"},
        },
        "risk_tiers": {
            "tier_3": {
                "description": "Critical path",
                "paths": ["payments/", "auth/"],
                "keywords": ["api_key", "secret"],
                "action": "block",
            },
            "tier_2": {
                "description": "Business logic",
                "paths": ["api/", "services/"],
                "keywords": [],
                "action": "flag",
            },
            "tier_1": {
                "description": "Low risk",
                "paths": ["config/", "docs/", "tests/"],
                "keywords": [],
                "action": "approve",
            },
        },
        "pipeline": {
            "max_diff_lines": 500,
            "test_timeout_seconds": 120,
            "forbidden_paths": [".env", "secrets/"],
        },
        "metrics": {
            "track_tokens": True,
            "track_timing": True,
            "track_cost": True,
        },
    }


@pytest.fixture
def sample_config_yaml(tmp_path: Path, sample_config_dict: dict) -> Path:
    """Write a sample config to a temporary YAML file and return its path.

    Args:
        tmp_path: Pytest temporary directory fixture.
        sample_config_dict: The config dictionary to serialize.

    Returns:
        Path to the temporary YAML config file.
    """
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config_dict, f)
    return config_path
