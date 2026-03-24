"""Tests for the SwarmPR CLI."""

from pathlib import Path

from typer.testing import CliRunner

from swarmpr.cli import app

runner = CliRunner()


class TestCLI:
    """Tests for CLI commands."""

    def test_config_init_creates_file(self, tmp_path: Path):
        """Test that config init creates a config file."""
        output = tmp_path / "config.yaml"
        result = runner.invoke(
            app, ["config", "--output", str(output)]
        )
        assert result.exit_code == 0
        assert output.exists()

    def test_config_init_default_contains_yaml(self, tmp_path: Path):
        """Test that generated config contains valid YAML content."""
        output = tmp_path / "config.yaml"
        runner.invoke(
            app, ["config", "--output", str(output)]
        )
        content = output.read_text()
        assert "project" in content or "providers" in content

    def test_run_requires_task(self):
        """Test that run command requires a task argument."""
        result = runner.invoke(app, ["run"])
        assert result.exit_code != 0

    def test_demo_shows_help(self):
        """Test that demo --help works."""
        result = runner.invoke(app, ["demo", "--help"])
        assert result.exit_code == 0
        assert "demo" in result.output.lower()

    def test_serve_shows_help(self):
        """Test that serve --help works."""
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "port" in result.output.lower()
