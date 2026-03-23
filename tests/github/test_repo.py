"""Tests for local git repository operations."""

import subprocess
from pathlib import Path

import pytest

from swarmpr.github.repo import RepoManager


def _run_git(repo_path: Path, *args: str) -> None:
    """Run a git command in the given repo path.

    Uses subprocess.run with a fixed argument list (no shell) for safety.

    Args:
        repo_path: The repository directory.
        *args: Git subcommand and arguments.
    """
    subprocess.run(
        ["git", *args],
        cwd=str(repo_path),
        capture_output=True,
        check=True,
    )


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository for testing.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Path to the initialized git repository.
    """
    repo_path = tmp_path / "test-repo"
    repo_path.mkdir()

    _run_git(repo_path, "init", "-b", "main")
    _run_git(repo_path, "config", "user.email", "test@test.com")
    _run_git(repo_path, "config", "user.name", "Test")

    (repo_path / "README.md").write_text("# Test Repo\n")
    _run_git(repo_path, "add", ".")
    _run_git(repo_path, "commit", "-m", "init")

    return repo_path


class TestRepoManager:
    """Tests for the RepoManager class."""

    def test_init_with_path(self, temp_git_repo: Path):
        """Test initializing RepoManager with a repo path."""
        manager = RepoManager(str(temp_git_repo))
        assert manager.repo_path == str(temp_git_repo)

    def test_create_branch(self, temp_git_repo: Path):
        """Test creating a new branch."""
        manager = RepoManager(str(temp_git_repo))
        manager.create_branch("swarmpr/test-branch")
        assert manager.current_branch() == "swarmpr/test-branch"

    def test_create_branch_from_main(self, temp_git_repo: Path):
        """Test that new branch is created from main."""
        manager = RepoManager(str(temp_git_repo))

        (temp_git_repo / "main_file.txt").write_text("on main\n")
        _run_git(temp_git_repo, "add", ".")
        _run_git(temp_git_repo, "commit", "-m", "main commit")

        manager.create_branch("swarmpr/feature")
        assert (temp_git_repo / "main_file.txt").exists()

    def test_write_file(self, temp_git_repo: Path):
        """Test writing a file to the repo."""
        manager = RepoManager(str(temp_git_repo))
        manager.create_branch("swarmpr/test")
        manager.write_file("new_file.py", "print('hello')\n")

        assert (temp_git_repo / "new_file.py").exists()
        assert (
            (temp_git_repo / "new_file.py").read_text()
            == "print('hello')\n"
        )

    def test_write_file_creates_directories(self, temp_git_repo: Path):
        """Test that write_file creates parent directories."""
        manager = RepoManager(str(temp_git_repo))
        manager.create_branch("swarmpr/test")
        manager.write_file(
            "payments/validator.py", "def validate(): pass\n"
        )
        assert (temp_git_repo / "payments" / "validator.py").exists()

    def test_commit_changes(self, temp_git_repo: Path):
        """Test committing staged changes."""
        manager = RepoManager(str(temp_git_repo))
        manager.create_branch("swarmpr/test")
        manager.write_file("new_file.py", "print('hello')\n")
        manager.commit("Add new file")

        log = manager.get_log(n=1)
        assert "Add new file" in log

    def test_get_diff(self, temp_git_repo: Path):
        """Test getting the diff between branch and main."""
        manager = RepoManager(str(temp_git_repo))
        manager.create_branch("swarmpr/test")
        manager.write_file("new_file.py", "print('hello')\n")
        manager.commit("Add file")

        diff = manager.get_diff_from_main()
        assert "new_file.py" in diff
        assert "print('hello')" in diff

    def test_get_changed_files(self, temp_git_repo: Path):
        """Test getting the list of changed files vs main."""
        manager = RepoManager(str(temp_git_repo))
        manager.create_branch("swarmpr/test")
        manager.write_file("file_a.py", "a\n")
        manager.write_file("file_b.py", "b\n")
        manager.commit("Add files")

        changed = manager.get_changed_files()
        assert "file_a.py" in changed
        assert "file_b.py" in changed

    def test_checkout_main(self, temp_git_repo: Path):
        """Test switching back to main branch."""
        manager = RepoManager(str(temp_git_repo))
        manager.create_branch("swarmpr/test")
        assert manager.current_branch() == "swarmpr/test"

        manager.checkout("main")
        assert manager.current_branch() == "main"

    def test_delete_file(self, temp_git_repo: Path):
        """Test deleting a file from the repo."""
        manager = RepoManager(str(temp_git_repo))
        manager.create_branch("swarmpr/test")
        manager.write_file("temp.py", "temp\n")
        manager.commit("Add temp")

        manager.delete_file("temp.py")
        assert not (temp_git_repo / "temp.py").exists()
