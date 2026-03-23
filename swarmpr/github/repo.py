"""Local git repository operations for SwarmPR.

Manages branch creation, file writing, committing, and diffing
using GitPython. All operations are local — remote push and PR
creation are handled by the pr module.
"""

from pathlib import Path

from git import Repo


class RepoManager:
    """Manages local git operations for a SwarmPR pipeline run.

    Handles branch creation, file manipulation, committing, and
    generating diffs against the main branch.

    Attributes:
        repo_path: Absolute path to the repository root.
        repo: The GitPython Repo instance.
    """

    def __init__(self, repo_path: str) -> None:
        """Initialize with a path to an existing git repository.

        Args:
            repo_path: Path to the repository root directory.
        """
        self.repo_path = repo_path
        self.repo = Repo(repo_path)

    def current_branch(self) -> str:
        """Return the name of the currently checked-out branch.

        Returns:
            The active branch name as a string.
        """
        return self.repo.active_branch.name

    def create_branch(self, branch_name: str) -> None:
        """Create and checkout a new branch from the current HEAD.

        Args:
            branch_name: Name for the new branch.
        """
        self.repo.git.checkout("-b", branch_name)

    def checkout(self, branch_name: str) -> None:
        """Checkout an existing branch.

        Args:
            branch_name: Name of the branch to switch to.
        """
        self.repo.git.checkout(branch_name)

    def write_file(self, relative_path: str, content: str) -> None:
        """Write content to a file in the repository.

        Creates parent directories if they don't exist.

        Args:
            relative_path: Path relative to the repo root.
            content: The file content to write.
        """
        full_path = Path(self.repo_path) / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)

    def delete_file(self, relative_path: str) -> None:
        """Delete a file from the repository.

        Args:
            relative_path: Path relative to the repo root.
        """
        full_path = Path(self.repo_path) / relative_path
        if full_path.exists():
            full_path.unlink()

    def commit(self, message: str) -> str:
        """Stage all changes and create a commit.

        Args:
            message: The commit message.

        Returns:
            The commit SHA as a string.
        """
        self.repo.git.add("-A")
        self.repo.git.commit("-m", message)
        return self.repo.head.commit.hexsha

    def get_diff_from_main(self, main_branch: str = "main") -> str:
        """Get the unified diff between the current branch and main.

        Args:
            main_branch: The name of the main/base branch.

        Returns:
            The unified diff as a string.
        """
        return self.repo.git.diff(main_branch, self.current_branch())

    def get_changed_files(self, main_branch: str = "main") -> list[str]:
        """Get the list of files changed vs main.

        Args:
            main_branch: The name of the main/base branch.

        Returns:
            A list of changed file paths relative to the repo root.
        """
        diff_output = self.repo.git.diff(
            main_branch, self.current_branch(), name_only=True
        )
        return [
            line.strip()
            for line in diff_output.splitlines()
            if line.strip()
        ]

    def get_log(self, n: int = 5) -> str:
        """Get recent commit log entries.

        Args:
            n: Number of log entries to return.

        Returns:
            The git log output as a string.
        """
        return self.repo.git.log(f"-{n}", "--oneline")
