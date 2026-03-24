"""CLI entry point for SwarmPR.

Provides commands for running the pipeline, launching the demo,
initializing config, and starting the API server.
"""

import asyncio
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from swarmpr.agents.coder import CoderAgent
from swarmpr.agents.planner import PlannerAgent
from swarmpr.agents.reviewer import ReviewerAgent
from swarmpr.agents.tester import TesterAgent
from swarmpr.config import load_config
from swarmpr.github.pr import PRCreator
from swarmpr.github.repo import RepoManager
from swarmpr.metrics.collector import MetricsCollector
from swarmpr.orchestrator.engine import PipelineEngine
from swarmpr.orchestrator.events import Event, EventBus, EventType
from swarmpr.orchestrator.state import PipelineStage, PipelineState
from swarmpr.providers.litellm_provider import LiteLLMProvider
from swarmpr.risk.classifier import RiskClassifier
from swarmpr.risk.policies import EscalationPolicy

app = typer.Typer(
    name="swarmpr",
    help="Multi-agent pipeline: task description → tested, risk-reviewed PR",
    add_completion=False,
)
console = Console()

# Stage-to-emoji mapping for terminal output.
_STAGE_ICONS = {
    PipelineStage.PLANNER: "📋",
    PipelineStage.CODER: "💻",
    PipelineStage.TESTER: "🧪",
    PipelineStage.REVIEWER: "🔍",
}


async def _terminal_handler(event: Event) -> None:
    """Print pipeline events to the terminal with rich formatting.

    Args:
        event: The event to display.
    """
    icon = ""
    if event.stage:
        icon = _STAGE_ICONS.get(event.stage, "")

    if event.event_type == EventType.PIPELINE_STARTED:
        console.print(
            Panel(
                f"[bold green]Pipeline Started[/]\n{event.message}",
                title="SwarmPR",
                border_style="green",
            )
        )
    elif event.event_type == EventType.AGENT_STARTED:
        console.print(
            f"  {icon} [bold cyan]{event.stage.value}[/] starting..."
        )
    elif event.event_type == EventType.AGENT_COMPLETED:
        duration = event.data.get("duration_ms", 0)
        tokens = event.data.get("tokens_in", 0) + event.data.get(
            "tokens_out", 0
        )
        console.print(
            f"  {icon} [bold green]{event.stage.value}[/] "
            f"completed ({duration}ms, {tokens} tokens)"
        )
    elif event.event_type == EventType.AGENT_FAILED:
        console.print(
            f"  {icon} [bold red]{event.stage.value}[/] "
            f"FAILED: {event.data.get('error', 'unknown')}"
        )
    elif event.event_type == EventType.PIPELINE_COMPLETED:
        console.print(
            Panel(
                "[bold green]Pipeline Completed[/]",
                border_style="green",
            )
        )
    elif event.event_type == EventType.PIPELINE_FAILED:
        console.print(
            Panel(
                f"[bold red]Pipeline Failed[/]\n{event.message}",
                border_style="red",
            )
        )


def _build_agents(config, classifier, policy):
    """Build the agent pipeline from config.

    Args:
        config: The SwarmPR configuration.
        classifier: The risk classifier instance.
        policy: The escalation policy instance.

    Returns:
        A list of agents in pipeline order.
    """
    planner_provider = LiteLLMProvider(config.providers["planner"])
    coder_provider = LiteLLMProvider(config.providers["coder"])
    tester_provider = LiteLLMProvider(config.providers["tester"])
    reviewer_provider = LiteLLMProvider(config.providers["reviewer"])

    return [
        PlannerAgent(
            provider=planner_provider, classifier=classifier
        ),
        CoderAgent(
            provider=coder_provider,
            pipeline_config=config.pipeline,
        ),
        TesterAgent(
            provider=tester_provider,
            pipeline_config=config.pipeline,
        ),
        ReviewerAgent(
            provider=reviewer_provider,
            classifier=classifier,
            policy=policy,
        ),
    ]


async def _run_pipeline(
    config_path: str,
    task: str,
    repo_path: str,
    create_pr: bool = False,
    github_token: str | None = None,
    repo_full_name: str | None = None,
) -> PipelineState:
    """Execute the full SwarmPR pipeline.

    Args:
        config_path: Path to the SwarmPR YAML config.
        task: The task description.
        repo_path: Path to the target repository.
        create_pr: Whether to create a GitHub PR.
        github_token: GitHub token for PR creation.
        repo_full_name: Full repo name (owner/repo) for PR creation.

    Returns:
        The final pipeline state.
    """
    config = load_config(Path(config_path))

    classifier = RiskClassifier(config.risk_tiers)
    policy = EscalationPolicy(config.risk_tiers, config.pipeline)

    event_bus = EventBus()
    event_bus.subscribe(_terminal_handler)

    engine = PipelineEngine(event_bus=event_bus)
    agents = _build_agents(config, classifier, policy)

    state = PipelineState(
        task_description=task,
        repo_path=repo_path,
    )

    state = await engine.run(state, agents=agents)

    # Apply generated files to repo if coder produced output.
    if state.generated_files and state.branch_name:
        try:
            repo_mgr = RepoManager(repo_path)
            repo_mgr.create_branch(state.branch_name)

            for file_path, content in state.generated_files.items():
                repo_mgr.write_file(file_path, content)

            repo_mgr.commit(
                f"[SwarmPR] {state.task_description[:50]}"
            )

            state.diff = repo_mgr.get_diff_from_main()
        except Exception as exc:
            console.print(
                f"[yellow]Warning: Could not apply changes "
                f"to repo: {exc}[/]"
            )

    # Create PR if requested.
    if create_pr and github_token and repo_full_name:
        try:
            pr_creator = PRCreator(token=github_token)
            pr_url = pr_creator.create_pr(
                repo_full_name=repo_full_name,
                state=state,
            )
            state.pr_url = pr_url
            console.print(f"\n[bold green]PR created:[/] {pr_url}")
        except Exception as exc:
            console.print(
                f"[bold red]Failed to create PR:[/] {exc}"
            )

    # Print metrics summary.
    collector = MetricsCollector()
    metrics = collector.collect(state)
    console.print()
    console.print(collector.format_summary(metrics))

    # Print review verdict.
    if state.review:
        verdict_color = {
            "approve": "green",
            "flag": "yellow",
            "block": "red",
        }.get(state.review.escalation_action, "white")

        console.print()
        console.print(
            Panel(
                f"[bold {verdict_color}]"
                f"Verdict: {state.review.escalation_action.upper()}"
                f"[/]\n\n"
                f"Risk Score: {state.review.risk_score}/10\n"
                f"{state.review.summary}",
                title="Review Result",
                border_style=verdict_color,
            )
        )

    return state


@app.command()
def run(
    repo: str = typer.Option(
        ".",
        "--repo",
        "-r",
        help="Path to the target repository",
    ),
    task: str = typer.Option(
        ...,
        "--task",
        "-t",
        help="Task description (what to implement)",
    ),
    config: str = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to SwarmPR config file",
    ),
    create_pr: bool = typer.Option(
        False,
        "--create-pr",
        help="Create a GitHub PR after pipeline completes",
    ),
    github_token: str = typer.Option(
        None,
        "--github-token",
        envvar="GITHUB_TOKEN",
        help="GitHub token for PR creation",
    ),
    github_repo: str = typer.Option(
        None,
        "--github-repo",
        help="GitHub repo (owner/repo) for PR creation",
    ),
) -> None:
    """Run the SwarmPR pipeline on a task."""
    console.print(
        f"[bold]SwarmPR[/] — Running pipeline for: {task}\n"
    )

    asyncio.run(
        _run_pipeline(
            config_path=config,
            task=task,
            repo_path=repo,
            create_pr=create_pr,
            github_token=github_token,
            repo_full_name=github_repo,
        )
    )


@app.command()
def demo(
    tier: int = typer.Option(
        None,
        "--tier",
        "-t",
        help="Run a specific tier demo (1, 2, or 3)",
    ),
) -> None:
    """Run the demo against the sample fintech repo."""
    demo_dir = Path(__file__).parent.parent / "demo"
    sample_repo = demo_dir / "sample_repo"
    tasks_dir = demo_dir / "sample_tasks"
    config_path = (
        Path(__file__).parent.parent / "config.example.yaml"
    )

    if not sample_repo.exists():
        console.print(
            "[red]Demo sample repo not found. "
            "Run from the SwarmPR project root.[/]"
        )
        raise typer.Exit(1)

    # Select task based on tier.
    task_files = {
        1: "tier1_config_change.yaml",
        2: "tier2_api_endpoint.yaml",
        3: "tier3_payment_logic.yaml",
    }

    if tier and tier in task_files:
        task_file = tasks_dir / task_files[tier]
    else:
        # Default to tier 2 for a balanced demo.
        task_file = tasks_dir / task_files[2]
        tier = 2

    if task_file.exists():
        import yaml

        with open(task_file) as f:
            task_data = yaml.safe_load(f)
        task_desc = task_data.get(
            "description", "Demo task"
        )
    else:
        task_desc = f"Demo tier {tier} task"

    console.print(
        Panel(
            f"[bold]Demo Mode — Tier {tier}[/]\n\n"
            f"Task: {task_desc}\n"
            f"Repo: {sample_repo}",
            title="SwarmPR Demo",
            border_style="blue",
        )
    )

    asyncio.run(
        _run_pipeline(
            config_path=str(config_path),
            task=task_desc,
            repo_path=str(sample_repo),
        )
    )


@app.command("config")
def config_init(
    output: str = typer.Option(
        "config.yaml",
        "--output",
        "-o",
        help="Output path for the config file",
    ),
) -> None:
    """Generate a starter SwarmPR config file."""
    example = Path(__file__).parent.parent / "config.example.yaml"
    dest = Path(output)

    if dest.exists():
        overwrite = typer.confirm(
            f"{dest} already exists. Overwrite?"
        )
        if not overwrite:
            raise typer.Exit(0)

    if example.exists():
        shutil.copy(example, dest)
    else:
        # Inline fallback if example not found.
        dest.write_text(
            "# SwarmPR Configuration\n"
            "# See config.example.yaml for full options\n"
            "project:\n"
            '  name: "my-project"\n'
            '  repo: "."\n'
            "\nproviders:\n"
            "  planner:\n"
            '    model: "anthropic/claude-sonnet-4-20250514"\n'
            "  coder:\n"
            '    model: "anthropic/claude-sonnet-4-20250514"\n'
            "  tester:\n"
            '    model: "anthropic/claude-haiku-4-5-20251001"\n'
            "  reviewer:\n"
            '    model: "openai/gpt-4o"\n'
        )

    console.print(
        f"[green]Config written to {dest}[/]\n"
        f"Edit the file to configure your models and risk tiers."
    )


@app.command()
def serve(
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Port for the API server",
    ),
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        help="Host to bind the server to",
    ),
) -> None:
    """Start the SwarmPR API server."""
    import uvicorn

    console.print(
        f"[bold]SwarmPR API Server[/] starting on {host}:{port}"
    )
    uvicorn.run(
        "swarmpr.server.app:app",
        host=host,
        port=port,
        reload=True,
    )


if __name__ == "__main__":
    app()
