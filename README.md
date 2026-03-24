# SwarmPR

[![CI](https://github.com/jamjahal/swarmpr/actions/workflows/ci.yml/badge.svg)](https://github.com/jamjahal/swarmpr/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Pydantic v2](https://img.shields.io/badge/pydantic-v2-e92063.svg)](https://docs.pydantic.dev/)

**Multi-agent pipeline that turns task descriptions into tested, risk-reviewed pull requests.**

SwarmPR orchestrates four specialized AI agents — Planner, Coder, Tester, and Reviewer — through a custom pipeline engine with event-driven observability, blast-radius risk classification, and fintech-grade escalation policies. No frameworks like LangGraph or CrewAI; just clean, auditable orchestration built from first principles.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     SwarmPR Pipeline                     │
│                                                         │
│  Task Description                                       │
│       │                                                 │
│       ▼                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  │ Planner  │──▶│  Coder   │──▶│  Tester  │──▶│ Reviewer │
│  │  Agent   │   │  Agent   │   │  Agent   │   │  Agent   │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘
│       │              │              │              │     │
│  Task Plan +    Generated      Test Results    Review +  │
│  Risk Tier      Files + Diff   (pytest)        Verdict   │
│                                                         │
│  ─────────── Event Bus (pub-sub) ──────────────────     │
│  ─────────── Metrics Collector ─────────────────────    │
│                                                         │
│       ▼                                                 │
│  ┌──────────────────────┐                               │
│  │  Risk Classification │  Tier 1: approve (auto-merge) │
│  │  & Escalation Policy │  Tier 2: flag (AI review)     │
│  │                      │  Tier 3: block (human review) │
│  └──────────────────────┘                               │
│       │                                                 │
│       ▼                                                 │
│  GitHub PR (with labels, risk summary, metrics)         │
└─────────────────────────────────────────────────────────┘
```

### Key Design Decisions

- **Custom orchestration engine** (~200 LOC) — no LangGraph, no CrewAI. Sequential agent execution with state passing, error recovery, and full lifecycle event emission. Easy to audit, easy to extend.
- **Model agnostic via LiteLLM** — swap between OpenAI, Anthropic, local Ollama models, or any LiteLLM-supported provider with a config change. No vendor lock-in.
- **Blast-radius risk classification** — path-prefix matching + keyword escalation. Highest tier wins. Unknown paths default to tier 2 (conservative).
- **Escalation policy engine** — tier-based action mapping with diff-size and forbidden-path overrides. Payment logic always requires human review.
- **Event-driven observability** — pub-sub EventBus emits typed events at every lifecycle point. Terminal handler for CLI, WebSocket broadcast for API clients.
- **Pydantic v2 throughout** — all state, config, metrics, and API models are validated Pydantic models.

---

## Project Structure

```
swarmpr/
├── agents/                 # AI agent implementations
│   ├── base.py             # BaseAgent ABC + LLM call helper
│   ├── planner.py          # Task decomposition → structured plan
│   ├── coder.py            # Plan → code generation + forbidden path enforcement
│   ├── tester.py           # pytest execution with timeout handling
│   └── reviewer.py         # Code review + risk scoring + escalation
├── orchestrator/           # Pipeline execution engine
│   ├── engine.py           # PipelineEngine — sequential agent runner
│   ├── events.py           # EventBus pub-sub system
│   └── state.py            # PipelineState + all data models
├── providers/              # LLM provider abstraction
│   ├── base.py             # AgentProvider protocol + Message model
│   └── litellm_provider.py # LiteLLM wrapper implementation
├── risk/                   # Risk classification & escalation
│   ├── classifier.py       # Path + keyword → tier assignment
│   └── policies.py         # Tier → action mapping with overrides
├── github/                 # Git & GitHub integration
│   ├── repo.py             # GitPython wrapper for local operations
│   └── pr.py               # PyGithub PR creation with risk-aware labels
├── metrics/                # Observability & cost tracking
│   └── collector.py        # Token, timing, cost aggregation
├── server/                 # API layer
│   └── app.py              # FastAPI + WebSocket for real-time streaming
├── cli.py                  # Typer CLI entry point
└── config.py               # YAML config loader + Pydantic models

demo/
├── sample_repo/            # Fintech codebase exercising all 3 risk tiers
│   ├── payments/           # Tier 3 — PCI-sensitive payment processing
│   ├── auth/               # Tier 3 — token management
│   ├── api/                # Tier 2 — API endpoints
│   └── config/             # Tier 1 — app settings
└── sample_tasks/           # Pre-built task descriptions per tier

tests/                      # 100+ tests, TDD from day one
├── agents/
├── orchestrator/
├── providers/
├── github/
└── ...
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- conda (recommended) or pip
- At least one LLM API key (OpenAI, Anthropic, etc.) — or a local model via Ollama

### Installation

```bash
# Clone the repo
git clone https://github.com/jamjahal/swarmpr.git
cd swarmpr

# Option A: conda (recommended)
conda env create -f environment.yml
conda activate swarmpr

# Option B: pip
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Configuration

```bash
# Generate a starter config
swarmpr config --output config.yaml
```

Edit `config.yaml` to add your model providers:

```yaml
providers:
  planner:
    model: "anthropic/claude-sonnet-4-20250514"
  coder:
    model: "anthropic/claude-sonnet-4-20250514"
  tester:
    model: "anthropic/claude-haiku-4-5-20251001"
  reviewer:
    model: "openai/gpt-4o"  # Cross-model review for independence
```

For local models (e.g., Ollama on a miniPC):

```yaml
providers:
  planner:
    model: "ollama/llama3"
    api_base: "http://192.168.1.100:11434"
```

### Run the Pipeline

```bash
# Run against any repo
swarmpr run --repo /path/to/repo --task "Add input validation to the signup endpoint"

# Create a GitHub PR automatically
swarmpr run \
  --repo /path/to/repo \
  --task "Add rate limiting to the payments API" \
  --create-pr \
  --github-token $GITHUB_TOKEN \
  --github-repo owner/repo
```

### Run the Demo

The demo runs against a sample fintech repo with pre-built tasks for each risk tier:

```bash
# Tier 1 — Config change (auto-approve)
swarmpr demo --tier 1

# Tier 2 — API endpoint (AI review + flag)
swarmpr demo --tier 2

# Tier 3 — Payment logic (human review required, PR as draft)
swarmpr demo --tier 3
```

### Start the API Server

```bash
swarmpr serve --port 8000
```

Endpoints:
- `GET /health` — service health check
- `GET /history` — past pipeline run summaries
- `WS /ws` — real-time event stream via WebSocket

---

## Risk Classification

SwarmPR assigns a **blast-radius tier** to every task based on which files are affected:

| Tier | Scope | Action | Example |
|------|-------|--------|---------|
| **Tier 1** | Config, docs, tests | `approve` — auto-merge ready | Change `LOG_LEVEL` in settings |
| **Tier 2** | APIs, services, models | `flag` — AI review, human notified | Add `/payments/summary` endpoint |
| **Tier 3** | Payments, auth, compliance | `block` — draft PR, human review required | Add `refund_payment()` method |

Classification uses path-prefix matching with keyword escalation (e.g., `api_key`, `encrypt`, `pci`). The highest tier always wins. Unknown paths default to tier 2.

The **escalation policy** layer adds overrides: diffs exceeding `max_diff_lines` escalate to flag; any change touching `forbidden_paths` (`.env`, `secrets/`, `credentials/`) is blocked unconditionally.

---

## Development

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=swarmpr --cov-report=term-missing

# Skip integration tests (no API keys needed)
pytest -m "not integration"
```

### Linting

```bash
ruff check swarmpr/ tests/
ruff format swarmpr/ tests/
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

---

## How It Works

1. **Planner Agent** receives the task description and target repo context. It produces a structured plan: files to modify, estimated complexity, and an initial risk tier based on which paths are touched.

2. **Risk Classifier** evaluates the plan against configured path prefixes and keywords. The highest matching tier is assigned. Payment paths, auth modules, and compliance code trigger tier 3.

3. **Coder Agent** implements the plan file-by-file, enforcing forbidden-path rules at generation time. Produces a diff and stores generated files on pipeline state.

4. **Tester Agent** runs `pytest` via subprocess with configurable timeout. Results (pass/fail/timeout) are captured on state. Failing tests block auto-approval regardless of tier.

5. **Reviewer Agent** performs an LLM-based code review, then the Escalation Policy maps the tier + overrides to a final action: approve, flag, or block.

6. **PR Creator** builds a GitHub PR with risk-aware labels (`risk:tier-1`, `risk:tier-3`), a structured body (changes, risk summary, test results, metrics), and marks tier-3 PRs as drafts.

Throughout, the **Event Bus** emits typed events (`PIPELINE_STARTED`, `AGENT_COMPLETED`, `PIPELINE_FAILED`, etc.) and the **Metrics Collector** tracks timing, token usage, and cost estimates per agent.

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| LLM Integration | LiteLLM | Model-agnostic; 100+ providers including local models |
| Orchestration | Custom engine | Auditable, no framework lock-in, ~200 LOC |
| Data Models | Pydantic v2 | Runtime validation, serialization, OpenAPI schemas |
| CLI | Typer + Rich | Professional terminal UX with minimal code |
| API | FastAPI + WebSocket | Async, auto-docs, real-time streaming |
| Git Operations | GitPython | Programmatic branch/commit/diff |
| PR Creation | PyGithub | GitHub API with risk-aware labels |
| Testing | pytest + pytest-asyncio | TDD from day one, async support |
| Linting | Ruff | Fast, replaces flake8/isort/pyupgrade |
| CI/CD | GitHub Actions | Lint, test (3.10-3.12), type check, security scan |

---

## Roadmap

- [ ] React dashboard frontend (WebSocket integration ready)
- [ ] Parallel agent execution for independent stages
- [ ] Persistent metrics storage (SQLite/Postgres)
- [ ] GitHub webhook integration for auto-triggered pipelines
- [ ] Plugin system for custom agents
- [ ] Multi-repo support

---

## License

MIT
