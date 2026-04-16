# Contributing to Agiwo

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
git clone https://github.com/xhwSkhizein/agiwo.git
cd agiwo
uv sync
```

### Console Frontend

```bash
cd console/web
npm install
```

## Development Workflow

### Running Tests

```bash
# SDK tests
uv run pytest tests/ -v

# Console backend tests
(cd console && uv run pytest tests/ -v)

# Console frontend checks
(cd console/web && npm run lint)
(cd console/web && npm test)
(cd console/web && npm run build)

# Release smoke check from built wheel
uv build
uv run python scripts/smoke_release_install.py dist/agiwo-0.1.0-py3-none-any.whl

# Run a specific test file
uv run pytest tests/agent/test_agent_tool.py -v
```

### Linting

After making code changes, run the low-noise linter:

```bash
# Only changed files
uv run python scripts/lint.py changed

# Specific files
uv run python scripts/lint.py files path/to/file.py

# Import contracts only
uv run python scripts/lint.py imports
```

The lint pipeline runs: **ruff** + **repo_guard.py** + **import-linter**.

### Code Style

- Python 3.11+ with full type annotations on public APIs
- Async logic stays explicit — don't hide `await` in hard-to-trace helpers
- Prefer `is not None` for sentinel checks; use truthy checks only when semantics are clear
- No dead compatibility layers — delete old paths unless explicitly requested
- Fix circular dependencies by restructuring dependency direction, not with local imports

## Project Structure

```
agiwo/          # SDK
├── agent/      # Agent runtime, execution, hooks, storage
├── llm/        # Model providers
├── tool/       # Tool system, builtin tools
├── scheduler/  # Orchestration layer
├── observability/  # Trace collection and storage
├── embedding/  # Embedding providers
├── memory/     # MEMORY indexing and search
├── skill/      # Skill discovery and loading
├── workspace/  # Workspace layout
├── config/     # SDK configuration
└── utils/      # Shared utilities

console/        # Control plane
├── server/     # FastAPI backend
└── web/        # Next.js frontend

tests/          # SDK tests (mirrors agiwo/ structure)
console/tests/  # Console tests
```

See [AGENTS.md](./AGENTS.md) for detailed architecture documentation.

## Submitting Changes

### Branch Naming

- `feat/` — New features
- `fix/` — Bug fixes
- `refactor/` — Code restructuring
- `docs/` — Documentation
- `chore/` — Maintenance (deps, CI, tooling)
- `test/` — Test additions/improvements

### Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear, focused commits
3. Run `uv run python scripts/lint.py changed` before committing
4. Ensure tests pass: `uv run pytest tests/ -v` and `(cd console && uv run pytest tests/ -v)`
5. If you changed release-facing behavior, also run `(cd console/web && npm run lint && npm test && npm run build)` and `uv run python scripts/smoke_release_install.py dist/agiwo-0.1.0-py3-none-any.whl` after `uv build`
6. Update documentation if you changed public APIs
7. Open a PR with a clear description of what and why

### Commit Messages

Use conventional commit format:

```
feat: add streaming support to scheduler
fix: handle abort signal in tool execution
docs: update getting started guide
refactor: extract state ops from scheduler engine
```

## Adding New Components

### Adding a LLM Provider

1. Implement the provider class in `agiwo/llm/`
2. Register the provider enum in `agiwo/config/settings.py`
3. Add a `ProviderSpec` in `agiwo/llm/factory.py`
4. Export from `agiwo/llm/__init__.py` if public
5. Add tests in `tests/llm/`

### Adding a Builtin Tool

1. Create implementation in `agiwo/tool/builtin/`
2. Register with `@builtin_tool(...)` decorator
3. Add `@default_enable` only if it should be auto-enabled
4. If Console needs custom display logic, update `console/server/tools.py`

### Adding a Hook

1. Add hook type and field in `agiwo/agent/hooks.py`
2. Wire it into the agent execution flow in `agiwo/agent/agent.py`, `agiwo/agent/run_loop.py`, or the relevant runtime module

## Reporting Issues

- Use GitHub Issues for bugs and feature requests
- Include reproduction steps for bugs
- Include your Python version, OS, and relevant package versions

## Code of Conduct

Be respectful, constructive, and inclusive. We're all here to build something good.
