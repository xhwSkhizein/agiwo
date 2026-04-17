# Python 3.10 Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `agiwo` and `agiwo-console` installable and validated on Python 3.10 by aligning code, packaging metadata, CI, release smoke checks, and live documentation to a 3.10+ minimum version.

**Architecture:** Keep the compatibility change narrow. Patch only confirmed 3.11-only runtime and tooling APIs, then lower the declared Python floor and move the automated validation path to Python 3.10 so the published support claim is continuously enforced.

**Tech Stack:** Python 3.10+, Hatchling, uv, Ruff, pytest, GitHub Actions

---

## File Structure

- Modify: `pyproject.toml`
  - Lower the root package Python floor to 3.10 and update Ruff's target version.
- Modify: `console/pyproject.toml`
  - Lower the console package Python floor to 3.10.
- Modify: `console/uv.lock`
  - Refresh the lock metadata so local/CI sync does not keep requiring Python 3.11.
- Modify: `agiwo/agent/llm_caller.py`
  - Replace the `asyncio.timeout` usage with a Python 3.10-safe timeout pattern.
- Modify: `scripts/check_release_metadata.py`
  - Add a Python 3.10 TOML parsing path that still uses stdlib `tomllib` when available.
- Modify: `tests/scripts/test_check_release_metadata.py`
  - Add or update coverage for the TOML loader path used by release metadata validation.
- Modify: `.github/workflows/ci.yml`
  - Run lint/tests/package smoke validation on Python 3.10.
- Modify: `.github/workflows/release.yml`
  - Build and smoke test release artifacts on Python 3.10 before publishing.
- Modify: `.github/workflows/public-docs.yml`
  - Keep the workflow Python version aligned with the new baseline where appropriate.
- Modify: `scripts/repo_guard.py`
  - Update guard text that still claims Python 3.11+ as the project standard.
- Modify: `README.md`
  - Update the Python badge and install/runtime wording to 3.10+.
- Modify: `docs/getting-started.md`
  - Update the documented Python requirement.
- Modify: `docs/release.md`
  - Update release instructions so maintainer verification reflects Python 3.10.
- Modify: `AGENTS.md`
  - Update the architecture boundary and standard command sections from 3.11+ to 3.10+.

### Task 1: Patch Confirmed Python 3.11-Only Code Paths

**Files:**
- Modify: `agiwo/agent/llm_caller.py`
- Modify: `scripts/check_release_metadata.py`
- Modify: `tests/scripts/test_check_release_metadata.py`
- Test: `tests/scripts/test_check_release_metadata.py`

- [ ] **Step 1: Add failing coverage for the release metadata TOML loader path**

```python
from pathlib import Path

from scripts.check_release_metadata import load_project


def test_load_project_reads_pyproject_without_runtime_tomllib(monkeypatch, tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "agiwo"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )

    monkeypatch.setattr("scripts.check_release_metadata.tomllib", None)

    project = load_project(pyproject)

    assert project["name"] == "agiwo"
    assert project["version"] == "0.1.0"
```

- [ ] **Step 2: Run the targeted script tests to confirm the fallback path is currently missing**

Run: `uv run pytest tests/scripts/test_check_release_metadata.py -v`

Expected: FAIL with an `AttributeError`, `TypeError`, or import-path failure because `load_project()` cannot parse TOML once `tomllib` is unavailable.

- [ ] **Step 3: Make `scripts/check_release_metadata.py` support Python 3.10**

```python
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None

try:
    import tomli
except ModuleNotFoundError:
    tomli = None


def _load_toml_bytes(data: bytes) -> dict[str, object]:
    if tomllib is not None:
        return tomllib.loads(data.decode("utf-8"))
    if tomli is not None:
        return tomli.loads(data.decode("utf-8"))
    raise SystemExit("Python 3.10 requires the tomli package for release metadata checks")


def load_project(pyproject_path: Path) -> dict[str, object]:
    data = _load_toml_bytes(pyproject_path.read_bytes())
    project = data.get("project")
    if not isinstance(project, dict):
        raise SystemExit(f"Missing [project] table in {pyproject_path}")
    return project
```

- [ ] **Step 4: Replace `asyncio.timeout` in the LLM streaming path with a 3.10-safe timeout**

```python
while True:
    try:
        chunk = await asyncio.wait_for(stream.__anext__(), timeout=_CHUNK_TIMEOUT_SECONDS)
    except StopAsyncIteration:
        break
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"LLM stream stalled: no chunk received for {_CHUNK_TIMEOUT_SECONDS}s"
        ) from exc
```

- [ ] **Step 5: Run the focused compatibility tests**

Run: `uv run pytest tests/scripts/test_check_release_metadata.py -v`

Expected: PASS

- [ ] **Step 6: Run the closest agent/runtime test slice that covers LLM calling**

Run: `uv run pytest tests/agent/ -v --tb=short`

Expected: PASS

- [ ] **Step 7: Commit the compatibility code fixes**

```bash
git add agiwo/agent/llm_caller.py scripts/check_release_metadata.py tests/scripts/test_check_release_metadata.py
git commit -m "fix: support python 3.10 compatibility paths"
```

### Task 2: Lower The Python Floor In Packaging And Automation

**Files:**
- Modify: `pyproject.toml`
- Modify: `console/pyproject.toml`
- Modify: `console/uv.lock`
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/release.yml`
- Modify: `.github/workflows/public-docs.yml`
- Modify: `scripts/repo_guard.py`

- [ ] **Step 1: Update root package metadata to a 3.10 baseline**

```toml
[project]
requires-python = ">=3.10"

[tool.ruff]
target-version = "py310"
```

- [ ] **Step 2: Update the console package metadata to a 3.10 baseline**

```toml
[project]
requires-python = ">=3.10"
```

- [ ] **Step 3: Refresh the committed console lockfile metadata**

Run: `cd console && uv lock`

Expected: `console/uv.lock` updates its `requires-python` header to `>=3.10` and resolves any package constraints against Python 3.10.

- [ ] **Step 4: Switch the backend CI jobs from Python 3.11 to 3.10**

```yaml
- name: Set up Python
  run: uv python install 3.10
```

- [ ] **Step 5: Switch the release workflow build jobs from Python 3.11 to 3.10**

```yaml
- name: Set up Python
  run: uv python install 3.10
```

- [ ] **Step 6: Update any repository guard text that still claims Python 3.11+**

```python
"the project standard is Python 3.10+ native annotations."
```

- [ ] **Step 7: Run the repository guard and backend lint path**

Run: `uv run python scripts/repo_guard.py && uv run ruff check --ignore C901 --ignore PLR0911 --ignore PLR0912 agiwo/ console/server/ tests/ console/tests/ scripts/`

Expected: PASS

- [ ] **Step 8: Commit the metadata and workflow baseline change**

```bash
git add pyproject.toml console/pyproject.toml console/uv.lock .github/workflows/ci.yml .github/workflows/release.yml .github/workflows/public-docs.yml scripts/repo_guard.py
git commit -m "build: validate python 3.10 as the minimum version"
```

### Task 3: Update Live Docs And Prove Python 3.10 End-To-End

**Files:**
- Modify: `README.md`
- Modify: `docs/getting-started.md`
- Modify: `docs/release.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Update live documentation references from Python 3.11+ to 3.10+**

```markdown
<img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
```

```markdown
- Python 3.10+
```

```markdown
- Python 版本基线是 3.10+。
```

- [ ] **Step 2: Run the changed-file lint shortcut**

Run: `uv run python scripts/lint.py changed`

Expected: PASS

- [ ] **Step 3: Run the SDK test suite on Python 3.10**

Run: `uv run pytest tests/ -v --tb=short`

Expected: PASS

- [ ] **Step 4: Run the Console backend test suite on Python 3.10**

Run: `cd console && uv run pytest tests/ -v --tb=short`

Expected: PASS

- [ ] **Step 5: Build and smoke-test the SDK wheel on Python 3.10**

Run: `uv build && uv run python scripts/smoke_release_install.py "$(ls dist/agiwo-*.whl | head -n 1)"`

Expected: PASS

- [ ] **Step 6: Build and smoke-test the Console wheel pairing on Python 3.10**

Run: `(cd console && uv build) && uv run python scripts/smoke_release_install.py "$(ls dist/agiwo-*.whl | head -n 1)" "$(ls console/dist/agiwo_console-*.whl | head -n 1)"`

Expected: PASS

- [ ] **Step 7: Commit the docs and final validation results**

```bash
git add README.md docs/getting-started.md docs/release.md AGENTS.md
git commit -m "docs: publish python 3.10 support baseline"
```

## Self-Review

- Spec coverage: Task 1 covers the confirmed 3.11-only code paths, Task 2 covers package metadata plus CI/release enforcement, and Task 3 covers live docs plus the acceptance validation path from the design.
- Placeholder scan: No task contains `TODO`, `TBD`, or undefined "add tests" language without concrete commands or snippets.
- Type consistency: The plan uses the existing file paths and command surfaces already present in the repository, and the Python-version floor is consistently described as 3.10+ across all tasks.
