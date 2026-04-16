# GitHub Release Publish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish both `agiwo` and `agiwo-console` from a GitHub Release so a maintainer can create `vX.Y.Z` in GitHub and let Actions validate, build, smoke-test, and upload both packages to PyPI.

**Architecture:** Add a dedicated `release.yml` workflow triggered only by `release.published`, with a prepare step for tag normalization, separate build and publish jobs per package, and Trusted Publisher uploads isolated to the publish jobs. Keep release metadata validation explicit with a tiny checked-in Python script so CI can fail early on version drift and maintainers can run the same check locally.

**Tech Stack:** GitHub Actions, Python 3.11, uv, Hatchling, PyPI Trusted Publisher, tomllib

---

## File Structure

- Create: `.github/workflows/release.yml`
  - Dedicated release workflow that triggers on `release.published`, normalizes `vX.Y.Z`, builds both packages, runs smoke checks, uploads artifacts, and publishes to PyPI through Trusted Publisher.
- Create: `scripts/check_release_metadata.py`
  - Small Python CLI that validates the root package version, console package version, and the console's `agiwo` dependency release line against the requested release version.
- Create: `tests/scripts/test_check_release_metadata.py`
  - Unit tests for the release metadata validator so version/tag drift checks are covered outside of workflow YAML.
- Create: `docs/release.md`
  - Maintainer runbook for PyPI Trusted Publisher setup, GitHub Release triggering, and partial-failure recovery.
- Modify: `docs/README.md`
  - Surface the release runbook from the main docs index.

### Task 1: Add a tested release metadata validator

**Files:**
- Create: `scripts/check_release_metadata.py`
- Create: `tests/scripts/test_check_release_metadata.py`
- Test: `tests/scripts/test_check_release_metadata.py`

- [ ] **Step 1: Write failing unit tests for release metadata validation**

```python
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_release_metadata import (
    expected_console_dependency,
    validate_release_metadata,
)


ROOT_PYPROJECT = Path("pyproject.toml")
CONSOLE_PYPROJECT = Path("console/pyproject.toml")


def test_expected_console_dependency_uses_major_minor_release_line() -> None:
    assert expected_console_dependency("0.1.0") == "agiwo ~= 0.1.0"
    assert expected_console_dependency("0.1.7") == "agiwo ~= 0.1.0"
    assert expected_console_dependency("1.4.2") == "agiwo ~= 1.4.0"


def test_validate_release_metadata_accepts_current_release_surface() -> None:
    validate_release_metadata(
        release_version="0.1.0",
        root_pyproject_path=ROOT_PYPROJECT,
        console_pyproject_path=CONSOLE_PYPROJECT,
    )


def test_validate_release_metadata_rejects_root_version_mismatch(tmp_path: Path) -> None:
    root_pyproject = tmp_path / "pyproject.toml"
    root_pyproject.write_text(
        "[project]\nname = 'agiwo'\nversion = '0.1.1'\n",
        encoding="utf-8",
    )

    console_dir = tmp_path / "console"
    console_dir.mkdir()
    console_pyproject = console_dir / "pyproject.toml"
    console_pyproject.write_text(
        "[project]\nname = 'agiwo-console'\nversion = '0.1.0'\ndependencies = ['agiwo ~= 0.1.0']\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="Root package version mismatch"):
        validate_release_metadata(
            release_version="0.1.0",
            root_pyproject_path=root_pyproject,
            console_pyproject_path=console_pyproject,
        )


def test_validate_release_metadata_rejects_console_dependency_drift(tmp_path: Path) -> None:
    root_pyproject = tmp_path / "pyproject.toml"
    root_pyproject.write_text(
        "[project]\nname = 'agiwo'\nversion = '0.2.1'\n",
        encoding="utf-8",
    )

    console_dir = tmp_path / "console"
    console_dir.mkdir()
    console_pyproject = console_dir / "pyproject.toml"
    console_pyproject.write_text(
        "[project]\nname = 'agiwo-console'\nversion = '0.2.1'\ndependencies = ['agiwo ~= 0.1.0']\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="Console dependency mismatch"):
        validate_release_metadata(
            release_version="0.2.1",
            root_pyproject_path=root_pyproject,
            console_pyproject_path=console_pyproject,
        )
```

- [ ] **Step 2: Run the targeted tests and confirm they fail because the validator does not exist yet**

Run: `uv run pytest tests/scripts/test_check_release_metadata.py -v`
Expected: FAIL with `ModuleNotFoundError` for `scripts.check_release_metadata`.

- [ ] **Step 3: Implement the release metadata validator**

```python
from __future__ import annotations

import sys
import tomllib
from pathlib import Path


def load_project(pyproject_path: Path) -> dict[str, object]:
    with pyproject_path.open("rb") as file:
        data = tomllib.load(file)
    project = data.get("project")
    if not isinstance(project, dict):
        raise SystemExit(f"Missing [project] table in {pyproject_path}")
    return project


def expected_console_dependency(release_version: str) -> str:
    major, minor, *_ = release_version.split(".")
    return f"agiwo ~= {major}.{minor}.0"


def validate_release_metadata(
    *,
    release_version: str,
    root_pyproject_path: Path,
    console_pyproject_path: Path,
) -> None:
    root_project = load_project(root_pyproject_path)
    console_project = load_project(console_pyproject_path)

    root_version = root_project.get("version")
    if root_version != release_version:
        raise SystemExit(
            f"Root package version mismatch: expected {release_version}, got {root_version}"
        )

    console_version = console_project.get("version")
    if console_version != release_version:
        raise SystemExit(
            f"Console package version mismatch: expected {release_version}, got {console_version}"
        )

    console_dependencies = console_project.get("dependencies")
    if not isinstance(console_dependencies, list):
        raise SystemExit("Console dependencies must be a list")

    expected_dependency = expected_console_dependency(release_version)
    if expected_dependency not in console_dependencies:
        raise SystemExit(
            "Console dependency mismatch: "
            f"expected {expected_dependency!r} in dependencies, got {console_dependencies!r}"
        )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit(
            "Usage: python scripts/check_release_metadata.py <release-version>"
        )

    validate_release_metadata(
        release_version=args[0],
        root_pyproject_path=Path("pyproject.toml"),
        console_pyproject_path=Path("console/pyproject.toml"),
    )
    print(f"release metadata ok: {args[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Re-run the targeted tests**

Run: `uv run pytest tests/scripts/test_check_release_metadata.py -v`
Expected: PASS.

- [ ] **Step 5: Run the validator against the current repository metadata**

Run: `uv run python scripts/check_release_metadata.py 0.1.0`
Expected: PASS with `release metadata ok: 0.1.0`.

- [ ] **Step 6: Commit the validator**

```bash
git add scripts/check_release_metadata.py tests/scripts/test_check_release_metadata.py
git commit -m "test: add release metadata validator"
```

### Task 2: Add the GitHub Release publishing workflow

**Files:**
- Create: `.github/workflows/release.yml`
- Modify: `.github/workflows/release.yml` only in this task
- Test: `.github/workflows/release.yml` via local package builds and smoke checks

- [ ] **Step 1: Add the release workflow with prepare, build, and publish job boundaries**

```yaml
name: Release Publish

on:
  release:
    types: [published]

concurrency:
  group: release-${{ github.event.release.tag_name }}
  cancel-in-progress: false

permissions:
  contents: read

jobs:
  prepare:
    name: Prepare Release
    runs-on: ubuntu-latest
    outputs:
      release_tag: ${{ steps.version.outputs.release_tag }}
      release_version: ${{ steps.version.outputs.release_version }}
    steps:
      - name: Normalize release tag
        id: version
        shell: bash
        run: |
          tag="${{ github.event.release.tag_name }}"
          if [[ ! "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            echo "Expected release tag in the form vX.Y.Z, got: $tag" >&2
            exit 1
          fi
          echo "release_tag=$tag" >> "$GITHUB_OUTPUT"
          echo "release_version=${tag#v}" >> "$GITHUB_OUTPUT"

  build-sdk:
    name: Build SDK
    runs-on: ubuntu-latest
    needs: prepare
    steps:
      - uses: actions/checkout@v4

      - uses: astral-sh/setup-uv@v4
        with:
          version: latest

      - name: Set up Python
        run: uv python install 3.11

      - name: Install dependencies
        run: uv sync

      - name: Validate release metadata
        run: uv run python scripts/check_release_metadata.py "${{ needs.prepare.outputs.release_version }}"

      - name: Build SDK package
        run: uv build

      - name: Smoke test built SDK wheel
        run: uv run python scripts/smoke_release_install.py "$(ls dist/agiwo-*.whl | head -n 1)"

      - name: Upload SDK artifacts
        uses: actions/upload-artifact@v4
        with:
          name: sdk-dist
          path: dist/*
          if-no-files-found: error

  publish-sdk:
    name: Publish SDK
    runs-on: ubuntu-latest
    needs: [prepare, build-sdk]
    environment: pypi
    permissions:
      contents: read
      id-token: write
    steps:
      - name: Download SDK artifacts
        uses: actions/download-artifact@v4
        with:
          name: sdk-dist
          path: dist

      - name: Publish SDK to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          packages-dir: dist/
          skip-existing: true
          print-hash: true

  build-console:
    name: Build Console
    runs-on: ubuntu-latest
    needs: [prepare, publish-sdk]
    steps:
      - uses: actions/checkout@v4

      - uses: astral-sh/setup-uv@v4
        with:
          version: latest

      - name: Set up Python
        run: uv python install 3.11

      - name: Install SDK dependencies
        run: uv sync

      - name: Install Console dependencies
        run: (cd console && uv sync)

      - name: Validate release metadata
        run: uv run python scripts/check_release_metadata.py "${{ needs.prepare.outputs.release_version }}"

      - name: Build SDK package
        run: uv build

      - name: Build Console package
        run: (cd console && uv build)

      - name: Smoke test built Console wheel
        run: uv run python scripts/smoke_release_install.py "$(ls dist/agiwo-*.whl | head -n 1)" "$(ls console/dist/agiwo_console-*.whl | head -n 1)"

      - name: Upload Console artifacts
        uses: actions/upload-artifact@v4
        with:
          name: console-dist
          path: console/dist/*
          if-no-files-found: error

  publish-console:
    name: Publish Console
    runs-on: ubuntu-latest
    needs: [prepare, build-console]
    environment: pypi
    permissions:
      contents: read
      id-token: write
    steps:
      - name: Download Console artifacts
        uses: actions/download-artifact@v4
        with:
          name: console-dist
          path: dist

      - name: Publish Console to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          packages-dir: dist/
          skip-existing: true
          print-hash: true
```

- [ ] **Step 2: Check the workflow file for the expected release trigger and OIDC publish boundaries**

Run: `rg -n "release:|types: \[published\]|id-token: write|skip-existing: true|pypa/gh-action-pypi-publish@release/v1" .github/workflows/release.yml`
Expected: PASS with matches for the release trigger, OIDC permissions, duplicate-tolerant publish, and PyPI publish action.

- [ ] **Step 3: Validate the release build path locally with the same commands used by the workflow**

Run: `uv sync && uv run python scripts/check_release_metadata.py 0.1.0 && uv build && (cd console && uv sync) && (cd console && uv build) && uv run python scripts/smoke_release_install.py "$(ls dist/agiwo-*.whl | head -n 1)" "$(ls console/dist/agiwo_console-*.whl | head -n 1)"`
Expected: PASS with `release metadata ok: 0.1.0`, `release smoke ok`, and `agiwo-console --help` succeeding.

- [ ] **Step 4: Commit the release workflow**

```bash
git add .github/workflows/release.yml
git commit -m "ci: publish packages from github releases"
```

### Task 3: Add maintainer release documentation

**Files:**
- Create: `docs/release.md`
- Modify: `docs/README.md`
- Test: `docs/release.md`, `docs/README.md`

- [ ] **Step 1: Add the maintainer release runbook**

```md
# Release Publishing

## Prerequisites

Configure PyPI Trusted Publisher for both projects before the first release:

- `agiwo`
- `agiwo-console`

Use the same GitHub repository and workflow:

- Repository: `xhwSkhizein/agiwo`
- Workflow file: `.github/workflows/release.yml`
- Environment: `pypi`

## Tag Format

Create GitHub Releases with tags in this format:

- `v0.1.0`
- `v0.1.1`

Do not create releases with bare tags like `0.1.0`.

## Publish Flow

1. Merge the release-ready changes into `main`.
2. Confirm both package versions match the intended release version.
3. Confirm `console/pyproject.toml` still depends on the matching `agiwo` release line.
4. Create a GitHub Release with tag `vX.Y.Z`.
5. Publish the release.
6. Open the `Release Publish` workflow in GitHub Actions and confirm all jobs succeed.

## Failure Handling

Publishing is not atomic.

- If `publish-sdk` fails, `agiwo-console` will not publish.
- If `publish-sdk` succeeds and `publish-console` fails, `agiwo` stays published.
- Fix the failure and rerun the workflow to publish the missing package.

The workflow is configured with duplicate-tolerant uploads so reruns can recover after a partial publish.
```

- [ ] **Step 2: Review the new release guide for the required operational details**

Run: `rg -n "Trusted Publisher|v0.1.0|Environment: `pypi`|not atomic|rerun" docs/release.md`
Expected: PASS with matches for Trusted Publisher setup, tag format, environment, and partial-failure handling.

- [ ] **Step 3: Link the release guide from the docs index**

```md
## Guides

- **[Custom Tools](./guides/custom-tools.md)** — Build your own tools
- **[Multi-Agent & Composition](./guides/multi-agent.md)** — Agent-as-tool and scheduler orchestration
- **[Streaming](./guides/streaming.md)** — Real-time streaming responses
- **[Hooks](./guides/hooks.md)** — Observe and intercept agent lifecycle events
- **[Storage & Observability](./guides/storage.md)** — Persist runs, sessions, and traces
- **[Skills](./guides/skills.md)** — File-based skill discovery and loading
- **[Release Publishing](./release.md)** — Maintainer runbook for GitHub Release driven PyPI publishing
```

- [ ] **Step 4: Confirm the docs index now exposes the release runbook**

Run: `rg -n "Release Publishing" docs/README.md docs/release.md`
Expected: PASS with a link in `docs/README.md` and the title in `docs/release.md`.

- [ ] **Step 5: Commit the docs update**

```bash
git add docs/README.md docs/release.md
git commit -m "docs: add github release publish runbook"
```

### Task 4: Run the final validation sweep

**Files:**
- Modify: none
- Test: `.github/workflows/release.yml`, `scripts/check_release_metadata.py`, `tests/scripts/test_check_release_metadata.py`, `docs/release.md`, `docs/README.md`

- [ ] **Step 1: Run the targeted tests and changed-file lint**

Run: `uv run pytest tests/scripts/test_check_release_metadata.py -v && uv run python scripts/lint.py changed`
Expected: PASS.

- [ ] **Step 2: Re-run the release build smoke end to end**

Run: `uv build && (cd console && uv build) && uv run python scripts/smoke_release_install.py "$(ls dist/agiwo-*.whl | head -n 1)" "$(ls console/dist/agiwo_console-*.whl | head -n 1)"`
Expected: PASS with `release smoke ok` and successful `agiwo-console --help` output.

- [ ] **Step 3: Run the repo guardrails required before shipping**

Run: `uv run ruff check --ignore C901 --ignore PLR0911 --ignore PLR0912 scripts/ tests/scripts/ && uv run ruff format --check scripts/ tests/scripts/ && uv run python scripts/repo_guard.py`
Expected: PASS, except for any already-known pre-existing warnings that are unchanged by this work.

- [ ] **Step 4: Push the branch and update the existing PR**

```bash
git push origin feat/first-release-prep
```

Expected: PASS and the PR shows the release workflow, validator, and docs changes.
