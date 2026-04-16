# Console CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the packaged Console startup command with a user-facing `agiwo-console serve` entrypoint.

**Architecture:** Add a thin CLI wrapper around the existing ASGI app startup path. The CLI only parses arguments and forwards them into `uvicorn.run("server.app:app", ...)`, while packaging and docs are updated to expose the new command.

**Tech Stack:** Python 3.11+, argparse, uvicorn, Hatchling, pytest

---

## File Structure

- Modify: `console/pyproject.toml`
  - Publish the `agiwo-console` script entry.
- Create: `console/server/cli.py`
  - Provide the top-level CLI parser and `serve` subcommand.
- Create: `console/tests/test_cli.py`
  - Verify argument parsing and `uvicorn.run(...)` dispatch.
- Modify: `README.md`
  - Replace packaged Console startup instructions with `agiwo-console serve`.
- Modify: `docs/console/overview.md`
  - Replace startup instructions with `agiwo-console serve`.
- Modify: `docs/console/feishu.md`
  - Keep Feishu setup docs aligned with the new startup command.
- Modify: `scripts/smoke_release_install.py`
  - Extend release smoke to verify the installed `agiwo-console` command can print help.

### Task 1: Add the packaged `agiwo-console` entrypoint

**Files:**
- Modify: `console/pyproject.toml`
- Create: `console/server/cli.py`
- Test: `console/tests/test_cli.py`

- [ ] **Step 1: Add a failing CLI test before implementing the command**

```python
from unittest.mock import patch

from server.cli import main


def test_cli_serve_dispatches_to_uvicorn():
    with patch("server.cli.uvicorn.run") as run:  # noqa: SIM117
        exit_code = main(["serve", "--host", "127.0.0.1", "--port", "9999"])

    assert exit_code == 0
    run.assert_called_once_with(
        "server.app:app",
        host="127.0.0.1",
        port=9999,
        reload=False,
        env_file=None,
        factory=False,
    )
```

- [ ] **Step 2: Run the new targeted test file to confirm it fails**

Run: `cd console && uv run pytest tests/test_cli.py -q`

Expected: FAIL because `server.cli` does not exist yet.

- [ ] **Step 3: Add the package script entry to `console/pyproject.toml`**

```toml
[project.scripts]
agiwo-console = "server.cli:main"
```

- [ ] **Step 4: Implement the thin CLI wrapper in `console/server/cli.py`**

```python
import argparse
from collections.abc import Sequence

import uvicorn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agiwo-console")
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="Start the Agiwo Console server")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8422)
    serve.add_argument("--env-file", default=None)
    serve.add_argument("--reload", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command != "serve":
        parser.print_help()
        return 0

    uvicorn.run(
        "server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        env_file=args.env_file,
        factory=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Expand tests to cover `--env-file` and `--reload` pass-through**

```python
def test_cli_serve_forwards_reload_and_env_file():
    with patch("server.cli.uvicorn.run") as run:
        exit_code = main(
            [
                "serve",
                "--env-file",
                "/tmp/console.env",
                "--reload",
            ]
        )

    assert exit_code == 0
    run.assert_called_once_with(
        "server.app:app",
        host="0.0.0.0",
        port=8422,
        reload=True,
        env_file="/tmp/console.env",
        factory=False,
    )
```

- [ ] **Step 6: Run the targeted CLI tests**

Run: `cd console && uv run pytest tests/test_cli.py -q`

Expected: PASS

- [ ] **Step 7: Run the full Console backend test suite**

Run: `cd console && uv run pytest tests/ -q`

Expected: PASS

- [ ] **Step 8: Commit the CLI entrypoint implementation**

```bash
git add console/pyproject.toml console/server/cli.py console/tests/test_cli.py
git commit -m "feat: add agiwo-console serve command"
```

### Task 2: Update packaged Console docs and smoke verification

**Files:**
- Modify: `README.md`
- Modify: `docs/console/overview.md`
- Modify: `docs/console/feishu.md`
- Modify: `scripts/smoke_release_install.py`

- [ ] **Step 1: Replace packaged startup docs in the README**

```md
```bash
pip install agiwo-console
agiwo-console serve --env-file .env
```
```

- [ ] **Step 2: Replace startup docs in Console overview and Feishu docs**

```md
```bash
agiwo-console serve --env-file .env
```
```

- [ ] **Step 3: Extend the release smoke script to verify the installed command exists**

```python
        cli_path = (
            venv_path / "Scripts" / "agiwo-console.exe"
            if sys.platform == "win32"
            else venv_path / "bin" / "agiwo-console"
        )
        run([str(cli_path), "--help"])
```

- [ ] **Step 4: Rebuild the packages and run smoke validation**

Run:

```bash
uv build
cd console && uv build
cd .. && uv run python scripts/smoke_release_install.py dist/agiwo-0.1.0-py3-none-any.whl
```

Expected: PASS, including installed `agiwo-console --help` output from the fresh environment.

- [ ] **Step 5: Run a grep to ensure release-facing docs no longer expose `server.app:app` as the packaged startup path**

Run: `rg -n 'uvicorn server.app:app' README.md docs/console`

Expected: no matches

- [ ] **Step 6: Commit the docs and smoke cleanup**

```bash
git add README.md docs/console/overview.md docs/console/feishu.md scripts/smoke_release_install.py
git commit -m "docs: publish console cli startup flow"
```

## Self-Review

- Spec coverage:
  - package entrypoint: Task 1
  - `serve` command: Task 1
  - docs replacement: Task 2
  - installed-package smoke: Task 2
- Placeholder scan:
  - No `TODO`, `TBD`, or vague steps remain
- Type consistency:
  - The CLI always dispatches through `main(argv: Sequence[str] | None = None) -> int` and forwards into `uvicorn.run("server.app:app", ...)`
