"""
Low-noise lint entry point for AI-driven changes.
"""

import argparse
import sys
from pathlib import Path

from lint_common import (
    ROOT,
    build_pythonpath_env,
    collect_changed_paths,
    collect_python_paths,
    normalize_paths,
    require_commands,
    run_command,
)


def _run_ruff(paths: list[Path]) -> None:
    if not paths:
        print("ruff: no Python files to check")
        return
    command = ["ruff", "check", *[path.as_posix() for path in paths]]
    run_command(command)


def _run_ruff_format_check() -> None:
    command = [
        "ruff",
        "format",
        "--check",
        "agiwo/",
        "console/server/",
        "tests/",
        "console/tests/",
        "scripts/",
    ]
    run_command(command)


def _run_repo_guard(paths: list[Path]) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts/repo_guard.py"),
        *[path.as_posix() for path in paths],
    ]
    run_command(command)


def _run_import_contracts() -> None:
    env = build_pythonpath_env()
    command = ["lint-imports", "--config", "lint/importlinter_agiwo.ini"]
    run_command(command, env=env)


def _run_bundle(paths: list[Path]) -> None:
    require_commands("ruff", "lint-imports")
    python_paths = collect_python_paths(paths)
    _run_ruff(python_paths)
    _run_repo_guard(python_paths)
    _run_import_contracts()


def _run_ci_lint() -> None:
    require_commands("ruff", "lint-imports")
    run_command(
        [
            "ruff",
            "check",
            "--ignore",
            "C901",
            "--ignore",
            "PLR0911",
            "--ignore",
            "PLR0912",
            "agiwo/",
            "console/server/",
            "tests/",
            "console/tests/",
            "scripts/",
        ]
    )
    _run_ruff_format_check()
    _run_import_contracts()
    run_command([sys.executable, str(ROOT / "scripts/repo_guard.py")])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the repository lint workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "changed",
        help="Lint the current Git working tree plus global import contracts.",
    )
    subparsers.add_parser(
        "ci",
        help="Run the CI-equivalent lightweight lint gate.",
    )

    files_parser = subparsers.add_parser(
        "files",
        help="Lint an explicit file list plus global import contracts.",
    )
    files_parser.add_argument("files", nargs="+", help="Files to lint.")

    subparsers.add_parser(
        "imports",
        help="Run only the import architecture contracts.",
    )

    args = parser.parse_args()

    if args.command == "changed":
        _run_bundle(collect_changed_paths())
        return

    if args.command == "ci":
        _run_ci_lint()
        return

    if args.command == "files":
        _run_bundle(normalize_paths(args.files))
        return

    if args.command == "imports":
        require_commands("lint-imports")
        _run_import_contracts()
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
