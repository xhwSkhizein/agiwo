"""
Repository verification entry points for hooks and manual checks.
"""

import argparse
import os

from lint_common import ROOT, run_command


def _build_console_test_env() -> dict[str, str]:
    env = dict(os.environ)
    env["AGIWO_ROOT_PATH"] = str(ROOT / "console" / ".agiwo")
    return env


def _run_sdk_tests() -> None:
    run_command(["uv", "run", "pytest", "tests/", "-v", "--tb=short"])


def _run_console_tests() -> None:
    run_command(
        ["uv", "run", "pytest", "tests/", "-v", "--tb=short"],
        cwd=ROOT / "console",
        env=_build_console_test_env(),
    )


def _run_pre_push() -> None:
    run_command(["uv", "run", "python", "scripts/lint.py", "ci"])
    _run_sdk_tests()
    _run_console_tests()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run repository verification workflows."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "sdk-tests",
        help="Run the SDK test suite used by local push checks.",
    )
    subparsers.add_parser(
        "console-tests",
        help="Run the Console backend test suite with an isolated AGIWO_ROOT_PATH.",
    )
    subparsers.add_parser(
        "pre-push",
        help="Run the full local push gate (lint + SDK tests + Console tests).",
    )

    args = parser.parse_args()

    if args.command == "sdk-tests":
        _run_sdk_tests()
        return

    if args.command == "console-tests":
        _run_console_tests()
        return

    if args.command == "pre-push":
        _run_pre_push()
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
