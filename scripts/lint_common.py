"""
Shared helpers for repository lint scripts.
"""

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEVANT_PREFIXES = (
    Path("agiwo"),
    Path("console/server"),
    Path("tests"),
    Path("console/tests"),
    Path("scripts"),
)


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def run_command(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> None:
    printable = [str(part) for part in command]
    print(f"$ {shlex.join(printable)}")
    completed = subprocess.run(printable, cwd=cwd or ROOT, env=env)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def build_pythonpath_env() -> dict[str, str]:
    env = dict(os.environ)
    parts = [str(ROOT / "console"), str(ROOT)]
    existing = env.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def git_output(command: list[str]) -> list[str] | None:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def git_has_head() -> bool:
    return git_output(["git", "rev-parse", "--verify", "HEAD"]) is not None


def normalize_paths(paths: list[str | Path]) -> list[Path]:
    normalized: list[Path] = []
    seen: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        candidate = path if path.is_absolute() else ROOT / path
        if not candidate.exists():
            continue
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(ROOT)
        except ValueError:
            continue
        relative_str = relative.as_posix()
        if not any(
            relative_str == prefix.as_posix()
            or relative_str.startswith(f"{prefix.as_posix()}/")
            for prefix in RELEVANT_PREFIXES
        ):
            continue
        if relative_str in seen:
            continue
        seen.add(relative_str)
        normalized.append(relative)
    normalized.sort()
    return normalized


def collect_changed_paths() -> list[Path]:
    raw_paths: list[str] = []
    if git_has_head():
        tracked = git_output(
            ["git", "diff", "--name-only", "--diff-filter=ACMR", "HEAD", "--"]
        )
        if tracked is not None:
            raw_paths.extend(tracked)
    else:
        tracked = git_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--"]
        )
        if tracked is not None:
            raw_paths.extend(tracked)

    untracked = git_output(["git", "ls-files", "--others", "--exclude-standard", "--"])
    if untracked is not None:
        raw_paths.extend(untracked)
    return normalize_paths(raw_paths)


def collect_python_paths(paths: list[Path]) -> list[Path]:
    return [path for path in paths if path.suffix == ".py"]


def iter_all_python_paths() -> list[Path]:
    paths: list[Path] = []
    for prefix in RELEVANT_PREFIXES:
        directory = ROOT / prefix
        if not directory.exists():
            continue
        for path in directory.rglob("*.py"):
            paths.append(path.relative_to(ROOT))
    paths.sort()
    return paths


def require_commands(*commands: str) -> None:
    missing = [command for command in commands if not command_exists(command)]
    if not missing:
        return
    print(
        "Missing required tooling: "
        + ", ".join(missing)
        + ". Run `uv sync` before linting.",
        file=sys.stderr,
    )
    raise SystemExit(2)
