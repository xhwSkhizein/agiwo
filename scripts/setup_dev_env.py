"""
Rebuild local SDK/Console virtual environments with uv-managed Python.

Use this when `.venv` was created from conda/homebrew Python and fails to
start (for example macOS SIGKILL / invalid code signature), or after cloning
the repository on a machine where conda is first on PATH.
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONSOLE_DIR = ROOT / "console"
PINNED_PYTHON = (ROOT / ".python-version").read_text(encoding="utf-8").strip()

_LEGACY_CONSOLE_ENV_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"^AGIWO_CONSOLE_HOST=", "AGIWO_CONSOLE_SERVER__HOST="),
    (r"^AGIWO_CONSOLE_PORT=", "AGIWO_CONSOLE_SERVER__PORT="),
    (r"^AGIWO_CONSOLE_RUN_STEP_STORAGE_TYPE=", "AGIWO_CONSOLE_STORAGE__RUN_LOG_TYPE="),
    (r"^AGIWO_CONSOLE_RUN_LOG_STORAGE_TYPE=", "AGIWO_CONSOLE_STORAGE__RUN_LOG_TYPE="),
    (r"^AGIWO_CONSOLE_TRACE_STORAGE_TYPE=", "AGIWO_CONSOLE_STORAGE__TRACE_TYPE="),
    (r"^AGIWO_CONSOLE_METADATA_STORAGE_TYPE=", "AGIWO_CONSOLE_STORAGE__METADATA_TYPE="),
    (r"^AGIWO_CONSOLE_FEISHU_", "AGIWO_CONSOLE_CHANNELS__FEISHU__"),
)


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    location = f" ({cwd})" if cwd is not None else ""
    print(f"+ {' '.join(cmd)}{location}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _resolve_python(venv_python: Path) -> Path:
    if not venv_python.exists():
        return venv_python
    return venv_python.resolve()


def _python_origin(python_path: Path) -> str:
    resolved = _resolve_python(python_path)
    text = str(resolved)
    if "miniconda" in text or "anaconda" in text:
        return "conda"
    if "/.local/share/uv/python/" in text:
        return "uv-managed"
    if text.startswith("/usr/bin/"):
        return "system"
    return "other"


def _inspect_venv(project_dir: Path) -> tuple[Path, str] | None:
    venv_python = project_dir / ".venv" / "bin" / "python3"
    if not venv_python.exists():
        return None
    return venv_python, _python_origin(venv_python)


def migrate_console_env_file(path: Path) -> bool:
    if not path.exists():
        return False

    original = path.read_text(encoding="utf-8")
    lines: list[str] = []
    changed = False
    for line in original.splitlines():
        updated = line
        for pattern, replacement in _LEGACY_CONSOLE_ENV_REPLACEMENTS:
            new_line = re.sub(pattern, replacement, updated)
            if new_line != updated:
                changed = True
                updated = new_line
                break
        lines.append(updated)

    if not changed:
        return False

    path.write_text(
        "\n".join(lines) + ("\n" if original.endswith("\n") else ""), encoding="utf-8"
    )
    return True


def _verify_python(project_dir: Path) -> None:
    venv_python = project_dir / ".venv" / "bin" / "python3"
    if not venv_python.exists():
        raise RuntimeError(f"Missing virtual environment at {project_dir / '.venv'}")

    origin = _python_origin(venv_python)
    if origin == "conda":
        raise RuntimeError(
            f"{project_dir} still uses conda Python at {venv_python.resolve()}. "
            "Re-run setup after ensuring conda is not overriding uv."
        )

    _run(
        [str(venv_python), "-c", "import sys; print(sys.version.split()[0])"],
        cwd=project_dir,
    )


def setup(*, migrate_env: bool, install_hooks: bool) -> None:
    for project_dir in (ROOT, CONSOLE_DIR):
        current = _inspect_venv(project_dir)
        if current is None:
            print(f"! no .venv in {project_dir}", flush=True)
            continue
        venv_python, origin = current
        print(
            f"! detected {origin} Python for {project_dir}: {venv_python.resolve()}",
            flush=True,
        )

    if migrate_env:
        for env_path in (CONSOLE_DIR / ".env", CONSOLE_DIR / ".env.example.full"):
            if migrate_console_env_file(env_path):
                print(f"✓ migrated legacy Console env keys in {env_path}", flush=True)

    _run(["uv", "python", "install", PINNED_PYTHON], cwd=ROOT)
    _run(["uv", "python", "pin", PINNED_PYTHON], cwd=ROOT)

    for project_dir in (ROOT, CONSOLE_DIR):
        venv_dir = project_dir / ".venv"
        if venv_dir.exists():
            print(f"! removing {venv_dir}", flush=True)
            shutil.rmtree(venv_dir)
        _run(["uv", "sync"], cwd=project_dir)
        _verify_python(project_dir)

    if install_hooks:
        _run(["uv", "run", "python", "scripts/install_git_hooks.py"], cwd=ROOT)

    print("\nDev environment is ready.", flush=True)
    print(f"  SDK:     cd {ROOT} && uv run python -V", flush=True)
    print(
        f"  Console: cd {CONSOLE_DIR} && uv run agiwo-console serve --env-file .env",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-env-migrate",
        action="store_true",
        help="Do not rewrite legacy AGIWO_CONSOLE_* keys in console/.env files.",
    )
    parser.add_argument(
        "--install-hooks",
        action="store_true",
        help="Install repository git hooks after rebuilding virtual environments.",
    )
    args = parser.parse_args(argv)

    try:
        setup(migrate_env=not args.skip_env_migrate, install_hooks=args.install_hooks)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
