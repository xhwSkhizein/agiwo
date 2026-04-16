import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TIMEOUT_SECONDS = 300


def run(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True, timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        print(
            "Command timed out after "
            f"{TIMEOUT_SECONDS}s: {' '.join(str(part) for part in cmd)}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def resolve_cli_path(venv_path: Path, name: str) -> Path:
    if sys.platform == "win32":
        return venv_path / "Scripts" / f"{name}.exe"
    return venv_path / "bin" / name


def main() -> int:
    if not (2 <= len(sys.argv) <= 3):
        raise SystemExit(
            "Usage: python scripts/smoke_release_install.py <agiwo-wheel-path> [agiwo-console-wheel-path]"
        )

    sdk_wheel_path = Path(sys.argv[1]).resolve()
    if not sdk_wheel_path.is_file():
        raise SystemExit(f"Wheel not found: {sdk_wheel_path}")

    console_wheel_path: Path | None = None
    if len(sys.argv) == 3:
        console_wheel_path = Path(sys.argv[2]).resolve()
        if not console_wheel_path.is_file():
            raise SystemExit(f"Wheel not found: {console_wheel_path}")

    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv executable not found in PATH")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        venv_path = tmp_path / "venv"
        python_path = resolve_cli_path(venv_path, "python")

        run([uv, "venv", str(venv_path)])
        install_cmd = [
            uv,
            "pip",
            "install",
            "--python",
            str(python_path),
            str(sdk_wheel_path),
        ]
        if console_wheel_path is not None:
            install_cmd.append(str(console_wheel_path))
        run(install_cmd)
        run(
            [
                str(python_path),
                "-c",
                (
                    "from agiwo.llm import OpenAIModel; "
                    "from agiwo.tool.manager import ToolManager; "
                    "model = OpenAIModel(name='gpt-5.4', api_key='test-key'); "
                    "assert model.id == 'gpt-5.4'; "
                    "assert model.name == 'gpt-5.4'; "
                    "defaults = set(ToolManager().list_default_tool_names()); "
                    "assert {'bash', 'bash_process', 'web_search', 'web_reader', 'memory_retrieval'} <= defaults; "
                    "print('release smoke ok')"
                ),
            ]
        )

        if console_wheel_path is not None:
            cli_path = resolve_cli_path(venv_path, "agiwo-console")
            run([str(cli_path), "--help"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
