import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/smoke_release_install.py <wheel-path>")

    wheel_path = Path(sys.argv[1]).resolve()
    if not wheel_path.is_file():
        raise SystemExit(f"Wheel not found: {wheel_path}")

    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv executable not found in PATH")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        venv_path = tmp_path / "venv"
        python_path = (
            venv_path / "Scripts" / "python.exe"
            if sys.platform == "win32"
            else venv_path / "bin" / "python"
        )

        run([uv, "venv", str(venv_path)])
        run([uv, "pip", "install", "--python", str(python_path), str(wheel_path)])
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
