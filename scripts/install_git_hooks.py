import subprocess
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    hooks_dir = root / ".githooks"
    subprocess.run(
        ["git", "config", "core.hooksPath", str(hooks_dir)],
        check=True,
        cwd=root,
    )
    print(f"git hooks installed: {hooks_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
