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
    parts = release_version.split(".")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise SystemExit(
            f"Release version must be in the form X.Y.Z, got: {release_version}"
        )
    major, minor, _patch = parts
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
            "Console package version mismatch: "
            f"expected {release_version}, got {console_version}"
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
