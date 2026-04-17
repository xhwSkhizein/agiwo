import ast
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None

VERSION_PATTERN = re.compile(r"^(version\s*=\s*)(['\"])([^'\"]+)(['\"])$", re.MULTILINE)
CONSOLE_DEP_PATTERN = re.compile(r"agiwo\s*~=\s*\d+\.\d+\.\d+")


def load_project(pyproject_path: Path) -> dict[str, object]:
    if tomllib is not None:
        with pyproject_path.open("rb") as file:
            data = tomllib.load(file)
    else:
        data = _load_project_fallback(pyproject_path)
    project = data.get("project")
    if not isinstance(project, dict):
        raise SystemExit(f"Missing [project] table in {pyproject_path}")
    return project


def _load_project_fallback(pyproject_path: Path) -> dict[str, object]:
    content = pyproject_path.read_text(encoding="utf-8")
    return {"project": _parse_project_table(content, pyproject_path)}


def _parse_project_table(content: str, pyproject_path: Path) -> dict[str, object]:
    project_lines = _extract_project_lines(content, pyproject_path)

    project: dict[str, object] = {}
    idx = 0
    while idx < len(project_lines):
        line = project_lines[idx].strip()
        if not line or line.startswith("#"):
            idx += 1
            continue
        if "=" not in line:
            raise SystemExit(f"Unsupported TOML syntax in {pyproject_path}: {line}")
        key, value = line.split("=", 1)
        key = key.strip()
        value_parts, idx = _collect_value_parts(
            project_lines, idx, value.strip(), pyproject_path
        )
        parsed_value = _parse_toml_value(" ".join(value_parts), pyproject_path)
        project[key] = parsed_value
        idx += 1

    return project


def _extract_project_lines(content: str, pyproject_path: Path) -> list[str]:
    in_project = False
    project_lines: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if stripped == "[project]":
                in_project = True
                continue
            if in_project:
                break
        if in_project:
            project_lines.append(line)
    if not in_project:
        raise SystemExit(f"Missing [project] table in {pyproject_path}")
    return project_lines


def _collect_value_parts(
    project_lines: list[str],
    idx: int,
    initial_value: str,
    pyproject_path: Path,
) -> tuple[list[str], int]:
    value_parts = [initial_value]
    if not initial_value.startswith("["):
        return value_parts, idx

    while value_parts[-1] and not value_parts[-1].rstrip().endswith("]"):
        idx += 1
        if idx >= len(project_lines):
            raise SystemExit(f"Unterminated TOML array in {pyproject_path}")
        value_parts.append(project_lines[idx].strip())
    return value_parts, idx


def _parse_toml_value(value: str, pyproject_path: Path) -> object:
    normalized = value.strip()
    if normalized.startswith("["):
        try:
            parsed = ast.literal_eval(normalized)
        except (SyntaxError, ValueError) as exc:
            raise SystemExit(
                f"Unsupported TOML array value in {pyproject_path}"
            ) from exc
        if not isinstance(parsed, list):
            raise SystemExit(f"Expected TOML array value in {pyproject_path}")
        return parsed
    if normalized.startswith(("'", '"')):
        try:
            parsed = ast.literal_eval(normalized)
        except (SyntaxError, ValueError) as exc:
            raise SystemExit(
                f"Unsupported TOML string value in {pyproject_path}"
            ) from exc
        if not isinstance(parsed, str):
            raise SystemExit(f"Expected TOML string value in {pyproject_path}")
        return parsed
    return normalized


def parse_release_version(release_version: str) -> tuple[str, str, str]:
    parts = release_version.split(".")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise SystemExit(
            f"Release version must be in the form X.Y.Z, got: {release_version}"
        )
    return parts[0], parts[1], parts[2]


def expected_console_dependency(release_version: str) -> str:
    major, minor, _patch = parse_release_version(release_version)
    return f"agiwo ~= {major}.{minor}.0"


def replace_single_pattern(
    *,
    content: str,
    pattern: re.Pattern[str],
    replacement: str | re.Match[str] | object,
    path: Path,
    error_label: str,
) -> str:
    updated_content, replacements = pattern.subn(replacement, content, count=1)
    if replacements != 1:
        raise SystemExit(f"Could not update {error_label} in {path}")
    return updated_content


def update_project_version(pyproject_path: Path, release_version: str) -> None:
    content = pyproject_path.read_text(encoding="utf-8")
    updated_content = replace_single_pattern(
        content=content,
        pattern=VERSION_PATTERN,
        replacement=lambda match: (
            f"{match.group(1)}{match.group(2)}{release_version}{match.group(4)}"
        ),
        path=pyproject_path,
        error_label="project version",
    )
    pyproject_path.write_text(updated_content, encoding="utf-8")


def update_console_dependency(
    console_pyproject_path: Path, release_version: str
) -> None:
    content = console_pyproject_path.read_text(encoding="utf-8")
    updated_content = replace_single_pattern(
        content=content,
        pattern=CONSOLE_DEP_PATTERN,
        replacement=expected_console_dependency(release_version),
        path=console_pyproject_path,
        error_label="console agiwo dependency",
    )
    console_pyproject_path.write_text(updated_content, encoding="utf-8")


def apply_release_metadata(
    *,
    release_version: str,
    root_pyproject_path: Path,
    console_pyproject_path: Path,
) -> None:
    parse_release_version(release_version)
    update_project_version(root_pyproject_path, release_version)
    update_project_version(console_pyproject_path, release_version)
    update_console_dependency(console_pyproject_path, release_version)


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
    apply_changes = False
    if args and args[0] == "--apply":
        apply_changes = True
        args = args[1:]

    if len(args) != 1:
        raise SystemExit(
            "Usage: python scripts/check_release_metadata.py [--apply] <release-version>"
        )

    release_version = args[0]
    root_pyproject_path = Path("pyproject.toml")
    console_pyproject_path = Path("console/pyproject.toml")

    if apply_changes:
        apply_release_metadata(
            release_version=release_version,
            root_pyproject_path=root_pyproject_path,
            console_pyproject_path=console_pyproject_path,
        )

    validate_release_metadata(
        release_version=release_version,
        root_pyproject_path=root_pyproject_path,
        console_pyproject_path=console_pyproject_path,
    )

    action = "applied and validated" if apply_changes else "validated"
    print(f"release metadata {action}: {release_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
