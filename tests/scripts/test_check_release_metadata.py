from pathlib import Path

import pytest

import scripts.check_release_metadata as release_metadata
from scripts.check_release_metadata import (
    apply_release_metadata,
    expected_console_dependency,
    load_project,
    validate_release_metadata,
)


ROOT_PYPROJECT = Path("pyproject.toml")
CONSOLE_PYPROJECT = Path("console/pyproject.toml")


def write_release_fixture(
    *,
    root_pyproject: Path,
    console_pyproject: Path,
    root_version: str,
    console_version: str,
    console_dependency: str,
) -> None:
    root_pyproject.write_text(
        f"[project]\nname = 'agiwo'\nversion = '{root_version}'\n",
        encoding="utf-8",
    )
    console_pyproject.write_text(
        "[project]\n"
        "name = 'agiwo-console'\n"
        f"version = '{console_version}'\n"
        f"dependencies = ['{console_dependency}']\n",
        encoding="utf-8",
    )


def test_expected_console_dependency_uses_major_minor_release_line() -> None:
    assert expected_console_dependency("0.1.0") == "agiwo ~= 0.1.0"
    assert expected_console_dependency("0.1.7") == "agiwo ~= 0.1.0"
    assert expected_console_dependency("1.4.2") == "agiwo ~= 1.4.0"


def test_load_project_supports_python310_fallback_parser(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[project]\n"
        "name = 'agiwo'\n"
        "version = '0.1.0'\n"
        "dependencies = [\n"
        "    'agiwo ~= 0.1.0',\n"
        "]\n",
        encoding="utf-8",
    )

    original_tomllib = release_metadata.tomllib
    release_metadata.tomllib = None
    try:
        project = load_project(pyproject)
    finally:
        release_metadata.tomllib = original_tomllib

    assert project["name"] == "agiwo"
    assert project["version"] == "0.1.0"
    assert project["dependencies"] == ["agiwo ~= 0.1.0"]


def test_validate_release_metadata_accepts_current_release_surface() -> None:
    validate_release_metadata(
        release_version="0.1.0",
        root_pyproject_path=ROOT_PYPROJECT,
        console_pyproject_path=CONSOLE_PYPROJECT,
    )


def test_apply_release_metadata_updates_versions_and_dependency(tmp_path: Path) -> None:
    root_pyproject = tmp_path / "pyproject.toml"
    console_dir = tmp_path / "console"
    console_dir.mkdir()
    console_pyproject = console_dir / "pyproject.toml"
    write_release_fixture(
        root_pyproject=root_pyproject,
        console_pyproject=console_pyproject,
        root_version="0.1.0",
        console_version="0.1.0",
        console_dependency="agiwo ~= 0.1.0",
    )

    apply_release_metadata(
        release_version="0.0.1",
        root_pyproject_path=root_pyproject,
        console_pyproject_path=console_pyproject,
    )

    validate_release_metadata(
        release_version="0.0.1",
        root_pyproject_path=root_pyproject,
        console_pyproject_path=console_pyproject,
    )


def test_validate_release_metadata_rejects_root_version_mismatch(
    tmp_path: Path,
) -> None:
    root_pyproject = tmp_path / "pyproject.toml"
    console_dir = tmp_path / "console"
    console_dir.mkdir()
    console_pyproject = console_dir / "pyproject.toml"
    write_release_fixture(
        root_pyproject=root_pyproject,
        console_pyproject=console_pyproject,
        root_version="0.1.1",
        console_version="0.1.0",
        console_dependency="agiwo ~= 0.1.0",
    )

    with pytest.raises(SystemExit, match="Root package version mismatch"):
        validate_release_metadata(
            release_version="0.1.0",
            root_pyproject_path=root_pyproject,
            console_pyproject_path=console_pyproject,
        )


def test_validate_release_metadata_rejects_console_dependency_drift(
    tmp_path: Path,
) -> None:
    root_pyproject = tmp_path / "pyproject.toml"
    console_dir = tmp_path / "console"
    console_dir.mkdir()
    console_pyproject = console_dir / "pyproject.toml"
    write_release_fixture(
        root_pyproject=root_pyproject,
        console_pyproject=console_pyproject,
        root_version="0.2.1",
        console_version="0.2.1",
        console_dependency="agiwo ~= 0.1.0",
    )

    with pytest.raises(SystemExit, match="Console dependency mismatch"):
        validate_release_metadata(
            release_version="0.2.1",
            root_pyproject_path=root_pyproject,
            console_pyproject_path=console_pyproject,
        )
