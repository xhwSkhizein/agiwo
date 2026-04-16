from pathlib import Path

import pytest

from scripts.check_release_metadata import (
    expected_console_dependency,
    validate_release_metadata,
)


ROOT_PYPROJECT = Path("pyproject.toml")
CONSOLE_PYPROJECT = Path("console/pyproject.toml")


def test_expected_console_dependency_uses_major_minor_release_line() -> None:
    assert expected_console_dependency("0.1.0") == "agiwo ~= 0.1.0"
    assert expected_console_dependency("0.1.7") == "agiwo ~= 0.1.0"
    assert expected_console_dependency("1.4.2") == "agiwo ~= 1.4.0"


def test_validate_release_metadata_accepts_current_release_surface() -> None:
    validate_release_metadata(
        release_version="0.1.0",
        root_pyproject_path=ROOT_PYPROJECT,
        console_pyproject_path=CONSOLE_PYPROJECT,
    )


def test_validate_release_metadata_rejects_root_version_mismatch(
    tmp_path: Path,
) -> None:
    root_pyproject = tmp_path / "pyproject.toml"
    root_pyproject.write_text(
        "[project]\nname = 'agiwo'\nversion = '0.1.1'\n",
        encoding="utf-8",
    )

    console_dir = tmp_path / "console"
    console_dir.mkdir()
    console_pyproject = console_dir / "pyproject.toml"
    console_pyproject.write_text(
        "[project]\nname = 'agiwo-console'\nversion = '0.1.0'\n"
        "dependencies = ['agiwo ~= 0.1.0']\n",
        encoding="utf-8",
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
    root_pyproject.write_text(
        "[project]\nname = 'agiwo'\nversion = '0.2.1'\n",
        encoding="utf-8",
    )

    console_dir = tmp_path / "console"
    console_dir.mkdir()
    console_pyproject = console_dir / "pyproject.toml"
    console_pyproject.write_text(
        "[project]\nname = 'agiwo-console'\nversion = '0.2.1'\n"
        "dependencies = ['agiwo ~= 0.1.0']\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="Console dependency mismatch"):
        validate_release_metadata(
            release_version="0.2.1",
            root_pyproject_path=root_pyproject,
            console_pyproject_path=console_pyproject,
        )
