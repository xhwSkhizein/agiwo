from pathlib import Path

import scripts.smoke_release_install as smoke_release_install


def test_sdk_wheel_force_includes_templates() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "[tool.hatch.build.targets.wheel.force-include]" in pyproject
    assert '"templates" = "templates"' in pyproject


def test_build_sdk_smoke_code_checks_installed_templates() -> None:
    smoke_code = smoke_release_install.build_sdk_smoke_code()

    assert (
        "templates_dir = Path(agiwo.__file__).resolve().parent.parent / 'templates'"
        in smoke_code
    )
    assert "assert templates_dir.is_dir()" in smoke_code
    assert "IDENTITY.md" in smoke_code
    assert "SOUL.md" in smoke_code
    assert "USER.md" in smoke_code
