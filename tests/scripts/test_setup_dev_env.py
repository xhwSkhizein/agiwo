from pathlib import Path

from scripts.setup_dev_env import migrate_console_env_file


def test_migrate_console_env_file_rewrites_legacy_keys(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "AGIWO_CONSOLE_HOST=0.0.0.0",
                "AGIWO_CONSOLE_PORT=8422",
                "AGIWO_CONSOLE_RUN_STEP_STORAGE_TYPE=sqlite",
                "AGIWO_CONSOLE_FEISHU_ENABLED=true",
                "AGIWO_CONSOLE_DEFAULT_AGENT__NAME=Walaha",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert migrate_console_env_file(env_path) is True

    migrated = env_path.read_text(encoding="utf-8")
    assert "AGIWO_CONSOLE_SERVER__HOST=0.0.0.0" in migrated
    assert "AGIWO_CONSOLE_SERVER__PORT=8422" in migrated
    assert "AGIWO_CONSOLE_STORAGE__RUN_LOG_TYPE=sqlite" in migrated
    assert "AGIWO_CONSOLE_CHANNELS__FEISHU__ENABLED=true" in migrated
    assert "AGIWO_CONSOLE_DEFAULT_AGENT__NAME=Walaha" in migrated
    assert "AGIWO_CONSOLE_HOST=" not in migrated


def test_migrate_console_env_file_is_idempotent(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("AGIWO_CONSOLE_SERVER__HOST=127.0.0.1\n", encoding="utf-8")

    assert migrate_console_env_file(env_path) is False
