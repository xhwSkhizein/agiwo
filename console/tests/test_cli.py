from pathlib import Path
from unittest.mock import patch

from server.cli import build_parser, main
from server.docker_runtime import ContainerUpOptions, DockerRuntimeError


def test_build_parser_supports_serve_subcommand() -> None:
    parser = build_parser()

    args = parser.parse_args(["serve"])

    assert args.command == "serve"
    assert args.host == "0.0.0.0"
    assert args.port == 8422
    assert args.env_file is None
    assert args.reload is False


def test_cli_serve_dispatches_to_uvicorn() -> None:
    with patch("server.cli.uvicorn.run") as run:
        exit_code = main(["serve", "--host", "127.0.0.1", "--port", "9999"])

    assert exit_code == 0
    run.assert_called_once_with(
        "server.app:app",
        host="127.0.0.1",
        port=9999,
        reload=False,
        env_file=None,
        factory=False,
    )


def test_cli_serve_forwards_reload_and_env_file() -> None:
    with patch("server.cli.uvicorn.run") as run:
        exit_code = main(
            [
                "serve",
                "--env-file",
                "/tmp/console.env",
                "--reload",
            ]
        )

    assert exit_code == 0
    run.assert_called_once_with(
        "server.app:app",
        host="0.0.0.0",
        port=8422,
        reload=True,
        env_file="/tmp/console.env",
        factory=False,
    )


def test_build_parser_supports_container_up_defaults() -> None:
    parser = build_parser()

    args = parser.parse_args(["container", "up", "--data-dir", "/tmp/data"])

    assert args.command == "container"
    assert args.container_command == "up"
    assert args.name == "agiwo-console"
    assert args.image == "agiwo-console:latest"
    assert args.data_dir == "/tmp/data"
    assert args.mount == []
    assert args.env == []
    assert args.env_file is None
    assert args.publish == "8422:8422"
    assert args.network_mode == "bridge"
    assert args.pull is False
    assert args.replace is False


def test_cli_container_up_dispatches_to_runtime() -> None:
    with patch("server.cli.container_up", return_value=0) as run:
        exit_code = main(
            [
                "container",
                "up",
                "--data-dir",
                "/tmp/data",
                "--mount",
                "/tmp/work:workspace",
                "--env",
                "OPENAI_API_KEY=test",
                "--replace",
            ]
        )

    assert exit_code == 0
    run.assert_called_once_with(
        ContainerUpOptions(
            name="agiwo-console",
            image="agiwo-console:latest",
            data_dir=Path("/tmp/data"),
            mount_specs=("/tmp/work:workspace",),
            env_file=None,
            env=("OPENAI_API_KEY=test",),
            publish="8422:8422",
            network_mode="bridge",
            pull=False,
            replace=True,
        )
    )


def test_cli_container_runtime_error_returns_failure(capsys) -> None:
    with patch("server.cli.container_status", side_effect=DockerRuntimeError("boom")):
        exit_code = main(["container", "status"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "boom" in captured.err
