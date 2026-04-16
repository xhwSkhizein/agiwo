from unittest.mock import patch

from server.cli import build_parser, main


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
