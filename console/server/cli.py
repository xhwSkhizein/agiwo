import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from server.docker_runtime import (
    DEFAULT_CONTAINER_NAME,
    DEFAULT_IMAGE,
    DEFAULT_PUBLISH,
    DockerRuntimeError,
    container_down,
    container_logs,
    container_restart,
    container_status,
    container_up,
    ContainerUpOptions,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agiwo-console")
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="Start the Agiwo Console server")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8422)
    serve.add_argument("--env-file", default=None)
    serve.add_argument("--reload", action="store_true")

    container = subparsers.add_parser(
        "container",
        help="Manage the Agiwo Console Docker container",
    )
    container_subparsers = container.add_subparsers(dest="container_command")

    up = container_subparsers.add_parser("up", help="Start the managed container")
    up.add_argument("--name", default=DEFAULT_CONTAINER_NAME)
    up.add_argument("--image", default=DEFAULT_IMAGE)
    up.add_argument("--data-dir", required=True)
    up.add_argument("--mount", action="append", default=[])
    up.add_argument("--env-file", default=None)
    up.add_argument("--env", action="append", default=[])
    up.add_argument("--publish", default=DEFAULT_PUBLISH)
    up.add_argument("--network-mode", choices=["bridge", "host"], default="bridge")
    up.add_argument("--pull", action="store_true")
    up.add_argument("--replace", action="store_true")

    down = container_subparsers.add_parser("down", help="Stop and remove the container")
    down.add_argument("--name", default=DEFAULT_CONTAINER_NAME)

    status = container_subparsers.add_parser("status", help="Show container status")
    status.add_argument("--name", default=DEFAULT_CONTAINER_NAME)

    logs = container_subparsers.add_parser("logs", help="Show container logs")
    logs.add_argument("--name", default=DEFAULT_CONTAINER_NAME)

    restart = container_subparsers.add_parser("restart", help="Restart the container")
    restart.add_argument("--name", default=DEFAULT_CONTAINER_NAME)
    restart.add_argument("--publish", default=DEFAULT_PUBLISH)
    restart.add_argument("--network-mode", choices=["bridge", "host"], default="bridge")

    return parser


def _run_container_command(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> int:
    try:
        if args.container_command == "up":
            result = container_up(
                ContainerUpOptions(
                    name=args.name,
                    image=args.image,
                    data_dir=Path(args.data_dir),
                    mount_specs=tuple(args.mount),
                    env_file=args.env_file,
                    env=tuple(args.env),
                    publish=args.publish,
                    network_mode=args.network_mode,
                    pull=args.pull,
                    replace=args.replace,
                )
            )
        else:
            handlers = {
                "down": lambda: container_down(args.name),
                "status": lambda: container_status(args.name),
                "logs": lambda: container_logs(args.name),
                "restart": lambda: container_restart(
                    args.name,
                    publish=args.publish,
                    network_mode=args.network_mode,
                ),
            }
            handler = handlers.get(args.container_command)
            if handler is None:
                parser.print_help()
                return 0
            result = handler()
    except DockerRuntimeError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 1
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "serve":
        uvicorn.run(
            "server.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            env_file=args.env_file,
            factory=False,
        )
        return 0

    if args.command == "container":
        return _run_container_command(args, parser)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
