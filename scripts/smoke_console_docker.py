import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from os import environ as os_environ
from os import getgid, getuid
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT_SECONDS = int(os.environ.get("AGIWO_CONSOLE_DOCKER_TIMEOUT_SECONDS", "900"))
HEALTH_TIMEOUT_SECONDS = 120
CONTAINER_PORT = 8422
LOOPBACK_PROXY_HOSTS = {"127.0.0.1", "::1", "localhost"}
PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def _proxy_hostname(value: str) -> str | None:
    parsed = urlsplit(value if "://" in value else f"//{value}")
    if parsed.hostname is not None:
        return parsed.hostname.lower()
    return None


def build_docker_env(
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os_environ if base_env is None else base_env)
    for key in PROXY_ENV_KEYS:
        value = env.get(key)
        if value is None:
            continue
        if _proxy_hostname(value) in LOOPBACK_PROXY_HOSTS:
            env.pop(key, None)
    return env


def docker_proxy_clear_build_args() -> list[str]:
    args: list[str] = []
    for key in PROXY_ENV_KEYS:
        args.extend(["--build-arg", f"{key}="])
    return args


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            timeout=TIMEOUT_SECONDS,
            text=True,
            capture_output=True,
            env=build_docker_env(env),
        )
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout, file=sys.stderr)
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
    except subprocess.TimeoutExpired as exc:
        print(
            "Command timed out after "
            f"{TIMEOUT_SECONDS}s: {' '.join(str(part) for part in cmd)}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def wait_for_http(url: str, *, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=3.0) as response:
                if getattr(response, "status", None) == 200:
                    return
                last_error = RuntimeError(
                    f"unexpected status {response.status} for {url}"
                )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(1)
    raise SystemExit(f"Timed out waiting for {url}: {last_error}")


def reserve_host_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def fix_volume_ownership(
    docker: str,
    image: str,
    *,
    data_dir: Path,
    workspace_dir: Path,
) -> None:
    subprocess.run(
        [
            docker,
            "run",
            "--rm",
            "-v",
            f"{data_dir}:/data",
            "-v",
            f"{workspace_dir}:/mnt/host/workspace",
            "--user",
            "root",
            "--entrypoint",
            "sh",
            image,
            "-c",
            f"chown -R {getuid()}:{getgid()} /data /mnt/host/workspace",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=build_docker_env(),
    )


def main() -> int:
    docker = shutil.which("docker")
    if docker is None:
        raise SystemExit("docker executable not found in PATH")

    configured_image = os.environ.get("AGIWO_CONSOLE_DOCKER_IMAGE", "").strip()
    image = configured_image or f"agiwo-console-smoke:{uuid4().hex[:8]}"
    build_image = configured_image == ""
    container = f"agiwo-console-smoke-{uuid4().hex[:8]}"
    host_port = reserve_host_port()

    tmp_path = Path(tempfile.mkdtemp(prefix="agiwo-console-smoke-"))
    data_dir = tmp_path / "data"
    workspace_dir = tmp_path / "workspace"
    data_dir.mkdir()
    workspace_dir.mkdir()

    try:
        if build_image:
            build_env = {**os.environ, "DOCKER_BUILDKIT": "1"}
            run(
                [
                    docker,
                    "build",
                    "--progress=plain",
                    *docker_proxy_clear_build_args(),
                    "-f",
                    "console/Dockerfile",
                    "-t",
                    image,
                    ".",
                ],
                cwd=ROOT,
                env=build_env,
            )
        run(
            [
                docker,
                "run",
                "-d",
                "--name",
                container,
                "--user",
                f"{getuid()}:{getgid()}",
                "-p",
                f"{host_port}:{CONTAINER_PORT}",
                "-v",
                f"{data_dir}:/data",
                "-v",
                f"{workspace_dir}:/mnt/host/workspace",
                image,
            ],
            cwd=ROOT,
        )

        wait_for_http(
            f"http://127.0.0.1:{host_port}/api/health",
            timeout_seconds=HEALTH_TIMEOUT_SECONDS,
        )
        wait_for_http(
            f"http://127.0.0.1:{host_port}/",
            timeout_seconds=HEALTH_TIMEOUT_SECONDS,
        )

        if not (data_dir / "root").is_dir():
            raise SystemExit("Expected data root to be created under mounted /data")

        run([docker, "exec", container, "test", "-d", "/mnt/host/workspace"])
    finally:
        logs = subprocess.run(
            [docker, "logs", container],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            env=build_docker_env(),
        )
        if logs.stdout:
            print(logs.stdout, file=sys.stderr, end="")
        if logs.stderr:
            print(logs.stderr, file=sys.stderr, end="")
        subprocess.run(
            [docker, "rm", "-f", container],
            check=False,
            env=build_docker_env(),
        )
        fix_volume_ownership(
            docker,
            image,
            data_dir=data_dir,
            workspace_dir=workspace_dir,
        )
        if build_image:
            subprocess.run(
                [docker, "rmi", image],
                check=False,
                env=build_docker_env(),
            )
        shutil.rmtree(tmp_path, ignore_errors=False)

    print("console docker smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
