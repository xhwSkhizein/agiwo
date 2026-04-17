import shutil
import subprocess
import sys
import tempfile
import time
from os import getgid, getuid
from pathlib import Path
from urllib.request import urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT_SECONDS = 300
HEALTH_TIMEOUT_SECONDS = 120
HOST_PORT = 18422


def run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            timeout=TIMEOUT_SECONDS,
            text=True,
            capture_output=True,
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
            "--entrypoint",
            "sh",
            image,
            "-c",
            f"chown -R {getuid()}:{getgid()} /data /mnt/host/workspace",
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> int:
    docker = shutil.which("docker")
    if docker is None:
        raise SystemExit("docker executable not found in PATH")

    image = f"agiwo-console-smoke:{uuid4().hex[:8]}"
    container = f"agiwo-console-smoke-{uuid4().hex[:8]}"

    tmp_path = Path(tempfile.mkdtemp(prefix="agiwo-console-smoke-"))
    data_dir = tmp_path / "data"
    workspace_dir = tmp_path / "workspace"
    data_dir.mkdir()
    workspace_dir.mkdir()

    try:
        run(
            [docker, "build", "-f", "console/Dockerfile", "-t", image, "."],
            cwd=ROOT,
        )
        run(
            [
                docker,
                "run",
                "-d",
                "--name",
                container,
                "-p",
                f"{HOST_PORT}:8422",
                "-v",
                f"{data_dir}:/data",
                "-v",
                f"{workspace_dir}:/mnt/host/workspace",
                image,
            ],
            cwd=ROOT,
        )

        wait_for_http(
            f"http://127.0.0.1:{HOST_PORT}/api/health",
            timeout_seconds=HEALTH_TIMEOUT_SECONDS,
        )
        wait_for_http(
            f"http://127.0.0.1:{HOST_PORT}/",
            timeout_seconds=HEALTH_TIMEOUT_SECONDS,
        )

        if not (data_dir / "root").is_dir():
            raise SystemExit("Expected data root to be created under mounted /data")

        run([docker, "exec", container, "test", "-d", "/mnt/host/workspace"])
    finally:
        subprocess.run(
            [docker, "logs", container],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        subprocess.run([docker, "rm", "-f", container], check=False)
        fix_volume_ownership(
            docker,
            image,
            data_dir=data_dir,
            workspace_dir=workspace_dir,
        )
        subprocess.run([docker, "rmi", image], check=False)
        shutil.rmtree(tmp_path, ignore_errors=False)

    print("console docker smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
