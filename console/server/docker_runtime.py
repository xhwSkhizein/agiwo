import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen


DEFAULT_CONTAINER_NAME = "agiwo-console"
DEFAULT_IMAGE = "agiwo-console:latest"
DEFAULT_PUBLISH = "8422:8422"
DEFAULT_HEALTH_TIMEOUT_SECONDS = 30.0
_ALIAS_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


class DockerRuntimeError(RuntimeError):
    """Raised when the managed Docker lifecycle cannot proceed."""


@dataclass(frozen=True)
class DockerMount:
    source: Path
    alias: str

    @property
    def target(self) -> str:
        return f"/mnt/host/{self.alias}"


@dataclass(frozen=True)
class ContainerUpOptions:
    name: str = DEFAULT_CONTAINER_NAME
    image: str = DEFAULT_IMAGE
    data_dir: Path = Path(".agiwo-console-docker")
    mount_specs: tuple[str, ...] = ()
    env_file: str | None = None
    env: tuple[str, ...] = ()
    publish: str = DEFAULT_PUBLISH
    network_mode: str = "bridge"
    pull: bool = False
    replace: bool = False
    health_timeout_seconds: float = DEFAULT_HEALTH_TIMEOUT_SECONDS


RunCommand = Callable[..., subprocess.CompletedProcess[str]]
SleepFn = Callable[[float], None]
MonotonicFn = Callable[[], float]
UrlOpenFn = Callable[..., object]


def _run_capture(
    cmd: Sequence[str],
    *,
    runner: RunCommand = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    return runner(cmd, capture_output=True, text=True, check=False)


def resolve_docker_binary(
    *,
    docker_which: Callable[[str], str | None] = shutil.which,
) -> str:
    docker_bin = docker_which("docker")
    if not docker_bin:
        raise DockerRuntimeError("docker executable not found in PATH")
    return docker_bin


def ensure_docker_access(
    *,
    docker_bin: str,
    runner: RunCommand = subprocess.run,
) -> None:
    info = _run_capture([docker_bin, "info"], runner=runner)
    if info.returncode != 0:
        detail = (info.stderr or info.stdout or "").strip() or "unknown docker error"
        raise DockerRuntimeError(f"docker daemon is unavailable: {detail}")


def ensure_supported_network_mode(network_mode: str) -> None:
    if network_mode != "host":
        return
    if sys.platform != "linux":
        raise DockerRuntimeError(
            "--network-mode=host is only supported as a first-class option on Linux"
        )


def ensure_data_dir(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    if not data_dir.is_dir():
        raise DockerRuntimeError(f"data directory is not a directory: {data_dir}")
    if not os.access(data_dir, os.W_OK):
        raise DockerRuntimeError(f"data directory is not writable: {data_dir}")
    return data_dir.resolve()


def parse_mount_spec(spec: str) -> DockerMount:
    source_text, sep, alias = spec.rpartition(":")
    if not sep or not source_text or not alias:
        raise DockerRuntimeError(f"invalid mount {spec!r}; expected <source>:<alias>")
    if not _ALIAS_PATTERN.fullmatch(alias):
        raise DockerRuntimeError(
            f"invalid mount alias {alias!r}; expected pattern {_ALIAS_PATTERN.pattern}"
        )
    source = Path(source_text).expanduser()
    if not source.exists():
        raise DockerRuntimeError(f"mount source does not exist: {source}")
    return DockerMount(source=source.resolve(), alias=alias)


def parse_mounts(specs: Sequence[str]) -> tuple[DockerMount, ...]:
    mounts = [parse_mount_spec(spec) for spec in specs]
    aliases = [mount.alias for mount in mounts]
    if len(aliases) != len(set(aliases)):
        raise DockerRuntimeError("mount aliases must be unique")
    return tuple(mounts)


def _container_exists(
    *,
    docker_bin: str,
    name: str,
    runner: RunCommand = subprocess.run,
) -> bool:
    inspect = _run_capture(
        [docker_bin, "container", "inspect", name],
        runner=runner,
    )
    return inspect.returncode == 0


def _tail_logs(
    *,
    docker_bin: str,
    name: str,
    runner: RunCommand = subprocess.run,
    tail: int = 50,
) -> str:
    logs = _run_capture(
        [docker_bin, "logs", "--tail", str(tail), name],
        runner=runner,
    )
    return (logs.stderr or logs.stdout or "").strip()


def resolve_healthcheck_url(publish: str, network_mode: str) -> str:
    if network_mode == "host":
        return "http://127.0.0.1:8422/api/health"
    host_port = publish.split(":", maxsplit=1)[0]
    if not host_port.isdigit():
        raise DockerRuntimeError(
            f"invalid publish value {publish!r}; expected HOST:CONTAINER"
        )
    return f"http://127.0.0.1:{host_port}/api/health"


def wait_for_health(
    url: str,
    *,
    timeout_seconds: float,
    sleep: SleepFn = time.sleep,
    monotonic: MonotonicFn = time.monotonic,
    opener: UrlOpenFn = urlopen,
) -> None:
    deadline = monotonic() + timeout_seconds
    last_error: Exception | None = None
    while monotonic() < deadline:
        try:
            with opener(url, timeout=2.0) as response:  # type: ignore[misc]
                status = getattr(response, "status", None)
                if status == 200:
                    return
                last_error = DockerRuntimeError(
                    f"unexpected health status {status} from {url}"
                )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        sleep(1.0)
    if last_error is None:
        raise DockerRuntimeError(f"health check timed out for {url}")
    raise DockerRuntimeError(f"health check failed for {url}: {last_error}")


def build_docker_run_command(
    options: ContainerUpOptions,
    *,
    mounts: Sequence[DockerMount],
) -> list[str]:
    command = [
        "docker",
        "run",
        "-d",
        "--name",
        options.name,
        "--restart",
        "unless-stopped",
    ]
    if options.network_mode == "host":
        command.extend(["--network", "host"])
    else:
        command.extend(["-p", options.publish])
    command.extend(["-v", f"{options.data_dir}:/data"])
    for mount in mounts:
        command.extend(["-v", f"{mount.source}:{mount.target}"])
    if options.env_file:
        command.extend(["--env-file", options.env_file])
    command.extend(["-e", "AGIWO_ROOT_PATH=/data/root"])
    for item in options.env:
        command.extend(["-e", item])
    command.append(options.image)
    return command


def container_up(
    options: ContainerUpOptions,
    *,
    docker_which: Callable[[str], str | None] = shutil.which,
    runner: RunCommand = subprocess.run,
    sleep: SleepFn = time.sleep,
    monotonic: MonotonicFn = time.monotonic,
    opener: UrlOpenFn = urlopen,
) -> int:
    docker_bin = resolve_docker_binary(docker_which=docker_which)
    ensure_docker_access(docker_bin=docker_bin, runner=runner)
    ensure_supported_network_mode(options.network_mode)

    resolved_data_dir = ensure_data_dir(options.data_dir)
    mounts = parse_mounts(options.mount_specs)
    normalized = ContainerUpOptions(
        name=options.name,
        image=options.image,
        data_dir=resolved_data_dir,
        mount_specs=options.mount_specs,
        env_file=options.env_file,
        env=options.env,
        publish=options.publish,
        network_mode=options.network_mode,
        pull=options.pull,
        replace=options.replace,
        health_timeout_seconds=options.health_timeout_seconds,
    )

    if _container_exists(docker_bin=docker_bin, name=normalized.name, runner=runner):
        if not normalized.replace:
            raise DockerRuntimeError(
                f"container {normalized.name!r} already exists; use --replace to recreate it"
            )
        replace = _run_capture(
            [docker_bin, "rm", "-f", normalized.name],
            runner=runner,
        )
        if replace.returncode != 0:
            detail = (
                replace.stderr or replace.stdout or ""
            ).strip() or "docker rm failed"
            raise DockerRuntimeError(detail)

    if normalized.pull:
        pull = _run_capture([docker_bin, "pull", normalized.image], runner=runner)
        if pull.returncode != 0:
            detail = (pull.stderr or pull.stdout or "").strip() or "docker pull failed"
            raise DockerRuntimeError(detail)

    command = build_docker_run_command(normalized, mounts=mounts)
    command[0] = docker_bin
    run = _run_capture(command, runner=runner)
    if run.returncode != 0:
        detail = (run.stderr or run.stdout or "").strip() or "docker run failed"
        raise DockerRuntimeError(detail)

    health_url = resolve_healthcheck_url(normalized.publish, normalized.network_mode)
    try:
        wait_for_health(
            health_url,
            timeout_seconds=normalized.health_timeout_seconds,
            sleep=sleep,
            monotonic=monotonic,
            opener=opener,
        )
    except DockerRuntimeError as exc:
        logs = _tail_logs(docker_bin=docker_bin, name=normalized.name, runner=runner)
        if logs:
            raise DockerRuntimeError(f"{exc}\nRecent container logs:\n{logs}") from exc
        raise
    return 0


def container_down(
    name: str,
    *,
    docker_which: Callable[[str], str | None] = shutil.which,
    runner: RunCommand = subprocess.run,
) -> int:
    docker_bin = resolve_docker_binary(docker_which=docker_which)
    ensure_docker_access(docker_bin=docker_bin, runner=runner)
    command = _run_capture([docker_bin, "rm", "-f", name], runner=runner)
    if command.returncode != 0:
        detail = (command.stderr or command.stdout or "").strip()
        raise DockerRuntimeError(detail or f"failed to remove container {name!r}")
    return 0


def container_logs(
    name: str,
    *,
    docker_which: Callable[[str], str | None] = shutil.which,
    runner: RunCommand = subprocess.run,
) -> int:
    docker_bin = resolve_docker_binary(docker_which=docker_which)
    ensure_docker_access(docker_bin=docker_bin, runner=runner)
    command = runner([docker_bin, "logs", name], text=True, check=False)
    if command.returncode != 0:
        raise DockerRuntimeError(f"failed to read logs for container {name!r}")
    return 0


def container_status(
    name: str,
    *,
    docker_which: Callable[[str], str | None] = shutil.which,
    runner: RunCommand = subprocess.run,
) -> int:
    docker_bin = resolve_docker_binary(docker_which=docker_which)
    ensure_docker_access(docker_bin=docker_bin, runner=runner)
    command = _run_capture(
        [
            docker_bin,
            "ps",
            "-a",
            "--filter",
            f"name=^{name}$",
            "--format",
            "{{.Names}}\t{{.State}}\t{{.Status}}",
        ],
        runner=runner,
    )
    if command.returncode != 0:
        detail = (command.stderr or command.stdout or "").strip()
        raise DockerRuntimeError(detail or f"failed to inspect container {name!r}")
    output = command.stdout.strip()
    if not output:
        print(f"container {name!r} not found")
        return 1
    print(output)
    return 0


def container_restart(
    name: str,
    *,
    publish: str = DEFAULT_PUBLISH,
    network_mode: str = "bridge",
    health_timeout_seconds: float = DEFAULT_HEALTH_TIMEOUT_SECONDS,
    docker_which: Callable[[str], str | None] = shutil.which,
    runner: RunCommand = subprocess.run,
    sleep: SleepFn = time.sleep,
    monotonic: MonotonicFn = time.monotonic,
    opener: UrlOpenFn = urlopen,
) -> int:
    docker_bin = resolve_docker_binary(docker_which=docker_which)
    ensure_docker_access(docker_bin=docker_bin, runner=runner)
    ensure_supported_network_mode(network_mode)
    command = _run_capture([docker_bin, "restart", name], runner=runner)
    if command.returncode != 0:
        detail = (command.stderr or command.stdout or "").strip()
        raise DockerRuntimeError(detail or f"failed to restart container {name!r}")
    wait_for_health(
        resolve_healthcheck_url(publish, network_mode),
        timeout_seconds=health_timeout_seconds,
        sleep=sleep,
        monotonic=monotonic,
        opener=opener,
    )
    return 0
