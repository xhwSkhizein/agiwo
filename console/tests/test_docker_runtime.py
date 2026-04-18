import subprocess
from pathlib import Path
from urllib.error import URLError

import pytest

from server.docker_runtime import (
    ContainerUpOptions,
    DockerRuntimeError,
    build_docker_run_command,
    container_up,
    ensure_supported_network_mode,
    parse_mount_spec,
    parse_mounts,
    resolve_healthcheck_url,
    resolve_container_user,
    wait_for_health,
)


class _FakeResponse:
    def __init__(self, status: int):
        self.status = status

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_parse_mount_spec_resolves_existing_source(tmp_path: Path) -> None:
    source = tmp_path / "workspace"
    source.mkdir()

    mount = parse_mount_spec(f"{source}:workspace")

    assert mount.source == source.resolve()
    assert mount.alias == "workspace"
    assert mount.target == "/mnt/host/workspace"


def test_parse_mounts_rejects_duplicate_aliases(tmp_path: Path) -> None:
    source = tmp_path / "workspace"
    source.mkdir()

    with pytest.raises(DockerRuntimeError, match="aliases must be unique"):
        parse_mounts(
            [
                f"{source}:shared",
                f"{source}:shared",
            ]
        )


def test_parse_mount_spec_rejects_relative_aliases(tmp_path: Path) -> None:
    source = tmp_path / "workspace"
    source.mkdir()

    with pytest.raises(DockerRuntimeError, match="relative path aliases"):
        parse_mount_spec(f"{source}:.")

    with pytest.raises(DockerRuntimeError, match="relative path aliases"):
        parse_mount_spec(f"{source}:..")


def test_build_docker_run_command_includes_defaults_and_mounts(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    mount_source = tmp_path / "project"
    mount_source.mkdir()
    mounts = parse_mounts([f"{mount_source}:project"])

    command = build_docker_run_command(
        ContainerUpOptions(
            name="console",
            image="example:latest",
            data_dir=data_dir,
            mount_specs=(f"{mount_source}:project",),
            env_file="/tmp/console.env",
            env=("OPENAI_API_KEY=test",),
            publish="9000:8422",
        ),
        mounts=mounts,
        container_user="1000:1000",
    )

    assert command[:10] == [
        "docker",
        "run",
        "-d",
        "--name",
        "console",
        "--restart",
        "unless-stopped",
        "--user",
        "1000:1000",
        "-p",
    ]
    assert "9000:8422" in command
    assert "--user" in command
    assert "1000:1000" in command
    assert f"{data_dir}:/data" in command
    assert f"{mount_source.resolve()}:/mnt/host/project" in command
    assert "AGIWO_ROOT_PATH=/data/root" in command
    assert "OPENAI_API_KEY=test" in command
    assert command[-1] == "example:latest"


def test_resolve_container_user_uses_host_uid_gid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("server.docker_runtime.os.getuid", lambda: 123)
    monkeypatch.setattr("server.docker_runtime.os.getgid", lambda: 456)

    assert resolve_container_user() == "123:456"


def test_wait_for_health_retries_until_success() -> None:
    attempts = {"count": 0}

    def opener(url: str, timeout: float) -> _FakeResponse:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise URLError("not ready")
        return _FakeResponse(200)

    wait_for_health(
        "http://127.0.0.1:8422/api/health",
        timeout_seconds=2.0,
        sleep=lambda _seconds: None,
        monotonic=iter([0.0, 0.1, 0.2]).__next__,
        opener=opener,
    )

    assert attempts["count"] == 2


def test_resolve_healthcheck_url_uses_host_port() -> None:
    assert (
        resolve_healthcheck_url("9000:8422", "bridge")
        == "http://127.0.0.1:9000/api/health"
    )
    assert (
        resolve_healthcheck_url("8422:8422", "host")
        == "http://127.0.0.1:8422/api/health"
    )


def test_resolve_healthcheck_url_rejects_publish_without_host_port() -> None:
    with pytest.raises(DockerRuntimeError, match="expected HOST:CONTAINER"):
        resolve_healthcheck_url("8422", "bridge")


def test_ensure_supported_network_mode_rejects_unknown_value() -> None:
    with pytest.raises(DockerRuntimeError, match="unsupported network mode"):
        ensure_supported_network_mode("weird")


def test_container_up_creates_data_dir_replaces_existing_container_and_waits_for_health(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    mount_source = tmp_path / "workspace"
    mount_source.mkdir()
    calls: list[list[str]] = []

    def runner(cmd, capture_output=False, text=False, check=False):
        calls.append(list(cmd))
        if cmd[1:] == ["info"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")
        if cmd[1:4] == ["container", "inspect", "console"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")
        if cmd[1:4] == ["rm", "-f", "console"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[1] == "run":
            return subprocess.CompletedProcess(cmd, 0, stdout="cid", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    options = ContainerUpOptions(
        name="console",
        image="example:latest",
        data_dir=data_dir,
        mount_specs=(f"{mount_source}:workspace",),
        replace=True,
        health_timeout_seconds=1.0,
    )
    monkeypatch.setattr(
        "server.docker_runtime.resolve_container_user",
        lambda: "1000:1000",
    )

    exit_code = container_up(
        options,
        docker_which=lambda _name: "/usr/bin/docker",
        runner=runner,
        sleep=lambda _seconds: None,
        monotonic=iter([0.0, 0.1]).__next__,
        opener=lambda _url, timeout: _FakeResponse(200),
    )

    assert exit_code == 0
    assert data_dir.is_dir()
    assert calls[0] == ["/usr/bin/docker", "info"]
    assert calls[1] == ["/usr/bin/docker", "container", "inspect", "console"]
    assert calls[2] == ["/usr/bin/docker", "rm", "-f", "console"]
    assert calls[3][:11] == [
        "/usr/bin/docker",
        "run",
        "-d",
        "--name",
        "console",
        "--restart",
        "unless-stopped",
        "--user",
        "1000:1000",
        "-p",
        "8422:8422",
    ]


def test_container_up_pulls_before_replacing_existing_container_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    calls: list[list[str]] = []

    def runner(cmd, capture_output=False, text=False, check=False):
        calls.append(list(cmd))
        if cmd[1:] == ["info"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")
        if cmd[1:4] == ["container", "inspect", "console"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")
        if cmd[1:3] == ["pull", "example:latest"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[1:4] == ["rm", "-f", "console"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[1] == "run":
            return subprocess.CompletedProcess(cmd, 0, stdout="cid", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(
        "server.docker_runtime.resolve_container_user",
        lambda: "1000:1000",
    )

    exit_code = container_up(
        ContainerUpOptions(
            name="console",
            image="example:latest",
            data_dir=data_dir,
            replace=True,
            pull=True,
            health_timeout_seconds=1.0,
        ),
        docker_which=lambda _name: "/usr/bin/docker",
        runner=runner,
        sleep=lambda _seconds: None,
        monotonic=iter([0.0, 0.1]).__next__,
        opener=lambda _url, timeout: _FakeResponse(200),
    )

    assert exit_code == 0
    assert calls[2] == ["/usr/bin/docker", "pull", "example:latest"]
    assert calls[3] == ["/usr/bin/docker", "rm", "-f", "console"]


def test_container_up_rejects_host_network_on_non_linux(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    monkeypatch.setattr("server.docker_runtime.sys.platform", "darwin")

    with pytest.raises(DockerRuntimeError, match="only supported"):
        container_up(
            ContainerUpOptions(
                data_dir=data_dir,
                network_mode="host",
            ),
            docker_which=lambda _name: "/usr/bin/docker",
            runner=lambda *args, **kwargs: subprocess.CompletedProcess(
                args[0], 0, "", ""
            ),
            sleep=lambda _seconds: None,
            monotonic=iter([0.0]).__next__,
            opener=lambda _url, timeout: _FakeResponse(200),
        )
