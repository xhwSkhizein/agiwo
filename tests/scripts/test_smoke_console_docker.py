import subprocess

import scripts.smoke_console_docker as smoke_console_docker


def test_build_docker_env_strips_loopback_proxy_values() -> None:
    env = smoke_console_docker.build_docker_env(
        {
            "HTTP_PROXY": "http://127.0.0.1:7890",
            "HTTPS_PROXY": "https://localhost:8443",
            "ALL_PROXY": "socks5://[::1]:1080",
            "KEEP_ME": "value",
        }
    )

    assert "HTTP_PROXY" not in env
    assert "HTTPS_PROXY" not in env
    assert "ALL_PROXY" not in env
    assert env["KEEP_ME"] == "value"


def test_build_docker_env_keeps_non_loopback_proxy_values() -> None:
    env = smoke_console_docker.build_docker_env(
        {
            "http_proxy": "http://proxy.internal:8080",
            "https_proxy": "https://10.0.0.2:8443",
            "all_proxy": "socks5://corp-proxy:1080",
        }
    )

    assert env["http_proxy"] == "http://proxy.internal:8080"
    assert env["https_proxy"] == "https://10.0.0.2:8443"
    assert env["all_proxy"] == "socks5://corp-proxy:1080"


def test_docker_proxy_clear_build_args_blank_all_supported_proxy_variables() -> None:
    build_args = smoke_console_docker.docker_proxy_clear_build_args()

    assert "--build-arg" in build_args
    assert "HTTP_PROXY=" in build_args
    assert "http_proxy=" in build_args
    assert "HTTPS_PROXY=" in build_args
    assert "https_proxy=" in build_args


def test_build_installed_templates_check_code_checks_required_templates() -> None:
    smoke_code = smoke_console_docker.build_installed_templates_check_code()

    assert (
        "templates_dir = Path(agiwo.__file__).resolve().parent.parent / 'templates'"
        in smoke_code
    )
    assert "required = ('IDENTITY.md', 'SOUL.md', 'USER.md',);" in smoke_code
    assert "assert templates_dir.is_dir()" in smoke_code
    assert "assert not missing, missing" in smoke_code


def test_fix_volume_ownership_surfaces_docker_failure(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    def fake_run(*args, **kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args[0],
            returncode=42,
            stdout="docker stdout\n",
            stderr="docker stderr\n",
        )

    monkeypatch.setattr(smoke_console_docker.subprocess, "run", fake_run)

    smoke_console_docker.fix_volume_ownership(
        "docker",
        "image",
        data_dir=tmp_path / "data",
        workspace_dir=tmp_path / "workspace",
    )

    captured = capsys.readouterr()
    assert "docker exited with 42" in captured.err
    assert "docker stdout" in captured.err
    assert "docker stderr" in captured.err
