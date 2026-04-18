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
