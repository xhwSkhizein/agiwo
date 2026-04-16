import pytest

from agiwo.llm import ModelSpec, create_model, create_model_from_dict
from agiwo.llm.anthropic import AnthropicModel
from agiwo.llm.deepseek import DeepseekModel
from agiwo.llm.openai import OpenAIModel
from agiwo.llm.openai_response import OpenAIResponsesModel


def test_create_model_builds_provider_specific_instance() -> None:
    model = create_model(
        ModelSpec(
            provider="anthropic",
            model_name="claude-3-5-sonnet-20240620",
            temperature=0.1,
            max_output_tokens=512,
        )
    )

    assert isinstance(model, AnthropicModel)
    assert model.name == "claude-3-5-sonnet-20240620"
    assert model.temperature == 0.1
    assert model.max_output_tokens == 512


def test_create_model_builds_openai_response_instance(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    model = create_model(
        ModelSpec(
            provider="openai-response",
            model_name="gpt-4.1-mini",
            temperature=0.2,
            max_output_tokens=256,
        )
    )

    assert isinstance(model, OpenAIResponsesModel)
    assert model.provider == "openai-response"
    assert model.temperature == 0.2
    assert model.max_output_tokens == 256


def test_create_model_from_dict_uses_max_output_tokens(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    model = create_model_from_dict(
        provider="deepseek",
        model_name="deepseek-chat",
        params={
            "max_output_tokens": 321,
            "temperature": 0.05,
            "api_key_env_name": "DEEPSEEK_API_KEY",
        },
    )

    assert isinstance(model, DeepseekModel)
    assert model.max_output_tokens == 321
    assert model.temperature == 0.05


def test_create_model_from_dict_resolves_openai_compatible_api_key_env(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MINIMAX_API_KEY", "minimax-key")

    model = create_model_from_dict(
        provider="openai-compatible",
        model_name="MiniMax-M2.5",
        params={
            "base_url": "https://api.minimax.chat/v1",
            "api_key_env_name": "MINIMAX_API_KEY",
        },
    )

    assert isinstance(model, OpenAIModel)
    assert model.provider == "openai-compatible"
    assert model.api_key == "minimax-key"
    assert model.base_url == "https://api.minimax.chat/v1"
    assert model.allow_env_fallback is False


def test_create_model_from_dict_builds_openai_response_instance(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    model = create_model_from_dict(
        provider="openai-response",
        model_name="gpt-4.1-mini",
        params={"temperature": 0.15, "max_output_tokens": 111},
    )

    assert isinstance(model, OpenAIResponsesModel)
    assert model.provider == "openai-response"
    assert model.temperature == 0.15
    assert model.max_output_tokens == 111


def test_create_model_from_dict_rejects_openai_compatible_without_env_name(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "shared-openai-key")

    with pytest.raises(ValueError, match="api_key"):
        create_model_from_dict(
            provider="openai-compatible",
            model_name="MiniMax-M2.5",
            params={"base_url": "https://api.minimax.chat/v1"},
        )


def test_create_model_from_dict_rejects_openai_compatible_without_base_url(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MINIMAX_API_KEY", "minimax-key")

    with pytest.raises(ValueError, match="base_url"):
        create_model_from_dict(
            provider="openai-compatible",
            model_name="MiniMax-M2.5",
            params={"api_key_env_name": "MINIMAX_API_KEY"},
        )


def test_create_model_from_dict_resolves_anthropic_compatible_api_key_env(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_COMPAT_API_KEY", "compat-key")

    model = create_model_from_dict(
        provider="anthropic-compatible",
        model_name="claude-compatible",
        params={
            "base_url": "https://anthropic-proxy.example.com/v1",
            "api_key_env_name": "ANTHROPIC_COMPAT_API_KEY",
        },
    )

    assert isinstance(model, AnthropicModel)
    assert model.provider == "anthropic-compatible"
    assert model.api_key == "compat-key"
    assert model.base_url == "https://anthropic-proxy.example.com/v1"
    assert model.allow_env_fallback is False


def test_create_model_from_dict_rejects_plain_api_key_param() -> None:
    with pytest.raises(ValueError, match="api_key is not supported"):
        create_model_from_dict(
            provider="openai-compatible",
            model_name="MiniMax-M2.5",
            params={
                "base_url": "https://api.minimax.chat/v1",
                "api_key": "sk-plain-text",
            },
        )
