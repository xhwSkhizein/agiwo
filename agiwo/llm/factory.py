"""Factory helpers for creating Model instances from configuration."""

from dataclasses import dataclass, fields
import os
from typing import Any, Callable, Literal
from urllib.parse import urlparse

from agiwo.llm.anthropic import AnthropicModel
from agiwo.llm.base import Model
from agiwo.llm.bedrock_anthropic import BedrockAnthropicModel
from agiwo.llm.deepseek import DeepseekModel
from agiwo.llm.nvidia import NvidiaModel
from agiwo.llm.openai import OpenAIModel

ModelProvider = Literal[
    "openai",
    "openai-compatible",
    "deepseek",
    "anthropic",
    "anthropic-compatible",
    "nvidia",
    "bedrock-anthropic",
]


PARAM_API_KEY = "api_key"
PARAM_API_KEY_ENV_NAME = "api_key_env_name"
PARAM_BASE_URL = "base_url"
PARAM_MAX_OUTPUT_TOKENS_PER_CALL = "max_output_tokens_per_call"
PARAM_MAX_TOKENS = "max_tokens"


@dataclass
class ModelConfig:
    """Serializable model construction config."""

    provider: ModelProvider
    model_name: str
    api_key_env_name: str | None = None
    base_url: str | None = None
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 4096
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    cache_hit_price: float = 0.0
    input_price: float = 0.0
    output_price: float = 0.0
    aws_region: str | None = None
    aws_profile: str | None = None


@dataclass(frozen=True)
class ProviderSpec:
    model_class: Callable[..., Model]
    requires_explicit_base_url: bool = False
    requires_api_key_env_name: bool = False
    disable_env_fallback: bool = False
    override_provider: ModelProvider | None = None
    include_aws_config: bool = False


def _resolve_api_key(config: ModelConfig) -> str | None:
    if config.api_key_env_name:
        return os.getenv(config.api_key_env_name)
    return None


def _build_shared_params(config: ModelConfig) -> dict[str, Any]:
    return {
        "id": config.model_name,
        "name": config.model_name,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
        "frequency_penalty": config.frequency_penalty,
        "presence_penalty": config.presence_penalty,
        "cache_hit_price": config.cache_hit_price,
        "input_price": config.input_price,
        "output_price": config.output_price,
    }


def _require_absolute_base_url(provider: str, base_url: str | None) -> str:
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError(f"{provider} models require an explicit base_url")
    normalized = base_url.strip()
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(
            f"{provider} base_url must start with http:// or https://"
        )
    return normalized


def _build_model_for_provider(
    provider: ModelProvider,
    spec: ProviderSpec,
    config: ModelConfig,
    resolved_api_key: str | None,
    shared_params: dict[str, Any],
) -> Model:
    if spec.requires_api_key_env_name and not config.api_key_env_name:
        raise ValueError(f"{provider} models require api_key_env_name")
    if spec.requires_api_key_env_name and not resolved_api_key:
        raise ValueError(
            f"{provider} api_key_env_name '{config.api_key_env_name}' is not set"
        )

    if spec.requires_explicit_base_url:
        base_url = _require_absolute_base_url(provider, config.base_url)
    else:
        base_url = config.base_url

    model_kwargs: dict[str, Any] = {
        **shared_params,
        "api_key": resolved_api_key,
        "base_url": base_url,
    }
    if spec.disable_env_fallback:
        model_kwargs["allow_env_fallback"] = False
    if spec.include_aws_config:
        model_kwargs["aws_region"] = config.aws_region
        model_kwargs["aws_profile"] = config.aws_profile

    model = spec.model_class(**model_kwargs)
    if spec.override_provider is not None:
        model.provider = spec.override_provider
    return model


PROVIDER_SPECS: dict[ModelProvider, ProviderSpec] = {
    "openai": ProviderSpec(model_class=OpenAIModel),
    "openai-compatible": ProviderSpec(
        model_class=OpenAIModel,
        requires_explicit_base_url=True,
        requires_api_key_env_name=True,
        disable_env_fallback=True,
        override_provider="openai-compatible",
    ),
    "deepseek": ProviderSpec(model_class=DeepseekModel),
    "anthropic": ProviderSpec(model_class=AnthropicModel),
    "anthropic-compatible": ProviderSpec(
        model_class=AnthropicModel,
        requires_explicit_base_url=True,
        requires_api_key_env_name=True,
        disable_env_fallback=True,
        override_provider="anthropic-compatible",
    ),
    "nvidia": ProviderSpec(model_class=NvidiaModel),
    "bedrock-anthropic": ProviderSpec(
        model_class=BedrockAnthropicModel,
        include_aws_config=True,
    ),
}


def create_model(config: ModelConfig) -> Model:
    """Create a concrete Model from config."""
    provider = config.provider
    resolved_api_key = _resolve_api_key(config)
    shared_params = _build_shared_params(config)
    provider_spec = PROVIDER_SPECS.get(provider)
    if provider_spec is None:
        raise ValueError(f"Unknown model provider: {provider}")
    return _build_model_for_provider(
        provider=provider,
        spec=provider_spec,
        config=config,
        resolved_api_key=resolved_api_key,
        shared_params=shared_params,
    )


def _normalize_model_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(params)
    if PARAM_API_KEY in normalized:
        raise ValueError(
            "api_key is not supported in model params; use api_key_env_name"
        )
    for key in (PARAM_BASE_URL, PARAM_API_KEY_ENV_NAME):
        value = normalized.get(key)
        if isinstance(value, str):
            normalized[key] = value.strip() or None

    max_output_tokens = normalized.pop(PARAM_MAX_OUTPUT_TOKENS_PER_CALL, None)
    if PARAM_MAX_TOKENS not in normalized and max_output_tokens is not None:
        normalized[PARAM_MAX_TOKENS] = max_output_tokens
    return normalized


MODEL_CONFIG_FIELD_NAMES = {field.name for field in fields(ModelConfig)}


def _filter_model_config_fields(params: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in params.items() if key in MODEL_CONFIG_FIELD_NAMES
    }


def create_model_from_dict(
    *,
    provider: str,
    model_name: str,
    params: dict[str, Any] | None = None,
) -> Model:
    """Create a concrete Model from provider/name plus loose params."""
    normalized = _normalize_model_params(dict(params or {}))
    valid_params = _filter_model_config_fields(normalized)
    model_config = ModelConfig(
        provider=provider,  # type: ignore[arg-type]
        model_name=model_name,
        **valid_params,
    )
    return create_model(model_config)


__all__ = ["ModelProvider", "ModelConfig", "create_model", "create_model_from_dict"]
