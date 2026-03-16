"""Factory helpers for creating Model instances from configuration."""

from dataclasses import dataclass
import os
from typing import Any, Callable
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict

from agiwo.llm.anthropic import AnthropicModel
from agiwo.llm.base import Model
from agiwo.llm.bedrock_anthropic import BedrockAnthropicModel
from agiwo.llm.config_policy import (
    sanitize_model_params_data,
    validate_provider_model_params,
)
from agiwo.llm.deepseek import DeepseekModel
from agiwo.llm.nvidia import NvidiaModel
from agiwo.llm.openai import OpenAIModel
from agiwo.config.settings import (
    ALL_MODEL_PROVIDERS,
    COMPATIBLE_MODEL_PROVIDERS,
    ModelProvider,
)


class ModelConfig(BaseModel):
    """Serializable model construction config."""
    model_config = ConfigDict(extra="ignore")

    provider: ModelProvider
    model_name: str
    api_key_env_name: str | None = None
    base_url: str | None = None
    temperature: float = 0.7
    top_p: float = 1.0
    max_output_tokens: int = 4096
    max_context_window: int = 200000
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
        "max_output_tokens": config.max_output_tokens,
        "max_context_window": config.max_context_window,
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
    validate_provider_model_params(provider, config)
    if provider in COMPATIBLE_MODEL_PROVIDERS and not resolved_api_key:
        raise ValueError(
            f"{provider} api_key_env_name '{config.api_key_env_name}' is not set"
        )

    if provider in COMPATIBLE_MODEL_PROVIDERS:
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
        disable_env_fallback=True,
        override_provider="openai-compatible",
    ),
    "deepseek": ProviderSpec(model_class=DeepseekModel),
    "anthropic": ProviderSpec(model_class=AnthropicModel),
    "anthropic-compatible": ProviderSpec(
        model_class=AnthropicModel,
        disable_env_fallback=True,
        override_provider="anthropic-compatible",
    ),
    "nvidia": ProviderSpec(model_class=NvidiaModel),
    "bedrock-anthropic": ProviderSpec(
        model_class=BedrockAnthropicModel,
        include_aws_config=True,
    ),
}

_missing_provider_specs = set(ALL_MODEL_PROVIDERS) - set(PROVIDER_SPECS)
if _missing_provider_specs:
    raise RuntimeError(
        f"Missing provider specs for: {sorted(_missing_provider_specs)}"
    )


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


MODEL_CONFIG_FIELD_NAMES = set(ModelConfig.model_fields.keys())


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
    normalized = sanitize_model_params_data(dict(params or {}))
    valid_params = _filter_model_config_fields(normalized)
    model_config = ModelConfig(
        provider=provider,  # type: ignore[arg-type]
        model_name=model_name,
        **valid_params,
    )
    return create_model(model_config)


__all__ = ["ModelProvider", "ModelConfig", "create_model", "create_model_from_dict"]
