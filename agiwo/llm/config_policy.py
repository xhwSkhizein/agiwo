"""Shared model-configuration normalization and validation policy."""

from typing import Any
from urllib.parse import urlparse

from agiwo.config.settings import COMPATIBLE_MODEL_PROVIDERS


PARAM_API_KEY = "api_key"
PARAM_API_KEY_ENV_NAME = "api_key_env_name"
PARAM_BASE_URL = "base_url"


def normalize_model_param_string(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return value


def validate_model_base_url(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("base_url must start with http:// or https://")
    return value


def sanitize_model_params_data(
    data: Any,
    *,
    preserve_non_dict: bool = False,
    drop_null_keys: bool = True,
    reject_plain_api_key: bool = True,
) -> Any:
    if not isinstance(data, dict):
        return data if preserve_non_dict else {}

    normalized = dict(data)
    if PARAM_API_KEY in normalized:
        if reject_plain_api_key:
            raise ValueError(
                "api_key is not supported in model params; use api_key_env_name"
            )
        normalized.pop(PARAM_API_KEY, None)

    for key in (PARAM_BASE_URL, PARAM_API_KEY_ENV_NAME):
        if key not in normalized:
            continue
        cleaned = normalize_model_param_string(normalized.get(key))
        if cleaned is None:
            if drop_null_keys:
                normalized.pop(key, None)
            else:
                normalized[key] = None
            continue
        if key == PARAM_BASE_URL:
            cleaned = validate_model_base_url(cleaned)
        normalized[key] = cleaned

    return normalized


def validate_provider_model_params(
    provider: str | None,
    model_params: Any,
) -> None:
    if provider not in COMPATIBLE_MODEL_PROVIDERS:
        return

    if isinstance(model_params, dict):
        base_url = model_params.get(PARAM_BASE_URL)
        api_key_env_name = model_params.get(PARAM_API_KEY_ENV_NAME)
    else:
        base_url = getattr(model_params, PARAM_BASE_URL, None)
        api_key_env_name = getattr(model_params, PARAM_API_KEY_ENV_NAME, None)

    if base_url is None:
        raise ValueError(f"{provider} models require base_url")
    if api_key_env_name is None:
        raise ValueError(f"{provider} models require api_key_env_name")


__all__ = [
    "PARAM_API_KEY",
    "PARAM_API_KEY_ENV_NAME",
    "PARAM_BASE_URL",
    "normalize_model_param_string",
    "sanitize_model_params_data",
    "validate_model_base_url",
    "validate_provider_model_params",
]
