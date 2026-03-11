"""LLM limit policy helpers."""

from agiwo.llm.base import Model


DEFAULT_MAX_CONTEXT_WINDOW = 200000
DEFAULT_MAX_OUTPUT_TOKENS = 4096


def resolve_max_context_window(model: Model) -> int:
    value = int(getattr(model, "max_context_window", DEFAULT_MAX_CONTEXT_WINDOW) or 0)
    if value < 1:
        return DEFAULT_MAX_CONTEXT_WINDOW
    return value


def resolve_max_output_tokens(model: Model) -> int:
    value = int(getattr(model, "max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS) or 0)
    if value < 1:
        return DEFAULT_MAX_OUTPUT_TOKENS
    return value


def resolve_max_input_tokens_per_call(
    configured_limit: int | None,
    model: Model,
) -> int:
    if configured_limit is not None:
        return configured_limit

    context_window = resolve_max_context_window(model)
    max_output_tokens = resolve_max_output_tokens(model)
    default_limit = context_window - max_output_tokens
    if default_limit <= 0:
        raise ValueError(
            "max_context_window must be greater than max_output_tokens when "
            "max_input_tokens_per_call is not configured"
        )
    return default_limit


__all__ = [
    "DEFAULT_MAX_CONTEXT_WINDOW",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "resolve_max_context_window",
    "resolve_max_input_tokens_per_call",
    "resolve_max_output_tokens",
]
