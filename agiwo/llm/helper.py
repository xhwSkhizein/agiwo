import json
import ast
from typing import Any


def _get_val(obj: Any, key: str) -> Any:
    """Get value from dict or object attribute safely."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def parse_json_tool_args(args: Any) -> dict[str, Any]:
    """
    Parse tool arguments from various formats (dict, JSON string, python-dict string) into a dictionary.
    
    Handles:
    - Already a dict
    - JSON string
    - Python literal string (fallback)
    - Empty or None inputs
    """
    if isinstance(args, dict):
        return args

    if not args or not isinstance(args, str):
        return {}

    # Try standard JSON
    try:
        return json.loads(args)
    except json.JSONDecodeError:
        pass

    # Try Python literal eval (fallback for weak models that output Python dicts)
    try:
        # Basic sanity check to avoid eval on dangerous strings
        if args.strip().startswith("{") and args.strip().endswith("}"):
            val = ast.literal_eval(args)
            if isinstance(val, dict):
                return val
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        pass

    # Failed to parse, return as raw argument for debugging/feedback
    return {"__raw_arguments__": args}


def normalize_usage_metrics(usage_data: Any) -> dict[str, int | None]:
    """
    Normalize model usage metrics to unified format.

    Handles both OpenAI-style and unified-style metrics, including prompt caching.
    Ensures 'input_tokens' always represents the TOTAL input (including cache).

    Args:
        usage_data: Raw usage data from model (dict or object)

    Returns:
        dict with normalized keys:
        - input_tokens (Total input)
        - output_tokens
        - total_tokens (input_tokens + output_tokens)
        - cache_read_tokens
        - cache_creation_tokens
    """
    if not usage_data:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "cache_read_tokens": None,
            "cache_creation_tokens": None,
        }

    # Helper to get value from usage_data
    def get(key: str) -> int | None:
        return _get_val(usage_data, key)

    # Extract base tokens
    # OpenAI: prompt_tokens (Total)
    # Anthropic: input_tokens (Excludes cache)
    base_input = get("input_tokens")
    if base_input is None:
        base_input = get("prompt_tokens")

    output_tokens = get("output_tokens")
    if output_tokens is None:
        output_tokens = get("completion_tokens")

    # Extract cache details
    # OpenAI: cached_tokens
    # Anthropic: cache_read_tokens, cache_creation_tokens
    cache_read_tokens = get("cache_read_tokens")
    if cache_read_tokens is None:
        cache_read_tokens = get("cache_read_input_tokens")
    if cache_read_tokens is None:
        cache_read_tokens = get("cached_tokens")

    cache_creation_tokens = get("cache_creation_tokens")
    if cache_creation_tokens is None:
        cache_creation_tokens = get("cache_creation_input_tokens")

    input_tokens = base_input
    # For Anthropic, base_input typically excludes cache tokens.
    # For OpenAI, prompt_tokens already includes them.
    # We detect Anthropic by checking for specific cache keys
    anthropic_cache_keys = (
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "cache_read_tokens",
        "cache_creation_tokens",
    )
    
    is_anthropic_style = False
    if isinstance(usage_data, dict):
        is_anthropic_style = any(key in usage_data for key in anthropic_cache_keys)
    else:
        is_anthropic_style = any(hasattr(usage_data, key) for key in anthropic_cache_keys)

    if is_anthropic_style and input_tokens is not None:
        input_tokens += (cache_read_tokens or 0) + (cache_creation_tokens or 0)

    # Calculate total_tokens (Input + Output)
    total_tokens = get("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_creation_tokens": cache_creation_tokens,
    }
