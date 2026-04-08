"""Retry utilities - Async retry decorator with exponential backoff.

Provides retry logic for async functions, commonly used for LLM API calls.
"""

import logging
from typing import Callable, TypeVar

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from agiwo.utils.logging import get_logger

T = TypeVar("T")

logger = get_logger(__name__)

RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)

try:
    from openai import (
        APIConnectionError as _OAIConn,
        APITimeoutError as _OAITimeout,
        RateLimitError as _OAIRate,
    )  # noqa: E501

    RETRYABLE_EXCEPTIONS = (*RETRYABLE_EXCEPTIONS, _OAIConn, _OAITimeout, _OAIRate)
except ImportError:
    pass

try:
    from anthropic import (
        APIConnectionError as _AConn,
        APITimeoutError as _ATimeout,
        RateLimitError as _ARate,
    )  # noqa: E501

    RETRYABLE_EXCEPTIONS = (*RETRYABLE_EXCEPTIONS, _AConn, _ATimeout, _ARate)
except ImportError:
    pass

try:
    from httpx import ConnectError as _HConn, ReadTimeout as _HRead

    RETRYABLE_EXCEPTIONS = (*RETRYABLE_EXCEPTIONS, _HConn, _HRead)
except ImportError:
    pass


def _is_retryable_error_with_types(
    exc: BaseException,
    extra_types: tuple[type[Exception], ...] = (),
) -> bool:
    """Check if an exception indicates a retryable error.

    Handles both standard exception types and provider-specific error responses
    (e.g., OpenRouter's upstream_error with 429 status).
    """
    # Check standard exception types (including extra_types from caller)
    all_retryable = (*RETRYABLE_EXCEPTIONS, *extra_types)
    if isinstance(exc, all_retryable):
        return True

    # Check for rate limit indicators in error message/content
    # This handles non-standard error formats like OpenRouter's upstream_error
    exc_str = str(exc).lower()
    retry_indicators = [
        "429",
        "rate limit",
        "rate-limit",
        "too many requests",
        "upstream_error",  # OpenRouter-style proxy errors
    ]
    if any(indicator in exc_str for indicator in retry_indicators):
        return True

    # Check for HTTP status code attributes (some SDKs attach these)
    if hasattr(exc, "status_code") and exc.status_code == 429:
        return True
    if hasattr(exc, "response") and hasattr(exc.response, "status_code"):
        if exc.response.status_code == 429:
            return True

    return False


def _make_retry_predicate(
    extra_types: tuple[type[Exception], ...] = (),
) -> Callable[[BaseException], bool]:
    """Create a retry predicate that includes extra exception types."""
    return lambda exc: _is_retryable_error_with_types(exc, extra_types)


def retry_async(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    exceptions: tuple[type[Exception], ...] = (),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for async functions to add retry logic with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        exceptions: Additional exception types to retry on (besides built-in retryable types)

    Returns:
        Decorator function
    """
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception(_make_retry_predicate(exceptions)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
