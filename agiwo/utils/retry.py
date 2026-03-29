"""
Retry utilities - Async retry decorator with exponential backoff.

Provides retry logic for async functions, commonly used for LLM API calls.
"""

import logging
from typing import Callable, TypeVar

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
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
    from openai import APIConnectionError as _OAIConn, APITimeoutError as _OAITimeout, RateLimitError as _OAIRate  # noqa: E501
    RETRYABLE_EXCEPTIONS = (*RETRYABLE_EXCEPTIONS, _OAIConn, _OAITimeout, _OAIRate)
except ImportError:
    pass

try:
    from anthropic import APIConnectionError as _AConn, APITimeoutError as _ATimeout, RateLimitError as _ARate  # noqa: E501
    RETRYABLE_EXCEPTIONS = (*RETRYABLE_EXCEPTIONS, _AConn, _ATimeout, _ARate)
except ImportError:
    pass

try:
    from httpx import ConnectError as _HConn, ReadTimeout as _HRead
    RETRYABLE_EXCEPTIONS = (*RETRYABLE_EXCEPTIONS, _HConn, _HRead)
except ImportError:
    pass


def retry_async(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    exceptions: tuple[type[Exception], ...] = RETRYABLE_EXCEPTIONS,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for async functions to add retry logic with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        exceptions: Tuple of exception types to retry on

    Returns:
        Decorator function
    """
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(exceptions),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
