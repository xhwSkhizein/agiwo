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

# Common retryable exceptions for LLM APIs
# We will catch generic Exception for now or specific ones if we import them from openai
# Ideally we should catch openai.RateLimitError, openai.APIError, openai.Timeout
RETRYABLE_EXCEPTIONS = (Exception,)


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
