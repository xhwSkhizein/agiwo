"""Shared decorator for uniform storage error logging."""

import functools
from typing import Any, Callable

from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


def storage_op(op_name: str, **log_extractors: Callable[..., Any]):
    """Wrap async storage methods with uniform error logging.

    Usage::

        @storage_op("save_run", run_id=lambda self, run: run.id)
        async def save_run(self, run: Run) -> None:
            ...
    """

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                extra = {}
                for key, extractor in log_extractors.items():
                    try:
                        extra[key] = extractor(*args, **kwargs)
                    except Exception:  # noqa: BLE001
                        pass
                logger.error(f"{op_name}_failed", error=str(exc), **extra)
                raise

        return wrapper

    return decorator
