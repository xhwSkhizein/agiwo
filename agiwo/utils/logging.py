"""
Unified Structured Logging Module for Agiwo

This module provides a structured logging framework using structlog with:
- Structured JSON output for production
- Human-readable console output for development
- Request ID and context tracking
- Sensitive data filtering
- Performance monitoring
- Best practices enforcement

Usage:
    from agiwo.utils.logging import get_logger

    logger = get_logger(__name__)
    logger.info("user_authenticated", user_id=user_id, session_id=session_id)
    logger.error("api_call_failed", error=str(e), status_code=500)
"""

import logging
import os
import sys
from contextvars import ContextVar

import structlog
from structlog.types import FilteringBoundLogger

# Context variables for request tracking
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)
session_id_var: ContextVar[str | None] = ContextVar("session_id", default=None)


# Sensitive data patterns to filter
SENSITIVE_KEYS = {
    "password",
    "api_key",
    "secret",
    "authorization",
    "auth",
    "apikey",
    "access_token",
    "refresh_token",
}

# Keys that should NOT be filtered (even if they contain sensitive patterns)
ALLOWED_KEYS = {
    "tokens",
    "total_tokens",
    "input_tokens",
    "output_tokens",
    "prompt_tokens",
    "completion_tokens",
}


def filter_sensitive_data(
    logger: FilteringBoundLogger, method_name: str, event_dict: dict
) -> dict:
    """Filter out sensitive information from logs."""
    for key in list(event_dict.keys()):
        # Skip allowed keys
        if key in ALLOWED_KEYS:
            continue
        # Check if key contains sensitive patterns
        if any(sensitive in key.lower() for sensitive in SENSITIVE_KEYS):
            event_dict[key] = "***REDACTED***"
    return event_dict


def add_context(
    logger: FilteringBoundLogger, method_name: str, event_dict: dict
) -> dict:
    """Add contextual information to log entries."""
    # Add request tracking context
    if request_id := request_id_var.get():
        event_dict["request_id"] = request_id
    if user_id := user_id_var.get():
        event_dict["user_id"] = user_id
    if session_id := session_id_var.get():
        event_dict["session_id"] = session_id

    return event_dict


def configure_logging(
    log_level: str = "INFO", json_logs: bool = False, log_file: str | None = None
) -> None:
    """
    Configure structlog with appropriate processors and renderers.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: If True, output JSON logs suitable for production
        log_file: Optional file path to write logs to
    """
    # Get log level from environment or parameter
    log_level = os.getenv("LOG_LEVEL", log_level).upper()
    json_logs = os.getenv("LOG_JSON", str(json_logs)).lower() in ("true", "1", "yes")

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level),
    )

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level))
        logging.getLogger().addHandler(file_handler)

    # Processor chain
    processors = [
        structlog.contextvars.merge_contextvars,
        add_context,
        filter_sensitive_data,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    # Add appropriate renderer
    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.extend(
            [
                structlog.processors.ExceptionPrettyPrinter(),
                structlog.dev.ConsoleRenderer(
                    colors=True,
                    exception_formatter=structlog.dev.plain_traceback,
                ),
            ]
        )

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "agiwo") -> FilteringBoundLogger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Configured structlog logger

    Example:
        logger = get_logger(__name__)
        logger.info("operation_completed", duration_ms=123, items_processed=45)
    """
    return structlog.get_logger(name)


def set_request_context(
    request_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> None:
    """
    Set request context for logging.

    This context will be automatically included in all log entries
    within the same async/thread context.

    Args:
        request_id: Unique request identifier
        user_id: User identifier
        session_id: Session identifier
    """
    if request_id:
        request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)
    if session_id:
        session_id_var.set(session_id)


def clear_request_context() -> None:
    """Clear request context."""
    request_id_var.set(None)
    user_id_var.set(None)
    session_id_var.set(None)


# Initialize logging with defaults
configure_logging()

# Export convenience logger for quick use
logger = get_logger("agiwo")


__all__ = [
    "get_logger",
    "configure_logging",
    "set_request_context",
    "clear_request_context",
    "logger",
]
