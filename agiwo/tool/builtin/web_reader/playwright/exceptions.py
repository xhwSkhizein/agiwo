"""
Playwright exceptions module.

Defines custom exceptions for Playwright-based web fetching.
"""


class BlockedException(Exception):
    """Exception raised when request is blocked."""

    def __init__(self, message: str, **context) -> None:
        super().__init__(message, **context)


class SessionInvalidException(Exception):
    """Exception raised when session is invalid."""

    def __init__(self, message: str, **context) -> None:
        super().__init__(message, **context)
