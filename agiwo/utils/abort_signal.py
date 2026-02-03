import asyncio


class AbortSignal:
    """
    Abort signal for graceful cancellation of long-running operations.

    Based on asyncio.Event, supports:
    - Synchronous abort status check
    - Async wait for abort signal
    - Recording abort reason

    Examples:
        >>> signal = AbortSignal()
        >>>
        >>> # Trigger abort in another task
        >>> signal.abort("User cancelled")
        >>>
        >>> # Check in tool execution
        >>> if signal.is_aborted():
        >>>     return  # Early exit
        >>>
        >>> # Or async wait
        >>> await signal.wait()
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason: str | None = None

    def abort(self, reason: str = "Operation cancelled") -> None:
        """Trigger abort signal."""
        self._reason = reason
        self._event.set()

    def is_aborted(self) -> bool:
        """Check if abort has been triggered."""
        return self._event.is_set()

    async def wait(self) -> None:
        """Async wait for abort signal."""
        await self._event.wait()

    @property
    def reason(self) -> str | None:
        """Get abort reason."""
        return self._reason

    def reset(self) -> None:
        """Reset abort signal for reuse."""
        self._event.clear()
        self._reason = None
