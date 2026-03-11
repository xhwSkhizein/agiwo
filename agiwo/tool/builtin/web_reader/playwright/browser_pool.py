"""
Browser pool management for Playwright-based fetching.

Provides reusable browser sessions with lifecycle control, idle eviction, and
broken-session handling.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Callable

from agiwo.tool.builtin.config import WebReaderApiConfig
from agiwo.tool.builtin.web_reader.playwright.chrome_session import (
    ChromeSessionManager,
)
from agiwo.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class BrowserLease:
    """A leased browser/session resource."""

    session_manager: ChromeSessionManager
    acquired_at: float
    last_used_at: float
    use_count: int = 0
    in_use: bool = False

    @property
    def context(self):
        return self.session_manager.context


class BrowserPool:
    """Manage Playwright browser sessions with reuse and cleanup."""

    def __init__(
        self,
        config: WebReaderApiConfig,
        session_factory: Callable[[], ChromeSessionManager] | None = None,
    ) -> None:
        self._config = config
        self._session_factory = session_factory or (
            lambda: ChromeSessionManager(config=self._config)
        )
        self._max_browsers = max(1, self._config.max_browsers)
        self._idle_ttl = max(30, self._config.browser_idle_ttl_seconds)
        self._max_use_count = max(1, self._config.browser_max_uses)

        self._lock = asyncio.Lock()
        self._resources: list[BrowserLease] = []
        self._waiters: list[asyncio.Future] = []

    async def acquire(self) -> BrowserLease:
        """Acquire a browser lease, waiting if pool is saturated."""
        async with self._lock:
            self._evict_stale_locked()

            # Reuse available lease
            for lease in self._resources:
                if not lease.in_use and lease.session_manager.is_connected():
                    lease.in_use = True
                    lease.last_used_at = time.time()
                    lease.use_count += 1
                    return lease

            # Create new if capacity allows
            if len(self._resources) < self._max_browsers:
                lease = await self._create_lease_locked()
                lease.in_use = True
                lease.last_used_at = time.time()
                lease.use_count += 1
                return lease

            # Otherwise wait for release
            loop = asyncio.get_running_loop()
            fut: asyncio.Future[BrowserLease] = loop.create_future()
            self._waiters.append(fut)

        lease = await fut
        return lease

    async def release(self, lease: BrowserLease, broken: bool = False) -> None:
        """Release a lease. If broken, destroy and recreate slot."""
        async with self._lock:
            if lease not in self._resources:
                return

            if (
                broken
                or not lease.session_manager.is_connected()
                or lease.use_count >= self._max_use_count
            ):
                await self._close_lease_locked(lease)
                self._resources.remove(lease)
                lease = await self._create_lease_locked()

            lease.in_use = False
            lease.last_used_at = time.time()

            self._wake_waiter_locked(lease)

    async def shutdown(self) -> None:
        """Close all resources."""
        async with self._lock:
            for lease in list(self._resources):
                await self._close_lease_locked(lease)
            self._resources.clear()

            for waiter in self._waiters:
                if not waiter.done():
                    waiter.cancel()
            self._waiters.clear()

    def _wake_waiter_locked(self, lease: BrowserLease) -> None:
        while self._waiters:
            waiter = self._waiters.pop(0)
            if not waiter.done():
                lease.in_use = True
                lease.last_used_at = time.time()
                lease.use_count += 1
                waiter.set_result(lease)
                return

    def _evict_stale_locked(self) -> None:
        now = time.time()
        stale = [
            lease
            for lease in self._resources
            if (not lease.in_use) and (now - lease.last_used_at > self._idle_ttl)
        ]
        for lease in stale:
            self._resources.remove(lease)
            # best-effort close without awaiting to avoid holding lock long
            asyncio.create_task(self._safe_close(lease))

    async def _safe_close(self, lease: BrowserLease) -> None:
        try:
            await lease.session_manager.disconnect()
        except Exception as exc:  # pragma: no cover - defensive log
            logger.warning("Failed to close stale lease", error=str(exc))

    async def _create_lease_locked(self) -> BrowserLease:
        manager = self._session_factory()
        await manager.connect()
        now = time.time()
        lease = BrowserLease(
            session_manager=manager,
            acquired_at=now,
            last_used_at=now,
        )
        self._resources.append(lease)
        return lease

    async def _close_lease_locked(self, lease: BrowserLease) -> None:
        try:
            await lease.session_manager.disconnect()
        except Exception as exc:  # pragma: no cover - defensive log
            logger.warning("Failed to close lease", error=str(exc))
