"""
Tool Result Cache - Cache expensive tool results within a session.

Caching is controlled by tool's `cacheable` attribute.
ToolExecutor checks this attribute before using cache.
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

from agiwo.tool.base import ToolResult
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """A cached tool result."""

    result: ToolResult
    created_at: float = field(default_factory=time.time)


class ToolResultCache:
    """
    In-memory cache for tool results.

    Cache is scoped to a session. Whether to use cache is determined
    by tool's `cacheable` attribute, checked by ToolExecutor.
    """

    def __init__(self, ttl_seconds: int = 3600) -> None:
        """
        Initialize cache.

        Args:
            ttl_seconds: Time-to-live for cache entries (default: 1 hour)
        """
        self._cache: dict[str, CacheEntry] = {}
        self._ttl = ttl_seconds

    def _make_key(self, session_id: str, tool_name: str, args: dict[str, Any]) -> str:
        """Create a cache key from session, tool, and arguments."""
        # Filter out internal args (starting with _)
        clean_args = {k: v for k, v in sorted(args.items()) if not k.startswith("_")}
        args_str = str(clean_args)
        key_str = f"{session_id}:{tool_name}:{args_str}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def get(
        self,
        session_id: str,
        tool_name: str,
        args: dict[str, Any],
    ) -> ToolResult | None:
        """
        Get cached result if exists and not expired.

        Args:
            session_id: Session ID
            tool_name: Tool name
            args: Tool arguments

        Returns:
            Cached ToolResult or None
        """
        key = self._make_key(session_id, tool_name, args)
        entry = self._cache.get(key)

        if entry is None:
            return None

        # Check TTL
        if time.time() - entry.created_at > self._ttl:
            del self._cache[key]
            return None

        logger.debug(
            "tool_cache_hit",
            tool_name=tool_name,
            session_id=session_id,
        )
        return entry.result

    def set(
        self,
        session_id: str,
        tool_name: str,
        args: dict[str, Any],
        result: ToolResult,
    ) -> None:
        """
        Cache a tool result.

        Args:
            session_id: Session ID
            tool_name: Tool name
            args: Tool arguments
            result: Tool execution result
        """
        if not result.is_success:
            return

        key = self._make_key(session_id, tool_name, args)
        self._cache[key] = CacheEntry(result=result)

        logger.debug(
            "tool_cache_set",
            tool_name=tool_name,
            session_id=session_id,
        )

    def clear_session(self, session_id: str) -> int:
        """
        Clear all cache entries for a session.

        Args:
            session_id: Session ID

        Returns:
            Number of entries cleared
        """
        # This is O(n) but cache size should be small
        keys_to_delete = [
            k
            for k in self._cache
            if k.startswith(session_id[:16])  # Rough prefix match
        ]
        for k in keys_to_delete:
            del self._cache[k]
        return len(keys_to_delete)

    def clear_all(self) -> None:
        """Clear entire cache."""
        self._cache.clear()


# Global cache instance (can be replaced with Redis etc. later)
_global_cache: ToolResultCache | None = None


def get_tool_cache() -> ToolResultCache:
    """Get the global tool cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = ToolResultCache()
    return _global_cache


__all__ = ["ToolResultCache", "get_tool_cache"]
