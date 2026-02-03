"""
PermissionManager - Unified permission manager.

Integrates all permission-related functionality:
- Permission checking (cache + DB)
- Authorization wait coordination
- Returns explicit authorization results (allowed/denied)
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from agiwo.agent.schema import StepEvent, StepEventType
from agiwo.tool.permission.consent_store import ConsentStore
from agiwo.tool.permission.consent_waiter import ConsentWaiter
from agiwo.tool.permission.service import PermissionDecision, PermissionService
from agiwo.agent.execution_context import ExecutionContext
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class ConsentResult(BaseModel):
    """Authorization check result"""

    allowed: bool  # True=allow execution, False=deny execution
    reason: str  # Reason description
    from_cache: bool = False  # Whether result is from cache


class PermissionManager:
    """
    Unified permission manager.

    Responsibilities:
    1. Permission checking (cache + DB)
    2. Authorization wait coordination
    3. Returns explicit authorization results (allowed/denied)
    """

    def __init__(
        self,
        consent_store: ConsentStore,
        consent_waiter: ConsentWaiter,
        permission_service: PermissionService,
        tool_configs: dict[str, dict] | None = None,
        cache_ttl: int = 300,  # Cache TTL (seconds)
        cache_size: int = 1000,  # Cache size
    ):
        """
        Initialize permission manager.

        Args:
            consent_store: ConsentStore instance
            consent_waiter: ConsentWaiter instance
            permission_service: PermissionService instance
            tool_configs: Tool configurations dict {tool_name: config}
            cache_ttl: Cache TTL in seconds (default: 300)
            cache_size: Maximum cache size (default: 1000)
        """
        self.consent_store = consent_store
        self.consent_waiter = consent_waiter
        self.permission_service = permission_service
        self._tool_configs = tool_configs or {}
        # Simple cache: key = (user_id, tool_name, args_hash), value = (ConsentResult, timestamp)
        self._cache: dict[str, tuple[ConsentResult, float]] = {}
        self._cache_ttl = cache_ttl
        self._cache_size = cache_size
        self._cache_lock = asyncio.Lock()

    async def check_and_wait_consent(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        context: ExecutionContext,
        timeout: float = 300.0,
    ) -> ConsentResult:
        """
        Check and wait for authorization, returns explicit authorization result.

        Args:
            tool_call_id: Tool call unique identifier
            tool_name: Tool name
            tool_args: Tool arguments
            context: Execution context
            timeout: Timeout in seconds

        Returns:
            ConsentResult: Authorization result (allowed=True/False)

        Process:
        1. Check cache (hit and not expired → return directly)
        2. Check tool config requires_consent (False → allowed)
        3. Query ConsentStore pattern matching (hit → allowed/denied)
        4. Call PermissionService (policy decision)
        5. If requires consent → send event + wait for user decision
        6. Return explicit ConsentResult (allowed=True/False)
        7. Cache result
        """
        user_id = context.user_id
        if not user_id:
            # No user_id, treat as requires consent
            return await self._request_consent(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                tool_args=tool_args,
                context=context,
                timeout=timeout,
            )

        # 1. Check cache
        cache_key = self._make_cache_key(user_id, tool_name, tool_args)
        cached_result = await self._get_from_cache(cache_key)
        if cached_result:
            return cached_result

        # 2. Check tool configuration
        tool_config = self._get_tool_config(tool_name)
        requires_consent = (
            tool_config.get("requires_consent", False) if tool_config else False
        )

        if not requires_consent:
            result = ConsentResult(allowed=True, reason="Tool does not require consent")
            await self._set_cache(cache_key, result)
            return result

        # 3. Query ConsentStore (pattern matching)
        store_result = await self.consent_store.check_consent(
            user_id=user_id,
            tool_name=tool_name,
            tool_args=tool_args,
        )

        if store_result == "allowed":
            result = ConsentResult(allowed=True, reason="Matched allow pattern")
            await self._set_cache(cache_key, result)
            return result

        if store_result == "denied":
            result = ConsentResult(allowed=False, reason="Matched deny pattern")
            await self._set_cache(cache_key, result)
            return result

        # 4. Call PermissionService
        permission_decision = await self.permission_service.check_permission(
            user_id=user_id,
            tool_name=tool_name,
            tool_args=tool_args,
            context=context,
        )

        if permission_decision.decision == "allowed":
            result = ConsentResult(allowed=True, reason=permission_decision.reason)
            await self._set_cache(cache_key, result)
            return result

        if permission_decision.decision == "denied":
            result = ConsentResult(allowed=False, reason=permission_decision.reason)
            await self._set_cache(cache_key, result)
            return result

        # 5. requires_consent: Request user authorization
        return await self._request_consent(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_args=tool_args,
            context=context,
            permission_decision=permission_decision,
            timeout=timeout,
        )

    async def _request_consent(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        context: ExecutionContext,
        permission_decision: PermissionDecision | None = None,
        timeout: float = 300.0,
    ) -> ConsentResult:
        """Request user authorization and wait for decision"""
        # 1. Send TOOL_AUTH_REQUIRED event
        await self._send_auth_required_event(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_args=tool_args,
            context=context,
            permission_decision=permission_decision,
        )

        # 2. Wait for user decision
        try:
            decision = await self.consent_waiter.wait_for_consent(
                tool_call_id=tool_call_id,
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            # Timeout treated as deny
            result = ConsentResult(
                allowed=False, reason="User consent request timed out"
            )
            await self._send_auth_denied_event(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                context=context,
                reason="Timeout",
            )
            return result

        # 3. Save authorization record (if allowed)
        if decision.decision == "allow" and decision.patterns:
            user_id = context.user_id
            if user_id:
                await self.consent_store.save_consent(
                    user_id=user_id,
                    tool_name=tool_name,
                    patterns=decision.patterns,
                    expires_at=decision.expires_at,
                )
                # Clear related cache
                await self._invalidate_cache(user_id, tool_name)

        # 4. Return result
        if decision.decision == "deny":
            result = ConsentResult(allowed=False, reason="User denied consent")
            await self._send_auth_denied_event(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                context=context,
                reason="User denied",
            )
            return result

        # 5. Allow execution
        result = ConsentResult(allowed=True, reason="User granted consent")
        # Cache result
        if context.user_id:
            cache_key = self._make_cache_key(context.user_id, tool_name, tool_args)
            await self._set_cache(cache_key, result)
        return result

    def _make_cache_key(
        self, user_id: str, tool_name: str, tool_args: dict[str, Any]
    ) -> str:
        """Generate cache key"""
        # Serialize arguments (exclude tool_call_id)
        args_copy = {k: v for k, v in tool_args.items() if k != "tool_call_id"}
        args_str = json.dumps(args_copy, sort_keys=True)
        args_hash = hashlib.md5(args_str.encode()).hexdigest()[:8]

        return f"{user_id}:{tool_name}:{args_hash}"

    async def _get_from_cache(self, cache_key: str) -> ConsentResult | None:
        """Get result from cache"""
        async with self._cache_lock:
            if cache_key not in self._cache:
                return None

            result, timestamp = self._cache[cache_key]

            # Check if expired
            if time.time() - timestamp > self._cache_ttl:
                del self._cache[cache_key]
                return None

            # Return cached result (marked as from cache)
            return ConsentResult(
                allowed=result.allowed,
                reason=result.reason,
                from_cache=True,
            )

    async def _set_cache(self, cache_key: str, result: ConsentResult) -> None:
        """Set cache"""
        async with self._cache_lock:
            # LRU eviction: if cache is full, delete oldest
            if len(self._cache) >= self._cache_size:
                # Delete oldest (simple strategy: delete first)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            self._cache[cache_key] = (result, time.time())

    async def _invalidate_cache(
        self, user_id: str, tool_name: str | None = None
    ) -> None:
        """Invalidate cache (called after consent record update)"""
        async with self._cache_lock:
            keys_to_delete = []
            prefix = f"{user_id}:{tool_name}:" if tool_name else f"{user_id}:"
            for key in self._cache.keys():
                if key.startswith(prefix):
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                del self._cache[key]

    def _get_tool_config(self, tool_name: str) -> dict[str, Any] | None:
        """Get tool configuration from tool_configs dict."""
        return self._tool_configs.get(tool_name)

    def _summarize_args(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        """Summarize tool arguments for display"""
        if tool_name == "bash":
            return tool_args.get("command", "")
        elif tool_name in ["file_read", "file_edit", "file_write"]:
            return tool_args.get("path") or tool_args.get("file_path", "")
        else:
            # Generic summary
            items = [f"{k}={v}" for k, v in tool_args.items() if k != "tool_call_id"]
            return ", ".join(items[:3])  # Limit to 3 items

    async def _send_auth_required_event(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        context: ExecutionContext,
        permission_decision: PermissionDecision | None = None,
    ) -> None:
        """Send TOOL_AUTH_REQUIRED event"""
        args_preview = self._summarize_args(tool_name, tool_args)

        event = StepEvent(
            type=StepEventType.TOOL_AUTH_REQUIRED,
            run_id=context.run_id,
            step_id=None,  # Tool call hasn't created Step yet
            data={
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "args_preview": args_preview,
                "suggested_patterns": (
                    permission_decision.suggested_patterns
                    if permission_decision
                    else None
                ),
                "reason": (
                    permission_decision.reason
                    if permission_decision
                    else "Tool requires user consent"
                ),
                "expires_at_hint": (
                    permission_decision.expires_at_hint.isoformat()
                    if permission_decision and permission_decision.expires_at_hint
                    else None
                ),
            },
            timestamp=datetime.now(),
        )

        await context.wire.write(event)
        logger.info(
            "tool_auth_required_sent",
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )

    async def _send_auth_denied_event(
        self,
        tool_call_id: str,
        tool_name: str,
        context: ExecutionContext,
        reason: str,
    ) -> None:
        """Send TOOL_AUTH_DENIED event"""
        event = StepEvent(
            type=StepEventType.TOOL_AUTH_DENIED,
            run_id=context.run_id,
            data={
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "reason": reason,
            },
            timestamp=datetime.now(),
        )

        await context.wire.write(event)
        logger.info(
            "tool_auth_denied_sent",
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            reason=reason,
        )


__all__ = ["PermissionManager", "ConsentResult"]
