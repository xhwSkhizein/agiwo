"""
PermissionManager - Unified permission manager.

Integrates all permission-related functionality:
- Permission policy service (PermissionService + PermissionDecision)
- Permission checking (cache + DB)
- Authorization wait coordination
- Global singleton factory (get_permission_manager / reset_permission_manager)
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel

from agiwo.agent.execution_context import ExecutionContext
from agiwo.agent.runtime import EventType, StreamEvent
from agiwo.tool.permission.store import ConsentDecision, ConsentStore, ConsentWaiter, InMemoryConsentStore
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


# ── Data Models ──────────────────────────────────────────────────────────


class PermissionDecision(BaseModel):
    """Permission decision result"""

    decision: Literal["allowed", "denied", "requires_consent"]
    reason: str
    suggested_patterns: list[str] | None = None  # Suggested patterns
    expires_at_hint: datetime | None = None  # Suggested expiration time


class ConsentResult(BaseModel):
    """Authorization check result"""

    allowed: bool  # True=allow execution, False=deny execution
    reason: str  # Reason description
    from_cache: bool = False  # Whether result is from cache


# ── PermissionService ────────────────────────────────────────────────────


class PermissionService:
    """
    Permission policy service.

    Generates permission decisions based on tool configuration and context.
    """

    def __init__(self, tool_configs: dict[str, dict] | None = None) -> None:
        self._tool_configs = tool_configs or {}

    async def check_permission(
        self,
        user_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        context: ExecutionContext,
    ) -> PermissionDecision:
        """
        Check tool execution permission.

        Decision logic:
        1. If tool config requires_consent=False → allowed
        2. If tool config requires_consent=True → requires_consent
        3. Agent can override default policy (e.g., read-only tool set)
        """
        # Get tool configuration
        tool_config = self._tool_configs.get(tool_name)

        if not tool_config:
            # Tool not found in config, default to requires_consent
            return PermissionDecision(
                decision="requires_consent",
                reason=f"Tool {tool_name} configuration not found",
                suggested_patterns=self._generate_suggested_patterns(
                    tool_name, tool_args
                ),
            )

        requires_consent = tool_config.get("requires_consent", False)

        if not requires_consent:
            return PermissionDecision(
                decision="allowed",
                reason="Tool does not require consent",
            )

        # Tool requires consent
        return PermissionDecision(
            decision="requires_consent",
            reason="Tool requires user consent",
            suggested_patterns=self._generate_suggested_patterns(tool_name, tool_args),
            expires_at_hint=datetime.now() + timedelta(days=30),  # Default: 30 days
        )

    def _generate_suggested_patterns(
        self, tool_name: str, tool_args: dict[str, Any]
    ) -> list[str]:
        """Generate suggested patterns for user consent."""
        patterns = []

        # Generate pattern based on tool type
        if tool_name == "bash":
            # For bash, suggest pattern based on command
            command = tool_args.get("command", "")
            if command:
                # Exact match pattern
                patterns.append(f"bash({command})")
                # If command contains common patterns, suggest wildcard
                if "run" in command.lower():
                    # Suggest pattern for npm/yarn run commands
                    parts = command.split()
                    if len(parts) >= 2:
                        base_cmd = " ".join(parts[:2])
                        patterns.append(f"bash({base_cmd} *)")

        elif tool_name in ["file_read", "file_edit", "file_write"]:
            # For file operations, suggest pattern based on file path
            path = tool_args.get("path") or tool_args.get("file_path")
            if path:
                patterns.append(f"{tool_name}({path})")
                # Suggest parent directory pattern
                if "/" in path:
                    parent = "/".join(path.split("/")[:-1])
                    if parent:
                        patterns.append(f"{tool_name}({parent}/*)")

        else:
            # Generic pattern: exact match
            args_str = self._serialize_args(tool_args)
            if args_str:
                patterns.append(f"{tool_name}({args_str})")

        return patterns[:3]  # Limit to 3 suggestions

    def _serialize_args(self, args: dict[str, Any]) -> str:
        """Serialize arguments to string for pattern generation"""
        items = sorted([(k, v) for k, v in args.items() if k != "tool_call_id"])
        return " ".join(f"{k}={v}" for k, v in items)


# ── PermissionManager ────────────────────────────────────────────────────


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

        event = StreamEvent(
            type=EventType.TOOL_AUTH_REQUIRED,
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

        await context.channel.write(event)
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
        event = StreamEvent(
            type=EventType.TOOL_AUTH_DENIED,
            run_id=context.run_id,
            data={
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "reason": reason,
            },
            timestamp=datetime.now(),
        )

        await context.channel.write(event)
        logger.info(
            "tool_auth_denied_sent",
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            reason=reason,
        )


# ── Global Singleton Factory ─────────────────────────────────────────────


_permission_manager: PermissionManager | None = None


def get_permission_manager() -> PermissionManager:
    """
    Get global PermissionManager singleton.

    The PermissionManager is shared across all Agents in the system.
    """
    global _permission_manager

    if _permission_manager is None:

        logger.info("initializing_global_permission_manager")

        _permission_manager = PermissionManager(
            consent_store=InMemoryConsentStore(),
            consent_waiter=ConsentWaiter(default_timeout=300.0),
            permission_service=PermissionService(),
            cache_ttl=300,
            cache_size=1000,
        )

    return _permission_manager


def reset_permission_manager() -> None:
    """
    Reset global PermissionManager singleton.

    This is primarily used for testing to ensure a clean state.
    """
    global _permission_manager
    _permission_manager = None
    logger.debug("permission_manager_reset")


__all__ = [
    "PermissionDecision",
    "ConsentResult",
    "PermissionService",
    "PermissionManager",
    "get_permission_manager",
    "reset_permission_manager",
]
