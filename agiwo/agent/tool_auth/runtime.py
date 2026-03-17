import asyncio
from dataclasses import dataclass
from typing import Any

from agiwo.agent.tool_auth.notifier import (
    NoOpToolConsentNotifier,
    ToolConsentNotifier,
)
from agiwo.agent.tool_auth.state import ConsentWaiter
from agiwo.agent.runtime import AgentContext
from agiwo.tool.authz import ConsentStore, PermissionPolicy


@dataclass(frozen=True)
class AuthorizationOutcome:
    allowed: bool
    reason: str
    from_cache: bool = False


class ToolAuthorizationRuntime:
    """Agent-side orchestration for tool authorization and consent."""

    def __init__(
        self,
        policy: PermissionPolicy | None = None,
        consent_store: ConsentStore | None = None,
        waiter: ConsentWaiter | None = None,
        notifier: ToolConsentNotifier | None = None,
    ) -> None:
        self._policy = policy
        self._consent_store = consent_store
        self._waiter = waiter or (ConsentWaiter() if policy is not None else None)
        self._notifier = notifier or NoOpToolConsentNotifier()

    async def authorize(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        context: AgentContext,
        timeout: float = 300.0,
    ) -> AuthorizationOutcome:
        if self._policy is None:
            return self._allow("Authorization disabled")

        policy_decision = await self._policy.evaluate(
            tool_name=tool_name,
            tool_args=tool_args,
            user_id=context.user_id,
        )
        if policy_decision.decision == "allowed":
            return self._allow(policy_decision.reason)

        if policy_decision.decision == "denied":
            return await self._deny(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                run_id=context.run_id,
                reason=policy_decision.reason,
            )

        if context.user_id is None:
            return await self._deny(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                run_id=context.run_id,
                reason="User consent required but user_id is missing",
            )

        cached_outcome = await self._check_cached_consent(
            user_id=context.user_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_args=tool_args,
            run_id=context.run_id,
        )
        if cached_outcome is not None:
            return cached_outcome

        await self._notifier.notify_required(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            run_id=context.run_id,
            args_preview=str(tool_args)[:500],
            reason=policy_decision.reason,
            suggested_patterns=policy_decision.suggested_patterns,
        )
        return await self._await_user_consent(
            user_id=context.user_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            run_id=context.run_id,
            timeout=timeout,
            suggested_patterns=policy_decision.suggested_patterns,
            expires_at_hint=policy_decision.expires_at_hint,
        )

    def _allow(self, reason: str, *, from_cache: bool = False) -> AuthorizationOutcome:
        return AuthorizationOutcome(allowed=True, reason=reason, from_cache=from_cache)

    async def _deny(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        run_id: str,
        reason: str,
        from_cache: bool = False,
    ) -> AuthorizationOutcome:
        await self._notifier.notify_denied(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            run_id=run_id,
            reason=reason,
        )
        return AuthorizationOutcome(
            allowed=False,
            reason=reason,
            from_cache=from_cache,
        )

    async def _check_cached_consent(
        self,
        *,
        user_id: str,
        tool_call_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        run_id: str,
    ) -> AuthorizationOutcome | None:
        if self._consent_store is None:
            return None

        cached_decision = await self._consent_store.check_consent(
            user_id,
            tool_name,
            tool_args,
        )
        if cached_decision == "allowed":
            return self._allow("Allowed by cached user consent", from_cache=True)
        if cached_decision == "denied":
            return await self._deny(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                run_id=run_id,
                reason="Denied by cached user consent",
                from_cache=True,
            )
        return None

    async def _await_user_consent(
        self,
        *,
        user_id: str,
        tool_call_id: str,
        tool_name: str,
        run_id: str,
        timeout: float,
        suggested_patterns: list[str] | None,
        expires_at_hint,
    ) -> AuthorizationOutcome:
        if self._waiter is None:
            return await self._deny(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                run_id=run_id,
                reason="User consent required but no consent waiter is configured",
            )

        try:
            decision = await self._waiter.wait_for_consent(tool_call_id, timeout)
        except asyncio.TimeoutError:
            return await self._deny(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                run_id=run_id,
                reason="User consent timed out",
            )

        consent_patterns = decision.patterns or suggested_patterns or []
        expires_at = decision.expires_at or expires_at_hint
        if decision.decision == "allow":
            if self._consent_store is not None:
                await self._consent_store.save_consent(
                    user_id=user_id,
                    tool_name=tool_name,
                    patterns=consent_patterns,
                    expires_at=expires_at,
                )
            return self._allow("Allowed by user consent")

        if self._consent_store is not None:
            await self._consent_store.save_consent(
                user_id=user_id,
                tool_name=tool_name,
                patterns=[],
                deny_patterns=consent_patterns,
                expires_at=expires_at,
            )
        return await self._deny(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            run_id=run_id,
            reason="Denied by user consent",
        )
