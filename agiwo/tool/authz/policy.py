from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal


@dataclass(frozen=True)
class ToolPermissionProfile:
    mode: Literal["allow", "deny", "require_consent"]
    ttl_seconds: int | None = None


@dataclass(frozen=True)
class PermissionDecision:
    decision: Literal["allowed", "denied", "requires_consent"]
    reason: str
    suggested_patterns: list[str] | None = None
    expires_at_hint: datetime | None = None


class PermissionPolicy:
    def __init__(
        self,
        tool_profiles: dict[str, ToolPermissionProfile] | None = None,
    ) -> None:
        self._tool_profiles = tool_profiles or {}

    async def evaluate(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        user_id: str | None,
    ) -> PermissionDecision:
        profile = self._tool_profiles.get(tool_name)
        if profile is None:
            return PermissionDecision(
                decision="allowed",
                reason="No permission profile configured",
            )
        if profile.mode == "allow":
            return PermissionDecision(
                decision="allowed",
                reason="Tool allowed by permission profile",
            )
        if profile.mode == "deny":
            return PermissionDecision(
                decision="denied",
                reason="Tool denied by permission profile",
            )
        suggested_pattern = self._serialize_pattern(tool_name, tool_args)
        expires_at = None
        if profile.ttl_seconds is not None:
            expires_at = datetime.now() + timedelta(seconds=profile.ttl_seconds)
        return PermissionDecision(
            decision="requires_consent",
            reason="Tool requires user consent",
            suggested_patterns=[suggested_pattern] if suggested_pattern else None,
            expires_at_hint=expires_at,
        )

    def _serialize_pattern(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        clean_args = " ".join(
            f"{key}={value}"
            for key, value in sorted(tool_args.items())
            if key != "tool_call_id"
        )
        return f"{tool_name}({clean_args})" if clean_args else f"{tool_name}(*)"
