"""Adapter bridging CommandSafetyValidator decisions to PermissionPolicy."""

from typing import Any, Literal

from agiwo.tool.authz.policy import PermissionDecision
from agiwo.tool.builtin.bash_tool.security import (
    CommandSafetyDecision,
    CommandSafetyValidator,
)

BashPermissionMode = Literal[
    "auto_allow",
    "risky_require_consent",
    "always_require_consent",
]


class BashCommandPolicyAdapter:
    """Maps CommandSafetyValidator results to PermissionDecision based on a 3-tier mode.

    Modes:
        auto_allow: Only critical commands are denied. Everything else is allowed
            without user consent. This is the default.
        risky_require_consent: Critical commands are denied. Risky commands
            (matched by potential risk rules or flagged by LLM) require user
            consent. Safe commands are allowed automatically.
        always_require_consent: Critical commands are denied. All non-allowlisted
            commands require user consent.
    """

    def __init__(
        self,
        validator: CommandSafetyValidator,
        mode: BashPermissionMode = "auto_allow",
    ) -> None:
        self._validator = validator
        self._mode = mode

    async def evaluate(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        user_id: str | None,
    ) -> PermissionDecision:
        command = str(tool_args.get("command", "")).strip()
        if not command:
            return PermissionDecision(
                decision="allowed",
                reason="Empty command passes through to tool validation",
            )

        safety = await self._validator.validate(command)

        if safety.risk_level == "critical":
            return PermissionDecision(
                decision="denied",
                reason=safety.message,
            )

        if self._mode == "auto_allow":
            return self._decide_auto_allow(safety)
        if self._mode == "risky_require_consent":
            return self._decide_risky_require_consent(safety, tool_name, tool_args)
        return self._decide_always_require_consent(safety, tool_name, tool_args)

    @staticmethod
    def _decide_auto_allow(safety: CommandSafetyDecision) -> PermissionDecision:
        return PermissionDecision(
            decision="allowed",
            reason=safety.message,
        )

    @staticmethod
    def _decide_risky_require_consent(
        safety: CommandSafetyDecision,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> PermissionDecision:
        if safety.stage == "allowlist" or safety.risk_level == "low":
            return PermissionDecision(
                decision="allowed",
                reason=safety.message,
            )
        return PermissionDecision(
            decision="requires_consent",
            reason=safety.message,
            suggested_patterns=[_serialize_pattern(tool_name, tool_args)],
        )

    @staticmethod
    def _decide_always_require_consent(
        safety: CommandSafetyDecision,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> PermissionDecision:
        if safety.stage == "allowlist":
            return PermissionDecision(
                decision="allowed",
                reason=safety.message,
            )
        return PermissionDecision(
            decision="requires_consent",
            reason=safety.message,
            suggested_patterns=[_serialize_pattern(tool_name, tool_args)],
        )


def _serialize_pattern(tool_name: str, tool_args: dict[str, Any]) -> str:
    clean_args = " ".join(
        f"{key}={value}"
        for key, value in sorted(tool_args.items())
        if key != "tool_call_id"
    )
    return f"{tool_name}({clean_args})" if clean_args else f"{tool_name}(*)"
