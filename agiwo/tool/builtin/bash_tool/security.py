"""Minimal deterministic safety checks for BashTool."""

import re
from dataclasses import dataclass, field
from typing import Literal, Pattern


@dataclass(frozen=True)
class SecurityRule:
    """Pattern-based hard block used during bash preflight checks."""

    description: str
    pattern: Pattern[str]


@dataclass(frozen=True)
class CommandSafetyDecision:
    """Minimal allow/deny decision for bash commands."""

    action: Literal["allow", "deny"]
    reason: str
    matched_rules: tuple[str, ...] = field(default_factory=tuple)


ABSOLUTE_BLOCK_RULES: list[SecurityRule] = [
    SecurityRule(
        description="Removing filesystem root is always blocked",
        pattern=re.compile(
            r"\brm\b[^\n]*(?:-rf|-fr|-r\s+-f|-f\s+-r)[^\n]*(?:^|[\s\"'])/(?:\s|$|[\"'])",
            re.IGNORECASE,
        ),
    ),
    SecurityRule(
        description="Removing home directory is always blocked",
        pattern=re.compile(
            r"\brm\b[^\n]*(?:-rf|-fr|-r\s+-f|-f\s+-r)[^\n]*(?:~|\$HOME|\$\{HOME\})",
            re.IGNORECASE,
        ),
    ),
    SecurityRule(
        description="Fork bomb pattern is always blocked",
        pattern=re.compile(r":\(\)\s*\{\s*:\|:&\s*\};:", re.IGNORECASE),
    ),
    SecurityRule(
        description="Filesystem formatting commands are always blocked",
        pattern=re.compile(r"\bmkfs(?:\.\w+)?\b", re.IGNORECASE),
    ),
    SecurityRule(
        description="Raw disk overwrite commands are always blocked",
        pattern=re.compile(r"\bdd\b[^\n]*\bof=/dev/(?:sd|vd|xvd|nvme)", re.IGNORECASE),
    ),
]


class CommandSafetyValidator:
    """Deny only absolute-dangerous commands; allow everything else."""

    async def validate(self, command: str) -> CommandSafetyDecision:
        blocked_matches = self._match_rules(command, ABSOLUTE_BLOCK_RULES)
        if blocked_matches:
            descriptions = tuple(rule.description for rule in blocked_matches)
            return CommandSafetyDecision(
                action="deny",
                reason=f"Blocked by hard safety rule: {descriptions[0]}",
                matched_rules=descriptions,
            )
        return CommandSafetyDecision(
            action="allow",
            reason="Command passed hard safety checks",
        )

    @staticmethod
    def _match_rules(command: str, rules: list[SecurityRule]) -> list[SecurityRule]:
        return [rule for rule in rules if rule.pattern.search(command)]


__all__ = [
    "ABSOLUTE_BLOCK_RULES",
    "CommandSafetyDecision",
    "CommandSafetyValidator",
    "SecurityRule",
]
