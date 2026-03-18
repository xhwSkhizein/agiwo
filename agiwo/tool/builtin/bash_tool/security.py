"""Command safety checks for BashTool."""

import fnmatch
import inspect
import re
import shlex
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Pattern

RiskLevel = Literal["low", "medium", "high", "critical", "unknown"]


@dataclass(frozen=True)
class SecurityRule:
    """Pattern-based security rule."""

    description: str
    pattern: Pattern[str]


@dataclass
class CommandRiskEvaluationRequest:
    """Input sent to an LLM risk evaluator."""

    command: str
    matched_rules: list[str]

    def prompt(self) -> str:
        """Build a prompt requesting strict JSON output from an LLM."""
        rule_lines = "\n".join(f"- {rule}" for rule in self.matched_rules)
        return (
            "Assess this shell command for security risk.\n"
            "Return ONLY JSON with keys: risk_level, summary, possible_risks, recommended_action.\n"
            "risk_level must be one of: low, medium, high, critical.\n"
            f"Command: {self.command}\n"
            f"Matched heuristics:\n{rule_lines}\n"
        )


@dataclass
class CommandRiskEvaluation:
    """Normalized command risk evaluation result."""

    risk_level: RiskLevel
    summary: str
    possible_risks: list[str] = field(default_factory=list)
    recommended_action: str | None = None
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize evaluation for tool responses."""
        return {
            "risk_level": self.risk_level,
            "summary": self.summary,
            "possible_risks": self.possible_risks,
            "recommended_action": self.recommended_action,
            "raw": self.raw,
        }


RiskEvaluatorOutput = CommandRiskEvaluation | dict[str, Any]
CommandRiskEvaluator = Callable[
    [CommandRiskEvaluationRequest],
    Awaitable[RiskEvaluatorOutput] | RiskEvaluatorOutput,
]


@dataclass
class CommandSafetyPolicy:
    """Policy controlling how safety checks are enforced."""

    llm_block_threshold: RiskLevel = "high"
    block_when_evaluator_missing: bool = True
    enable_safe_command_whitelist: bool = True
    safe_command_allowlist: tuple[str, ...] = ()
    safe_command_pattern_allowlist: tuple[str, ...] = ()


@dataclass
class CommandSafetyDecision:
    """Decision from command safety validation."""

    allowed: bool
    stage: Literal["allow", "allowlist", "blacklist", "llm"]
    risk_level: RiskLevel
    message: str
    matched_rules: list[str] = field(default_factory=list)
    evaluation: CommandRiskEvaluation | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize decision for tool responses."""
        return {
            "allowed": self.allowed,
            "stage": self.stage,
            "risk_level": self.risk_level,
            "message": self.message,
            "matched_rules": self.matched_rules,
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
        }


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

POTENTIAL_RISK_RULES: list[SecurityRule] = [
    SecurityRule(
        description="Recursive force deletion can remove large amounts of data",
        pattern=re.compile(r"\brm\b[^\n]*(?:-rf|-fr|-r\s+-f|-f\s+-r)", re.IGNORECASE),
    ),
    SecurityRule(
        description="Privilege escalation command detected",
        pattern=re.compile(r"\b(?:sudo|su)\b", re.IGNORECASE),
    ),
    SecurityRule(
        description="Downloaded script piped to shell",
        pattern=re.compile(r"\b(?:curl|wget)\b[^\n]*\|\s*(?:sh|bash)\b", re.IGNORECASE),
    ),
    SecurityRule(
        description="Writing directly into system directories",
        pattern=re.compile(
            r"(?:>|>>)\s*/(?:etc|boot|usr|bin|sbin|root)\b", re.IGNORECASE
        ),
    ),
    SecurityRule(
        description="Dangerous recursive permission change",
        pattern=re.compile(r"\bchmod\b[^\n]*\b-R\b[^\n]*\b777\b[^\n]*/", re.IGNORECASE),
    ),
]

DEFAULT_SAFE_COMMAND_ALLOWLIST: tuple[str, ...] = (
    "ls",
    "pwd",
    "whoami",
    "id",
    "date",
    "uname",
    "echo",
    "cat",
    "head",
    "tail",
    "wc",
    "grep",
    "find",
    "ps",
    "env",
    "which",
    "type",
    "stat",
)

SAFE_GIT_SUBCOMMANDS: set[str] = {
    "status",
    "log",
    "diff",
    "show",
    "branch",
    "rev-parse",
}


class CommandSafetyValidator:
    """Two-layer command safety validator."""

    def __init__(
        self,
        risk_evaluator: CommandRiskEvaluator | None = None,
        policy: CommandSafetyPolicy | None = None,
    ) -> None:
        self.risk_evaluator = risk_evaluator
        self.policy = policy or CommandSafetyPolicy()
        self._safe_commands = {
            item.strip()
            for item in (
                *DEFAULT_SAFE_COMMAND_ALLOWLIST,
                *self.policy.safe_command_allowlist,
            )
            if item.strip()
        }
        self._safe_command_patterns = tuple(
            pattern.strip()
            for pattern in self.policy.safe_command_pattern_allowlist
            if pattern.strip()
        )

    async def validate(self, command: str) -> CommandSafetyDecision:
        """Validate command and return allow/block decision."""
        blocked_matches = self._match_rules(command, ABSOLUTE_BLOCK_RULES)
        if blocked_matches:
            descriptions = [rule.description for rule in blocked_matches]
            return CommandSafetyDecision(
                allowed=False,
                stage="blacklist",
                risk_level="critical",
                message=f"Blocked by hard safety rule: {descriptions[0]}",
                matched_rules=descriptions,
            )

        if self._is_allowlisted_safe_command(command):
            return CommandSafetyDecision(
                allowed=True,
                stage="allowlist",
                risk_level="low",
                message="Command matched safe allowlist and skipped risk review",
            )

        potential_matches = self._match_rules(command, POTENTIAL_RISK_RULES)
        if not potential_matches:
            return CommandSafetyDecision(
                allowed=True,
                stage="allow",
                risk_level="low",
                message="Command passed safety checks",
            )

        matched_descriptions = [rule.description for rule in potential_matches]
        if self.risk_evaluator is None:
            if self.policy.block_when_evaluator_missing:
                return CommandSafetyDecision(
                    allowed=False,
                    stage="llm",
                    risk_level="unknown",
                    message=(
                        "Command flagged as potentially risky and no risk evaluator is configured"
                    ),
                    matched_rules=matched_descriptions,
                )
            return CommandSafetyDecision(
                allowed=True,
                stage="allow",
                risk_level="medium",
                message="Potential risk detected but allowed without evaluator",
                matched_rules=matched_descriptions,
            )

        request = CommandRiskEvaluationRequest(
            command=command, matched_rules=matched_descriptions
        )
        raw_evaluation = self.risk_evaluator(request)
        if inspect.isawaitable(raw_evaluation):
            raw_evaluation = await raw_evaluation

        evaluation = self._normalize_evaluation(raw_evaluation)
        if self._is_block_level(evaluation.risk_level, self.policy.llm_block_threshold):
            return CommandSafetyDecision(
                allowed=False,
                stage="llm",
                risk_level=evaluation.risk_level,
                message=f"Blocked by LLM risk evaluation: {evaluation.summary}",
                matched_rules=matched_descriptions,
                evaluation=evaluation,
            )

        return CommandSafetyDecision(
            allowed=True,
            stage="allow",
            risk_level=evaluation.risk_level,
            message=f"Allowed by LLM risk evaluation: {evaluation.summary}",
            matched_rules=matched_descriptions,
            evaluation=evaluation,
        )

    def _is_allowlisted_safe_command(self, command: str) -> bool:
        if not self.policy.enable_safe_command_whitelist:
            return False
        if self._has_shell_operators(command):
            return False

        try:
            tokens = shlex.split(command)
        except ValueError:
            return False
        if not tokens:
            return False

        program = tokens[0]
        if program == "git":
            if len(tokens) < 2:
                return False
            return tokens[1] in SAFE_GIT_SUBCOMMANDS

        if program in self._safe_commands:
            return True
        return self._matches_pattern_allowlist(tokens)

    def _matches_pattern_allowlist(self, tokens: list[str]) -> bool:
        for pattern in self._safe_command_patterns:
            if self._matches_one_pattern(tokens, pattern):
                return True
        return False

    @staticmethod
    def _matches_one_pattern(tokens: list[str], pattern: str) -> bool:
        try:
            pattern_tokens = shlex.split(pattern)
        except ValueError:
            return False
        if not pattern_tokens:
            return False

        trailing_wildcard = pattern_tokens[-1] == "*"
        if trailing_wildcard:
            prefix = pattern_tokens[:-1]
            if len(tokens) < len(prefix):
                return False
            for index, part in enumerate(prefix):
                if not fnmatch.fnmatchcase(tokens[index], part):
                    return False
            return True

        if len(tokens) != len(pattern_tokens):
            return False
        for token, part in zip(tokens, pattern_tokens):
            if not fnmatch.fnmatchcase(token, part):
                return False
        return True

    @staticmethod
    def _has_shell_operators(command: str) -> bool:
        operators = (";", "|", "&", ">", "<", "`", "$(", "\n")
        return any(op in command for op in operators)

    @staticmethod
    def _match_rules(command: str, rules: list[SecurityRule]) -> list[SecurityRule]:
        return [rule for rule in rules if rule.pattern.search(command)]

    @staticmethod
    def _normalize_evaluation(raw: RiskEvaluatorOutput) -> CommandRiskEvaluation:
        if isinstance(raw, CommandRiskEvaluation):
            return raw

        risk_level = CommandSafetyValidator._normalize_risk_level(
            raw.get("risk_level") or raw.get("level")
        )
        summary_raw = raw.get("summary") or raw.get("reason") or "No summary provided"
        summary = str(summary_raw)

        risks_raw = raw.get("possible_risks") or raw.get("risks") or []
        if isinstance(risks_raw, str):
            possible_risks = [risks_raw]
        elif isinstance(risks_raw, list):
            possible_risks = [str(item) for item in risks_raw]
        else:
            possible_risks = []

        recommended_raw = raw.get("recommended_action")
        recommended_action = (
            str(recommended_raw) if recommended_raw is not None else None
        )

        return CommandRiskEvaluation(
            risk_level=risk_level,
            summary=summary,
            possible_risks=possible_risks,
            recommended_action=recommended_action,
            raw=raw,
        )

    @staticmethod
    def _normalize_risk_level(value: Any) -> RiskLevel:
        normalized = str(value or "unknown").strip().lower()
        if normalized in {"low", "medium", "high", "critical"}:
            return normalized  # type: ignore[return-value]
        return "unknown"

    @staticmethod
    def _is_block_level(level: RiskLevel, threshold: RiskLevel) -> bool:
        order = {
            "low": 1,
            "medium": 2,
            "high": 3,
            "critical": 4,
            "unknown": 3,
        }
        return order[level] >= order[threshold]
