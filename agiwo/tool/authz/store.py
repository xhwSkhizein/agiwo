import fnmatch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass(frozen=True)
class ConsentRecord:
    user_id: str
    tool_name: str | None = None
    patterns: list[str] = field(default_factory=list)
    deny_patterns: list[str] = field(default_factory=list)
    expires_at: datetime | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class ConsentDecision:
    decision: Literal["allow", "deny"]
    patterns: list[str] = field(default_factory=list)
    expires_at: datetime | None = None


class ConsentStore(ABC):
    @abstractmethod
    async def check_consent(
        self,
        user_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> Literal["allowed", "denied", None]: ...

    @abstractmethod
    async def save_consent(
        self,
        user_id: str,
        tool_name: str | None,
        patterns: list[str],
        deny_patterns: list[str] | None = None,
        expires_at: datetime | None = None,
    ) -> None: ...

    def match_pattern(
        self,
        pattern: str,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> bool:
        if not pattern.endswith(")"):
            return False
        open_paren = pattern.find("(")
        if open_paren == -1:
            return False
        pattern_tool_name = pattern[:open_paren]
        arg_pattern = pattern[open_paren + 1 : -1]
        if not fnmatch.fnmatch(tool_name, pattern_tool_name):
            return False
        return fnmatch.fnmatch(self._serialize_args(tool_args), arg_pattern)

    def _serialize_args(self, args: dict[str, Any]) -> str:
        items = sorted((k, v) for k, v in args.items() if k != "tool_call_id")
        return " ".join(f"{k}={v}" for k, v in items)


class InMemoryConsentStore(ConsentStore):
    def __init__(self) -> None:
        self._records: dict[str, ConsentRecord] = {}

    def _make_key(self, user_id: str, tool_name: str | None) -> str:
        return f"{user_id}:{tool_name or 'global'}"

    async def check_consent(
        self,
        user_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> Literal["allowed", "denied", None]:
        keys_to_check = [
            self._make_key(user_id, tool_name),
            self._make_key(user_id, None),
        ]
        for key in keys_to_check:
            record = self._records.get(key)
            if record is None:
                continue
            if record.expires_at and record.expires_at < datetime.now():
                continue
            for deny_pattern in record.deny_patterns:
                if self.match_pattern(deny_pattern, tool_name, tool_args):
                    return "denied"
            for pattern in record.patterns:
                if self.match_pattern(pattern, tool_name, tool_args):
                    return "allowed"
        return None

    async def save_consent(
        self,
        user_id: str,
        tool_name: str | None,
        patterns: list[str],
        deny_patterns: list[str] | None = None,
        expires_at: datetime | None = None,
    ) -> None:
        key = self._make_key(user_id, tool_name)
        self._records[key] = ConsentRecord(
            user_id=user_id,
            tool_name=tool_name,
            patterns=patterns,
            deny_patterns=[] if deny_patterns is None else deny_patterns,
            expires_at=expires_at,
        )
