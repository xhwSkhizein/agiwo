"""Parameter parsing and validation for BashTool."""

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParseError:
    """Sentinel for parameter parsing failures."""

    message: str


class BashParameterParser:
    """Parses and validates raw tool parameters into typed values."""

    @staticmethod
    def parse_bool(value: Any) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off"}:
                return False
        return None

    def parse_timeout(self, parameters: dict[str, Any]) -> float | None | ParseError:
        """Return parsed timeout, None, or a ParseError."""
        timeout_value = parameters.get("timeout")
        if timeout_value is None:
            return None
        # Reject boolean values explicitly
        if isinstance(timeout_value, bool):
            return ParseError("timeout must be a number")
        try:
            parsed = float(timeout_value)
        except (TypeError, ValueError):
            return ParseError("timeout must be a number")
        # Reject non-finite values (NaN, inf, -inf)
        if not math.isfinite(parsed):
            return ParseError("timeout must be a finite number")
        return parsed

    def parse_flag(self, parameters: dict[str, Any], *, key: str) -> bool | ParseError:
        """Return parsed boolean or a ParseError."""
        value = parameters.get(key)
        parsed = self.parse_bool(value)
        if value is not None and parsed is None:
            return ParseError(f"{key} must be a boolean")
        return bool(parsed)

    def parse_stdin(self, parameters: dict[str, Any]) -> str | None | ParseError:
        """Return parsed stdin value, None, or a ParseError."""
        stdin_value = parameters.get("stdin")
        if stdin_value is None:
            return None
        if not isinstance(stdin_value, str):
            return ParseError("stdin must be a string")
        return stdin_value

    def parse_modes(self, parameters: dict[str, Any]) -> tuple[bool, bool] | ParseError:
        """Return (background, use_pty) or a ParseError."""
        background = self.parse_flag(parameters, key="background")
        if isinstance(background, ParseError):
            return background

        use_pty = self.parse_flag(parameters, key="pty")
        if isinstance(use_pty, ParseError):
            return use_pty

        return background, use_pty
