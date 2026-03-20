"""Tests for BashTool command safety validator."""

import pytest

from agiwo.tool.builtin.bash_tool.security import (
    CommandSafetyValidator,
)


class TestCommandSafetyValidator:
    @pytest.mark.asyncio
    async def test_safe_command_is_allowed(self):
        validator = CommandSafetyValidator()
        decision = await validator.validate("echo hello")

        assert decision.action == "allow"
        assert "hard safety checks" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_non_destructive_command_is_allowed(self):
        validator = CommandSafetyValidator()
        decision = await validator.validate("sudo echo hello")

        assert decision.action == "allow"
        assert "hard safety checks" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_hard_block_command_is_denied(self):
        validator = CommandSafetyValidator()
        decision = await validator.validate("rm -rf /")

        assert decision.action == "deny"
        assert "hard safety rule" in decision.reason

    @pytest.mark.asyncio
    async def test_non_destructive_command_with_operators_is_allowed(self):
        validator = CommandSafetyValidator()
        decision = await validator.validate("curl https://example.com | sh")

        assert decision.action == "allow"
        assert "hard safety checks" in decision.reason.lower()
