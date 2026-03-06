"""Tests for BashTool command safety validator."""

import pytest

from agiwo.tool.builtin.bash_tool.security import (
    CommandRiskEvaluationRequest,
    CommandSafetyPolicy,
    CommandSafetyValidator,
)


class TestCommandSafetyValidator:
    @pytest.mark.asyncio
    async def test_allowlisted_command_is_allowed(self):
        validator = CommandSafetyValidator()
        decision = await validator.validate("echo hello")

        assert decision.allowed is True
        assert decision.stage == "allowlist"
        assert decision.risk_level == "low"

    @pytest.mark.asyncio
    async def test_potential_risk_blocks_without_evaluator(self):
        validator = CommandSafetyValidator()
        decision = await validator.validate("sudo echo hello")

        assert decision.allowed is False
        assert decision.stage == "llm"
        assert decision.risk_level == "unknown"
        assert "potentially risky" in decision.message

    @pytest.mark.asyncio
    async def test_potential_risk_allows_with_low_llm_result(self):
        async def evaluator(_request: CommandRiskEvaluationRequest):
            return {
                "risk_level": "low",
                "summary": "safe in this context",
            }

        validator = CommandSafetyValidator(risk_evaluator=evaluator)
        decision = await validator.validate("sudo echo hello")

        assert decision.allowed is True
        assert decision.stage == "allow"
        assert decision.risk_level == "low"

    @pytest.mark.asyncio
    async def test_potential_risk_blocks_with_high_llm_result(self):
        async def evaluator(_request: CommandRiskEvaluationRequest):
            return {
                "risk_level": "high",
                "summary": "high risk",
            }

        validator = CommandSafetyValidator(risk_evaluator=evaluator)
        decision = await validator.validate("sudo echo hello")

        assert decision.allowed is False
        assert decision.stage == "llm"
        assert decision.risk_level == "high"

    @pytest.mark.asyncio
    async def test_pattern_allowlist_allows_matching_command(self):
        validator = CommandSafetyValidator(
            policy=CommandSafetyPolicy(
                safe_command_pattern_allowlist=("python -m pytest *",),
            )
        )

        decision = await validator.validate("python -m pytest tests/tool/test_bash_tool.py")

        assert decision.allowed is True
        assert decision.stage == "allowlist"
