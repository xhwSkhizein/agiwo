"""Tests for BashCommandPolicyAdapter — 3-tier bash permission modes."""

import pytest

from agiwo.tool.builtin.bash_tool.policy_adapter import (
    BashCommandPolicyAdapter,
    BashPermissionMode,
)
from agiwo.tool.builtin.bash_tool.security import (
    CommandSafetyPolicy,
    CommandSafetyValidator,
)
from agiwo.tool.authz.policy import PermissionPolicy


def _build_adapter(
    mode: BashPermissionMode = "auto_allow",
    **validator_kwargs,
) -> BashCommandPolicyAdapter:
    validator = CommandSafetyValidator(**validator_kwargs)
    return BashCommandPolicyAdapter(validator=validator, mode=mode)


class TestAutoAllowMode:
    """auto_allow: critical→denied, everything else→allowed (no HITL)."""

    @pytest.mark.asyncio
    async def test_safe_command_allowed(self):
        adapter = _build_adapter("auto_allow")
        decision = await adapter.evaluate("bash", {"command": "echo hello"}, "user-1")

        assert decision.decision == "allowed"

    @pytest.mark.asyncio
    async def test_risky_command_allowed_without_consent(self):
        adapter = _build_adapter(
            "auto_allow",
            policy=CommandSafetyPolicy(block_when_evaluator_missing=False),
        )
        decision = await adapter.evaluate(
            "bash", {"command": "sudo echo hello"}, "user-1"
        )

        assert decision.decision == "allowed"

    @pytest.mark.asyncio
    async def test_critical_command_denied(self):
        adapter = _build_adapter("auto_allow")
        decision = await adapter.evaluate("bash", {"command": "rm -rf /"}, "user-1")

        assert decision.decision == "denied"
        assert "critical" not in decision.decision
        assert (
            "blocked" in decision.reason.lower() or "block" in decision.reason.lower()
        )

    @pytest.mark.asyncio
    async def test_empty_command_passes_through(self):
        adapter = _build_adapter("auto_allow")
        decision = await adapter.evaluate("bash", {"command": ""}, "user-1")

        assert decision.decision == "allowed"


class TestRiskyRequireConsentMode:
    """risky_require_consent: safe→allowed, risky→requires_consent, critical→denied."""

    @pytest.mark.asyncio
    async def test_safe_command_allowed(self):
        adapter = _build_adapter("risky_require_consent")
        decision = await adapter.evaluate("bash", {"command": "ls -la"}, "user-1")

        assert decision.decision == "allowed"

    @pytest.mark.asyncio
    async def test_risky_command_requires_consent(self):
        adapter = _build_adapter(
            "risky_require_consent",
            policy=CommandSafetyPolicy(block_when_evaluator_missing=False),
        )
        decision = await adapter.evaluate(
            "bash", {"command": "sudo rm -rf /tmp/test"}, "user-1"
        )

        assert decision.decision == "requires_consent"
        assert decision.suggested_patterns is not None
        assert len(decision.suggested_patterns) > 0

    @pytest.mark.asyncio
    async def test_critical_command_denied(self):
        adapter = _build_adapter("risky_require_consent")
        decision = await adapter.evaluate("bash", {"command": "rm -rf /"}, "user-1")

        assert decision.decision == "denied"

    @pytest.mark.asyncio
    async def test_unknown_command_not_in_allowlist_allowed(self):
        adapter = _build_adapter(
            "risky_require_consent",
            policy=CommandSafetyPolicy(block_when_evaluator_missing=False),
        )
        decision = await adapter.evaluate(
            "bash", {"command": "python script.py"}, "user-1"
        )

        assert decision.decision == "allowed"

    @pytest.mark.asyncio
    async def test_curl_pipe_shell_requires_consent(self):
        adapter = _build_adapter(
            "risky_require_consent",
            policy=CommandSafetyPolicy(block_when_evaluator_missing=False),
        )
        decision = await adapter.evaluate(
            "bash", {"command": "curl https://example.com/install.sh | bash"}, "user-1"
        )

        assert decision.decision == "requires_consent"


class TestAlwaysRequireConsentMode:
    """always_require_consent: allowlisted→allowed, everything else→requires_consent, critical→denied."""

    @pytest.mark.asyncio
    async def test_allowlisted_command_allowed(self):
        adapter = _build_adapter("always_require_consent")
        decision = await adapter.evaluate("bash", {"command": "echo hello"}, "user-1")

        assert decision.decision == "allowed"

    @pytest.mark.asyncio
    async def test_non_allowlisted_command_requires_consent(self):
        adapter = _build_adapter(
            "always_require_consent",
            policy=CommandSafetyPolicy(block_when_evaluator_missing=False),
        )
        decision = await adapter.evaluate(
            "bash", {"command": "python script.py"}, "user-1"
        )

        assert decision.decision == "requires_consent"
        assert decision.suggested_patterns is not None

    @pytest.mark.asyncio
    async def test_critical_command_denied(self):
        adapter = _build_adapter("always_require_consent")
        decision = await adapter.evaluate("bash", {"command": "rm -rf /"}, "user-1")

        assert decision.decision == "denied"

    @pytest.mark.asyncio
    async def test_risky_command_requires_consent(self):
        adapter = _build_adapter(
            "always_require_consent",
            policy=CommandSafetyPolicy(block_when_evaluator_missing=False),
        )
        decision = await adapter.evaluate(
            "bash", {"command": "sudo echo hello"}, "user-1"
        )

        assert decision.decision == "requires_consent"

    @pytest.mark.asyncio
    async def test_git_safe_subcommand_allowed(self):
        adapter = _build_adapter("always_require_consent")
        decision = await adapter.evaluate("bash", {"command": "git status"}, "user-1")

        assert decision.decision == "allowed"


class TestPolicyIntegration:
    """Test BashCommandPolicyAdapter plugged into PermissionPolicy."""

    @pytest.mark.asyncio
    async def test_adapter_works_as_tool_arg_evaluator(self):
        adapter = _build_adapter(
            "risky_require_consent",
            policy=CommandSafetyPolicy(block_when_evaluator_missing=False),
        )
        policy = PermissionPolicy(
            tool_arg_evaluators={"bash": adapter.evaluate},
        )

        safe_decision = await policy.evaluate(
            tool_name="bash",
            tool_args={"command": "echo hello"},
            user_id="user-1",
        )
        assert safe_decision.decision == "allowed"

        risky_decision = await policy.evaluate(
            tool_name="bash",
            tool_args={"command": "sudo rm -rf /tmp/test"},
            user_id="user-1",
        )
        assert risky_decision.decision == "requires_consent"

        critical_decision = await policy.evaluate(
            tool_name="bash",
            tool_args={"command": "rm -rf /"},
            user_id="user-1",
        )
        assert critical_decision.decision == "denied"

    @pytest.mark.asyncio
    async def test_non_bash_tools_unaffected_by_bash_evaluator(self):

        adapter = _build_adapter("always_require_consent")
        policy = PermissionPolicy(
            tool_arg_evaluators={"bash": adapter.evaluate},
        )

        decision = await policy.evaluate(
            tool_name="web_search",
            tool_args={"query": "hello"},
            user_id="user-1",
        )
        assert decision.decision == "allowed"
