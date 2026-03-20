"""Build bash tool permission policy with consent support."""

from agiwo.tool.authz import PermissionPolicy
from agiwo.tool.builtin.bash_tool import (
    BashCommandPolicyAdapter,
    BashPermissionMode,
)
from agiwo.tool.builtin.bash_tool.security import (
    CommandSafetyPolicy,
    CommandSafetyValidator,
)


def build_bash_permission_policy(
    mode: BashPermissionMode = "risky_require_consent",
) -> PermissionPolicy:
    """Build a PermissionPolicy with bash command-level authorization.
    
    Args:
        mode: Permission mode for bash commands
            - "auto_allow": only critical commands denied, no consent needed
            - "risky_require_consent": risky commands require user consent (default)
            - "always_require_consent": all non-allowlisted commands require consent
    
    Returns:
        PermissionPolicy configured with bash command adapter
    """
    validator = CommandSafetyValidator(
        policy=CommandSafetyPolicy(
            block_when_evaluator_missing=False,
            llm_block_threshold="high",
        )
    )
    
    adapter = BashCommandPolicyAdapter(validator=validator, mode=mode)
    
    return PermissionPolicy(
        tool_arg_evaluators={"bash": adapter.evaluate},
    )


__all__ = ["build_bash_permission_policy"]
