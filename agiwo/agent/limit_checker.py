"""
Execution Limit Checker - Validates execution constraints.
"""

import time
from typing import TYPE_CHECKING

from agiwo.agent.base import AgentConfigOptions


if TYPE_CHECKING:
    from agiwo.agent.executor import RunState


class ExecutionLimitChecker:
    """执行限制检查器"""

    def __init__(self, config: AgentConfigOptions):
        self.config = config

    def check_limits(self, state: "RunState") -> str | None:
        """
        检查所有执行限制

        Returns:
            终止原因，如果未达到限制则返回 None
        """
        # 步数限制
        if state.current_step >= self.config.max_steps:
            return "max_steps"

        # 运行时间限制
        if self.config.run_timeout and state.elapsed > self.config.run_timeout:
            return "timeout"

        # 上下文级超时
        if state.context.timeout_at and time.time() >= state.context.timeout_at:
            return "timeout"

        # Token 限制
        if (
            self.config.max_output_tokens
            and state.tracker.total_tokens >= self.config.max_output_tokens
        ):
            return "max_tokens"

        return None
