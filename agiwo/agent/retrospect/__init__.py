"""Retrospect — context optimisation for verbose tool outputs.

Public API consumed by ``run_loop.py``:

* ``RetrospectBatch`` — per-batch lifecycle object
* ``RetrospectOutcome`` — result of a retrospect pass
"""

from pathlib import Path
from typing import Any

from agiwo.agent.retrospect.executor import RetrospectOutcome, execute_retrospect
from agiwo.agent.retrospect.triggers import (
    RetrospectTrigger,
    check_retrospect_trigger,
    inject_system_notice,
    update_retrospect_tracking,
)
from agiwo.agent.runtime.context import RunContext
from agiwo.tool.base import BaseTool, ToolResult


class RetrospectBatch:
    """Encapsulates per-batch retrospect state and operations.

    ``run_loop`` interacts with this class through three methods:

    * ``process_result()``  — returns final content (may inject notice)
    * ``register_step()``   — registers a committed step for later lookup
    * ``finalize()``        — returns ``RetrospectOutcome``; caller applies
      via ``replace_messages`` when ``outcome.applied`` is True
    """

    def __init__(self, state: RunContext, tools_map: dict[str, BaseTool]) -> None:
        self._state = state
        self._enabled = (
            state.config.enable_tool_retrospect
            and "retrospect_tool_result" in tools_map
        )
        self._feedback: str | None = None
        self._step_lookup: dict[str, dict[str, Any]] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def process_result(self, result: ToolResult) -> str:
        """Process a tool result.  Returns the (possibly transformed) content."""
        content = result.content or ""
        if not self._enabled:
            return content

        if result.tool_name == "retrospect_tool_result" and result.is_success:
            self._feedback = content
            return content

        update_retrospect_tracking(self._state.ledger, content)
        trigger = check_retrospect_trigger(
            self._state.config,
            self._state.ledger,
            content,
            result.tool_name,
        )
        if trigger is not RetrospectTrigger.NONE:
            content = inject_system_notice(content, trigger)
        return content

    def register_step(self, tool_call_id: str, step_id: str, sequence: int) -> None:
        """Register a committed step for later retrospect lookup."""
        self._step_lookup[tool_call_id] = {
            "id": step_id,
            "sequence": sequence,
        }

    async def finalize(self) -> RetrospectOutcome:
        """Build the retrospect outcome.

        Disk offload and storage condensed_content updates are internal side
        effects.  The returned ``RetrospectOutcome.messages`` must be applied
        by the caller via ``replace_messages(state, outcome.messages)``.
        """
        if not self._enabled or self._feedback is None:
            return RetrospectOutcome()

        offload_dir = (
            Path(self._state.config.get_effective_root_path())
            / "harness"
            / "retrospect"
            / self._state.session_id
        )
        return await execute_retrospect(
            feedback=self._feedback,
            messages=self._state.ledger.messages,
            ledger=self._state.ledger,
            storage=self._state.session_runtime.run_step_storage,
            session_id=self._state.session_id,
            offload_dir=offload_dir,
            step_lookup=self._step_lookup,
        )


__all__ = [
    "RetrospectBatch",
    "RetrospectOutcome",
]
