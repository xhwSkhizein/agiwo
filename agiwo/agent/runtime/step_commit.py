"""Shared step-commit protocol for orchestrator-owned commit pipelines."""

from typing import Protocol

from agiwo.agent.models.step import LLMCallContext, StepView


class StepCommitter(Protocol):
    async def __call__(
        self,
        step: StepView,
        *,
        llm: LLMCallContext | None = None,
        append_message: bool = True,
        track_state: bool = True,
    ) -> StepView: ...


__all__ = ["StepCommitter"]
