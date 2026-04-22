"""Helpers for dispatching hook-registry phases."""

from typing import Any

from agiwo.agent.hooks import HookPhase, HookRegistry


class HookDispatcher:
    """Thin dispatcher wrapper around a HookRegistry."""

    def __init__(self, registry: HookRegistry) -> None:
        self._registry = registry

    async def dispatch(
        self,
        phase: HookPhase,
        payload: dict[str, Any],
        *,
        allow_transform: bool = True,
    ) -> dict[str, Any]:
        return await self._registry._dispatch(
            phase,
            payload,
            allow_transform=allow_transform,
        )


__all__ = ["HookDispatcher"]
