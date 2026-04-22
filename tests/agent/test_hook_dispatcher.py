from dataclasses import dataclass

import pytest

from agiwo.agent.hooks import HookCapability, HookPhase, HookRegistration, HookRegistry
from agiwo.agent.models.log import HookFailed
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.storage.base import InMemoryRunLogStorage


def test_hook_registration_exposes_phase_and_capability() -> None:
    async def rewrite_messages(payload: dict) -> dict:
        return payload

    registration = HookRegistration(
        phase=HookPhase.BEFORE_LLM,
        capability=HookCapability.TRANSFORM,
        handler_name="rewrite_messages",
        handler=rewrite_messages,
    )

    assert registration.phase is HookPhase.BEFORE_LLM
    assert registration.capability is HookCapability.TRANSFORM
    assert registration.critical is False


def test_hook_registry_validates_constructor_registrations() -> None:
    async def observe_finalize(payload: dict) -> None:
        del payload

    with pytest.raises(ValueError):
        HookRegistry(
            registrations=[
                HookRegistration(
                    phase=HookPhase.RUN_FINALIZED,
                    capability=HookCapability.OBSERVE_ONLY,
                    handler_name="observe_finalize",
                    handler=observe_finalize,
                    critical=True,
                )
            ]
        )


@pytest.mark.asyncio
async def test_hook_registry_records_hook_failed_entries_for_noncritical_errors() -> (
    None
):
    async def broken_handler(payload: dict) -> None:
        del payload
        raise RuntimeError("boom")

    @dataclass
    class _FakeContext:
        session_id: str
        run_id: str
        agent_id: str
        session_runtime: SessionRuntime

    storage = InMemoryRunLogStorage()
    session_runtime = SessionRuntime(
        session_id="sess-1",
        run_log_storage=storage,
    )
    context = _FakeContext(
        session_id="sess-1",
        run_id="run-1",
        agent_id="agent-1",
        session_runtime=session_runtime,
    )
    registry = HookRegistry(
        registrations=[
            HookRegistration(
                phase=HookPhase.AFTER_LLM,
                capability=HookCapability.OBSERVE_ONLY,
                handler_name="broken",
                handler=broken_handler,
            )
        ]
    )

    await registry._dispatch(
        HookPhase.AFTER_LLM,
        {"context": context, "step": object()},
        allow_transform=False,
    )

    entries = await storage.list_entries(session_id="sess-1")
    assert len(entries) == 1
    hook_failed = entries[0]
    assert isinstance(hook_failed, HookFailed)
    assert hook_failed.phase == HookPhase.AFTER_LLM.value
    assert hook_failed.handler_name == "broken"
    assert hook_failed.critical is False
    assert hook_failed.error == "boom"
    assert hook_failed.traceback is not None
