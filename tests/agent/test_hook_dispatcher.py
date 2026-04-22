from dataclasses import dataclass

import pytest

import agiwo.agent.runtime.state_writer as state_writer_module
from agiwo.agent.hooks import (
    HookCapability,
    HookGroup,
    HookPhase,
    HookRegistration,
    HookRegistry,
    observe,
    transform,
)
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
    assert registration.group is HookGroup.USER
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
async def test_hook_registry_orders_group_then_order_then_registration() -> None:
    seen: list[str] = []

    def handler(name: str):
        async def _run(payload: dict) -> dict:
            seen.append(name)
            return payload

        return _run

    registry = HookRegistry(
        registrations=[
            HookRegistration(
                phase=HookPhase.BEFORE_LLM,
                group=HookGroup.USER,
                capability=HookCapability.TRANSFORM,
                handler_name="user_second",
                handler=handler("user_second"),
                order=200,
            ),
            HookRegistration(
                phase=HookPhase.BEFORE_LLM,
                group=HookGroup.SYSTEM,
                capability=HookCapability.TRANSFORM,
                handler_name="system_first",
                handler=handler("system_first"),
                order=100,
            ),
            HookRegistration(
                phase=HookPhase.BEFORE_LLM,
                group=HookGroup.RUNTIME_ADAPTER,
                capability=HookCapability.TRANSFORM,
                handler_name="adapter_middle",
                handler=handler("adapter_middle"),
                order=100,
            ),
        ]
    )

    await registry._dispatch(
        HookPhase.BEFORE_LLM,
        {
            "messages": [{"role": "user", "content": "hi"}],
            "context": object(),
            "model_settings_override": None,
            "llm_advice": None,
        },
        allow_transform=True,
    )

    assert seen == ["system_first", "adapter_middle", "user_second"]


@pytest.mark.asyncio
async def test_hook_registry_rejects_transform_fields_outside_phase_allowlist() -> None:
    async def bad_transform(payload: dict) -> dict:
        updated = dict(payload)
        updated["illegal_field"] = "boom"
        return updated

    registry = HookRegistry(
        registrations=[transform(HookPhase.BEFORE_TOOL_CALL, "bad", bad_transform)]
    )

    with pytest.raises(ValueError, match="illegal_field"):
        await registry._dispatch(
            HookPhase.BEFORE_TOOL_CALL,
            {
                "tool_call_id": "call-1",
                "tool_name": "bash",
                "parameters": {"cmd": "pwd"},
                "context": object(),
                "tool_advice": None,
            },
            allow_transform=True,
        )


def test_hook_registry_rejects_critical_after_phase() -> None:
    async def after_llm(payload: dict) -> None:
        del payload

    with pytest.raises(ValueError, match="Critical hooks"):
        HookRegistry(
            registrations=[
                observe(
                    HookPhase.AFTER_LLM,
                    "after_llm",
                    after_llm,
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


@pytest.mark.asyncio
async def test_hook_registry_routes_hook_failed_recording_through_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def broken_handler(payload: dict) -> None:
        del payload
        raise RuntimeError("boom")

    recorded: dict[str, object] = {}

    class _SpyWriter:
        def __init__(self, state: object) -> None:
            recorded["state"] = state

        async def record_hook_failed(self, **kwargs: object) -> list[object]:
            recorded["kwargs"] = kwargs
            return []

    @dataclass
    class _FakeContext:
        session_runtime: object

    monkeypatch.setattr(state_writer_module, "RunStateWriter", _SpyWriter)
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
        {"context": _FakeContext(session_runtime=object()), "step": object()},
        allow_transform=False,
    )

    assert "state" in recorded
    kwargs = recorded["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["phase"] == HookPhase.AFTER_LLM.value
    assert kwargs["handler_name"] == "broken"
    assert kwargs["critical"] is False
    assert kwargs["error"] == "boom"
    assert kwargs["traceback"] is not None
