from agiwo.agent.hooks import HookCapability, HookPhase, HookRegistration


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
