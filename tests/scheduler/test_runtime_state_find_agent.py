"""RuntimeState agent-resolution helpers."""

from types import SimpleNamespace

from agiwo.scheduler.runtime_state import RuntimeState


def test_find_agent_by_registry_key() -> None:
    runtime = SimpleNamespace(id="agent-runtime")
    canonical = SimpleNamespace(id="agent-canonical")
    rt = RuntimeState(
        agents={"state-1": runtime},  # type: ignore[dict-item]
        canonical_agents={"state-1": canonical},  # type: ignore[dict-item]
    )
    assert rt.find_agent("state-1") is runtime


def test_find_agent_falls_back_to_agent_id_scan() -> None:
    runtime = SimpleNamespace(id="agent-a")
    rt = RuntimeState(
        agents={"other-key": runtime},  # type: ignore[dict-item]
    )
    assert rt.find_agent("agent-a") is runtime
    assert rt.find_agent("missing") is None
