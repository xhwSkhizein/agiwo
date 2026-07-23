"""JSON serialization for SessionIntent persistence."""

from dataclasses import asdict
from typing import Any

from agiwo.agent.intent.models import IntentEntry, SessionIntent
from agiwo.agent.models.plan import Milestone, RunPlan


def serialize_milestone(milestone: Milestone) -> dict[str, Any]:
    return asdict(milestone)


def deserialize_milestone(data: dict[str, Any]) -> Milestone:
    return Milestone(**data)


def serialize_run_plan(plan: RunPlan) -> dict[str, Any]:
    return {
        "milestones": [serialize_milestone(item) for item in plan.milestones],
        "revision": plan.revision,
    }


def deserialize_run_plan(data: dict[str, Any]) -> RunPlan:
    milestones_raw = data.get("milestones", [])
    milestones: list[Milestone] = []
    if isinstance(milestones_raw, list):
        for item in milestones_raw:
            if isinstance(item, Milestone):
                milestones.append(item)
            elif isinstance(item, dict):
                milestones.append(deserialize_milestone(item))
    return RunPlan(milestones=milestones, revision=int(data.get("revision", 0)))


def serialize_intent_entry(entry: IntentEntry) -> dict[str, Any]:
    return {
        "kind": entry.kind,
        "text": entry.text,
        "at": entry.at,
        "run_id": entry.run_id,
    }


def deserialize_intent_entry(data: dict[str, Any]) -> IntentEntry:
    kind = data["kind"]
    if kind not in {"user_input", "run_report"}:
        raise ValueError(f"Invalid IntentEntry.kind: {kind!r}")
    return IntentEntry(
        kind=kind,
        text=str(data["text"]),
        at=int(data["at"]),
        run_id=data.get("run_id"),
    )


def serialize_session_intent(intent: SessionIntent) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "entries": [serialize_intent_entry(entry) for entry in intent.entries],
        "updated_at": intent.updated_at,
    }
    if intent.last_run_plan is not None:
        payload["last_run_plan"] = serialize_run_plan(intent.last_run_plan)
    else:
        payload["last_run_plan"] = None
    return payload


def deserialize_session_intent(data: dict[str, Any]) -> SessionIntent:
    entries_raw = data.get("entries", [])
    entries: list[IntentEntry] = []
    if isinstance(entries_raw, list):
        for item in entries_raw:
            if isinstance(item, IntentEntry):
                entries.append(item)
            elif isinstance(item, dict):
                entries.append(deserialize_intent_entry(item))
    last_run_plan_raw = data.get("last_run_plan")
    last_run_plan = None
    if isinstance(last_run_plan_raw, dict):
        last_run_plan = deserialize_run_plan(last_run_plan_raw)
    return SessionIntent(
        entries=entries,
        last_run_plan=last_run_plan,
        updated_at=int(data.get("updated_at", 0)),
    )


__all__ = [
    "deserialize_intent_entry",
    "deserialize_run_plan",
    "deserialize_session_intent",
    "serialize_intent_entry",
    "serialize_run_plan",
    "serialize_session_intent",
]
