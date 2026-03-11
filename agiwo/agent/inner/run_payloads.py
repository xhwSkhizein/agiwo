"""Shared payload helpers for run lifecycle events."""

from typing import Any

from agiwo.agent.schema import RunMetrics, RunOutput

_RUN_COMPLETED_METRIC_DEFAULTS: dict[str, int | float] = {
    "duration_ms": 0.0,
    "total_tokens": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_read_tokens": 0,
    "cache_creation_tokens": 0,
    "token_cost": 0.0,
    "steps_count": 0,
    "tool_calls_count": 0,
}


def serialize_run_metrics_payload(
    metrics: RunMetrics | None,
) -> dict[str, int | float]:
    """Serialize the stable metrics subset used in run-completed events."""
    if metrics is None:
        return dict(_RUN_COMPLETED_METRIC_DEFAULTS)

    payload = dict(_RUN_COMPLETED_METRIC_DEFAULTS)
    for field_name in payload:
        payload[field_name] = getattr(metrics, field_name, payload[field_name])
    return payload


def apply_run_metrics_payload(
    metrics: RunMetrics,
    payload: dict[str, Any] | None,
) -> None:
    """Apply a run-completed metrics payload onto an existing RunMetrics object."""
    if not payload:
        return

    for field_name, default in _RUN_COMPLETED_METRIC_DEFAULTS.items():
        setattr(metrics, field_name, payload.get(field_name, default))


def build_run_completed_event_data(result: RunOutput) -> dict[str, Any]:
    """Build the event payload emitted for a completed run."""
    data: dict[str, Any] = {
        "response": result.response or "",
        "metrics": serialize_run_metrics_payload(result.metrics),
    }
    if result.termination_reason is not None:
        data["termination_reason"] = result.termination_reason.value
    return data


__all__ = [
    "apply_run_metrics_payload",
    "build_run_completed_event_data",
    "serialize_run_metrics_payload",
]
