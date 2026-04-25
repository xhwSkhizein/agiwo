# Agent Runtime Audit Matrix

## Status Legend

- `SHIPPED`
- `SPEC_UPDATE`
- `CODE_GAP`

## Core Runtime

| Spec Item | Current Code | Status | Notes |
| --- | --- | --- | --- |
| Single persisted source of truth is `RunLog` | `agiwo/agent/models/log.py`, `agiwo/agent/storage/base.py` | `SHIPPED` | Canonical runtime record is append-only `RunLog` |
| `Run`, `StepRecord`, `AgentHooks` are no longer canonical runtime models | `agiwo/agent/`, `tests/` | `SHIPPED` | No compatibility layer retained |
| Stable public read surface is explicitly documented | `agiwo/agent/__init__.py`, `agiwo/agent/storage/serialization.py` | `SPEC_UPDATE` | Shipped read surface is `RunView`, `StepView`, replayed `AgentStreamItem`, and runtime-decision views; no standalone `TimelineView` |
| Latest runtime-decision state is queryable without raw entry scanning | `agiwo/agent/models/runtime_decision.py`, `agiwo/agent/storage/base.py`, `agiwo/agent/storage/sqlite.py` | `SHIPPED` | `termination`, `compaction`, `step_back`, and rollback all have replayed latest-state queries |
| Live stream facts can be rebuilt from `RunLog` | `agiwo/agent/models/stream.py`, `tests/agent/test_run_log_replay_parity.py` | `SHIPPED` | Replay parity covered for replayable event families |

## Scheduler

| Spec Item | Current Code | Status | Notes |
| --- | --- | --- | --- |
| Scheduler reads replayed runtime facts instead of duplicating source models | `agiwo/scheduler/runtime_facts.py` | `SHIPPED` | `RunView`, `StepView`, and runtime-decision state read through `RunLogStorage` |
| Scheduler can query latest runtime decisions | `agiwo/scheduler/runtime_facts.py`, `tests/scheduler/test_runtime_facts.py` | `SHIPPED` | Dedicated read surface added; no raw entry scanning in scheduler call sites |
| Scheduler runner remains understandable enough for follow-on work | `agiwo/scheduler/runner.py`, `agiwo/scheduler/runner_output.py` | `SPEC_UPDATE` | Large file remains, but output classification and periodic completion helpers are now extracted into a focused helper module |

## Console

| Spec Item | Current Code | Status | Notes |
| --- | --- | --- | --- |
| Console run/session queries use a `RunLog`-backed facade | `console/server/services/runtime/run_query_service.py`, `console/server/services/runtime/session_view_service.py` | `SHIPPED` | Routers do not replay raw entries directly |
| Console query facade exposes replayed runtime-decision state | `console/server/services/runtime/run_query_service.py`, `console/tests/test_run_query_service.py` | `SHIPPED` | Session/run detail consumers can request latest decision state through one service |
| Console replay semantics are parity-tested | `console/tests/test_runtime_replay_consistency.py` | `SHIPPED` | Query facade results are cross-checked against direct `RunLogStorage` replay |

## Open Items

- No generic `TimelineView` should be introduced unless a concrete external use case appears that cannot be expressed with `RunView`, `StepView`, replayed stream items, and runtime-decision views.
- `agiwo/scheduler/runner.py` is still the main runtime-maintainability hotspot; future changes should keep extracting pure decision logic instead of broad rewrites.
