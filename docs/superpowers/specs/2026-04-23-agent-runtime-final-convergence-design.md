# Agent Runtime Final Convergence Design

**Goal:** Finish the strict convergence of `agiwo.agent` so `RunStateWriter` is the only committed runtime-truth write path, `RunLog` is the only replayable truth source, and no legacy trace or message-mutation path remains in production.

**Architecture:** Keep `RunLoopOrchestrator` as the only phase decider. Move every committed runtime mutation, including remaining bootstrap and steer-adjacent state writes, behind `RunStateWriter`. Delete the legacy direct-trace callback path and make live trace, replay trace, and replayable stream/query views all consume committed `RunLog` entries only.

**This document is a follow-up to:** [2026-04-22-agent-runtime-strict-convergence-design.md](./2026-04-22-agent-runtime-strict-convergence-design.md)

## Why Another Follow-up

PR #76 completed most of the strict convergence work, but review of the merged implementation still found four remaining gaps:

1. accepted steer input can be drained before any committed runtime fact is written
2. `AgentTraceCollector` still contains a legacy direct-callback construction path
3. a few committed runtime fields are still mutated outside `RunStateWriter`
4. internal LLM turns and live trace persistence are not yet fully aligned with the canonical replay model

This follow-up closes those gaps without preserving backward compatibility.

## Scope

This pass covers only the remaining convergence work inside `agiwo.agent` and its immediate query/observability consumers.

This pass does not cover:

1. unrelated scheduler semantics changes
2. new public product features
3. migration layers for old internal APIs

## Final Contract

The final runtime contract after this pass is:

1. `RunLoopOrchestrator` decides runtime phases
2. `RunStateWriter` is the only path allowed to mutate committed runtime truth
3. committed runtime truth means:
   - mutable in-memory committed runtime state
   - typed append-only `RunLog` entries
4. `SessionRuntime` may:
   - allocate sequences
   - append committed entries
   - project views from committed entries
5. `SessionRuntime` may not:
   - mutate committed runtime truth directly
   - synthesize replayable facts independently of committed entries
6. `TraceBuilder`, `StreamBuilder`, and query builders are pure `RunLog` consumers
7. `StepDeltaEvent` remains the only live-only transport exception
8. no legacy trace callback path remains in production code

## Problem Details

### 1. Steer Is Not Yet Lossless

`apply_steering_messages(...)` drains `steering_queue` before the runtime commits any canonical fact describing the consumed steer input.

If a fatal `before_llm` hook failure or writer/storage failure happens after queue drain but before commit, the accepted steer input disappears without a matching committed runtime fact.

That violates the strict truth contract.

### 2. Trace Still Has A Dual Architecture

`AgentTraceCollector` now supports committed-entry projection, but it still exposes and implements:

1. `on_run_started(...)`
2. `on_step(...)`
3. `on_run_completed(...)`
4. `on_run_failed(...)`

Those methods build trace state independently of committed `RunLog` entries. Keeping them around preserves a second architecture and invites future drift.

### 3. Remaining Writer Bypasses Still Exist

The merged code still mutates some committed runtime state outside `RunStateWriter`, including:

1. bootstrap tool schema state
2. bootstrap compaction metadata state
3. bootstrap `run_start_seq`
4. compaction failure counters

These are small, but they still violate the strict rule.

### 4. Internal LLM Turns Are Not Fully Canonical

Normal assistant turns write `LLMCallStarted` and `LLMCallCompleted`.

Internal LLM turns such as:

1. `compaction`
2. termination summary generation

still commit steps but do not consistently commit canonical LLM-call facts through the same path.

That makes replayed trace structure weaker than live runtime intent.

### 5. Live Trace Persistence Is Too Deferred

`on_run_log_entries(...)` updates in-memory trace state but does not persist the updated trace snapshot immediately.

For long-lived persistent runtimes, console trace queries can lag behind the actual committed runtime facts until close/final stop.

## Design

### A. Lossless Steer Consumption

Steer handling becomes a staged commit flow.

The runtime must distinguish between:

1. pending steer inputs not yet committed into the current request state
2. committed message-state rewrites that now include steer input

The new rule is:

1. reading pending steer input is allowed
2. draining/removing it from pending state is not considered complete until the resulting message rewrite is committed through `RunStateWriter`

Implementation shape:

1. replace the current destructive queue-drain helper with a helper that stages pending steer inputs without final consumption
2. apply `before_llm` transforms to the staged message set
3. if the staged message set differs from current committed messages, commit it through `RunStateWriter` as `MessagesRebuilt(reason="before_llm")`
4. only after that commit succeeds may the staged steer items be considered consumed
5. if commit fails, the steer input must remain recoverable for a future retry in the same session runtime

This keeps accepted steer input lossless.

### B. `RunStateWriter` Owns The Remaining Runtime Mutations

`RunStateWriter` gains explicit ownership over the remaining committed-state fields that still bypass it.

That includes:

1. tool schema state for the active run
2. latest compaction metadata state
3. `run_start_seq`
4. compaction failure count updates

The rule is:

1. if a field influences later committed runtime behavior or replay semantics, it belongs to writer-owned committed runtime state
2. direct mutation helpers may remain only for explicitly ephemeral state that is documented as non-canonical

This pass does not introduce new ambiguous middle ground.

### C. Internal LLM Turns Become Canonical LLM Facts

Internal LLM turns that are part of the runtime contract must use the same writer-mediated LLM fact pipeline as ordinary assistant turns.

That means `compaction` and termination summary generation must commit:

1. `LLMCallStarted`
2. committed step facts
3. `LLMCallCompleted`

using the same canonical sequence:

1. writer commits request fact
2. stream delta remains live-only if applicable
3. final committed assistant step goes through writer
4. final LLM completion fact goes through writer
5. trace and replayable stream project from those entries

No special hidden trace-only path is allowed.

### D. Trace Becomes Entry-Only

`AgentTraceCollector` keeps only:

1. `start(...)`
2. `on_run_log_entries(...)`
3. `build_from_entries(...)`
4. `stop()`

The legacy direct-callback methods are removed.

Tests and callers must build trace state only from committed entries.

The collector must also manage projection state with bounded memory:

1. committed assistant-tool-call cache gets an explicit cap, matching the previous legacy cache safety level
2. `LLMCallStarted` correlation state is cleared once a matching completion is processed or the run reaches terminal state

### E. Live Trace Persistence Tracks Committed Facts

Whenever committed entries are applied to the live trace, the updated trace snapshot must be persisted frequently enough that console queries observe current committed runtime truth during long-lived sessions.

The required rule is:

1. after a committed entry batch is applied to the live trace, the collector persists the updated trace snapshot
2. this persistence remains best-effort and may warn on failure without breaking the runtime
3. replay and live query surfaces must not depend on session close to become accurate

This keeps trace storage aligned with persistent scheduler-managed runtimes.

### F. Query / Debug Visibility

First-class runtime facts must remain practically observable.

For this pass:

1. `HookFailed` remains a committed first-class `RunLog` fact
2. it must remain replayable from storage and available to runtime/debug query flows
3. no separate compatibility event layer is introduced just to preserve old paths

This pass does not require adding a new end-user public stream event for `HookFailed` unless implementation naturally reuses an existing runtime-debug view.

## Required Deletions

This pass explicitly deletes:

1. legacy direct trace callback construction paths
2. production code paths that mutate committed runtime truth before writer commit
3. hidden internal LLM call handling that bypasses canonical LLM facts

Backward compatibility for those internal paths is out of scope.

## File-Level Impact

Primary modules expected to change:

1. `agiwo/agent/run_loop.py`
2. `agiwo/agent/runtime/state_writer.py`
3. `agiwo/agent/runtime/session.py`
4. `agiwo/agent/trace_writer.py`
5. `agiwo/agent/run_bootstrap.py`
6. `agiwo/agent/compaction.py`
7. `agiwo/agent/termination/summarizer.py`
8. `agiwo/agent/prompt.py` or a replacement runtime helper if steer staging is moved out of prompt helpers
9. `tests/agent/test_run_loop_contracts.py`
10. `tests/agent/test_run_engine.py`
11. `tests/observability/test_collector.py`
12. `tests/agent/test_run_log_replay_parity.py`
13. `console/tests/test_runtime_replay_consistency.py`

## Error Handling Rules

### Steer Failure

If staging or committing steer-derived message rewrites fails:

1. no silent steer loss is allowed
2. the pending steer input remains recoverable
3. fatal run failure still writes `RunFailed` if the run had already started

### Trace Persistence Failure

If trace snapshot persistence fails:

1. committed runtime execution continues
2. warning-level logging is emitted
3. replay from `RunLog` remains the correctness fallback

### Internal LLM Failure

If an internal LLM turn fails:

1. any already-committed facts remain canonical
2. fatal propagation still ends in `RunFailed` where applicable
3. best-effort summary generation may still degrade gracefully, but it may not bypass canonical fact-writing for the facts it does commit

## Acceptance Criteria

This cleanup is complete only when all of the following are true:

1. accepted steer input cannot be silently lost before a committed runtime fact exists
2. `RunStateWriter` owns all committed runtime-truth mutations remaining in bootstrap, compaction, and related flows
3. internal canonical LLM turns write `LLMCallStarted` and `LLMCallCompleted`
4. production trace building uses only committed `RunLog` entry projection
5. legacy direct trace callback methods are removed from production code
6. live trace storage stays aligned with committed runtime facts during long-lived sessions
7. committed trace-correlation caches are memory-bounded
8. no production code path mutates committed runtime truth outside `RunStateWriter`
9. live replayable stream parity and live/replayed trace parity still hold

## Testing Strategy

The implementation must add or update tests at four layers:

1. unit tests
   - writer-owned state mutations
   - steer staging semantics
   - trace cache cleanup and bounds
2. runtime integration tests
   - `before_llm` failure after accepted steer input
   - compaction internal LLM facts
   - termination summary internal LLM facts
3. parity tests
   - live stream vs replay stream
   - live trace vs replay trace
4. guardrail tests
   - fail if production code still uses legacy trace callback APIs
   - fail if known committed-state bypass points reappear

## Non-Goals

This pass does not:

1. redesign scheduler semantics
2. add new product-facing observability UIs beyond what the cleaned runtime contract requires
3. keep deprecated internal APIs alive

## Final Statement

After this pass, the runtime contract is intentionally simple:

1. orchestrator decides
2. writer commits runtime truth
3. session runtime appends and projects
4. trace/stream/query consume committed entries

That is the final strict-convergence end state for the current runtime architecture.
