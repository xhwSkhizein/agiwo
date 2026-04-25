# Agent Runtime Strict Convergence Design

**Goal:** Close the remaining architecture gaps after the first `RunLog` migration so `RunLog` becomes the complete runtime truth, replayable runtime facts are written through one path, and hook/trace/stream behavior is fully aligned with a stricter public contract.

**Architecture:** Keep `RunLoopOrchestrator` as the single phase decider and make `RunStateWriter` the only path that may mutate runtime truth. Runtime truth means committed in-memory runtime state plus canonical typed `RunLog` entries. `TraceBuilder` and `StreamBuilder` become pure consumers of committed `RunLog` entries, with `StepDeltaEvent` kept as the only live transport exception outside canonical replay.

**Tech Stack:** Python 3.10+, existing `Model` and `BaseTool` abstractions, `RunLogStorage`, `RunLoopOrchestrator`, `HookRegistry`, typed `RunLog` models, scheduler/console replay readers, pytest, ruff

## Status Of This Document

This document is a follow-up to [2026-04-21-agent-runtime-refactor-design.md](./2026-04-21-agent-runtime-refactor-design.md).

It does not reopen the high-level direction of the refactor. It tightens the contract where shipped code still allows architectural drift:

1. `RunLog` is not yet the complete runtime truth in all failure paths
2. hook phases and capabilities are not yet fully constrained by a stable public contract
3. `compaction` failure is not yet a first-class replayable runtime fact
4. live `trace` and replayable `stream` still have direct runtime side channels
5. `RunStateWriter` is not yet the only write path for runtime truth
6. the original phase list in the earlier spec is now outdated relative to the better shipped phase structure

This follow-up spec supersedes the earlier document where they differ.

## Scope

This design covers the remaining strict-convergence work inside `agiwo.agent`.

It includes:

1. making every started run terminate with `RunFinished` or `RunFailed`
2. tightening the hook contract into a stable public runtime contract
3. adding first-class `compaction` failure entries to `RunLog`
4. removing replayable trace/stream side channels that bypass committed `RunLog`
5. making `RunStateWriter` the only runtime-truth write path
6. updating the public runtime phase contract to match the better current design

It does not include:

1. replaying token-level provider deltas as canonical persisted history
2. redesigning scheduler or console APIs beyond consuming the stricter replay contract
3. replacing `RunLoopOrchestrator` with a new top-level runtime owner name
4. introducing a generic event bus

## Fixed Decisions

These decisions are fixed for this convergence pass:

1. `RunLoopOrchestrator` remains the single phase decider
2. `RunStateWriter` becomes the only path allowed to mutate runtime truth
3. `TraceBuilder` and `StreamBuilder` stay outside the write path and only consume committed entries
4. `StepDeltaEvent` remains a live transport exception and is not part of canonical replay
5. the shipped broader phase set is kept and formalized rather than collapsed back to the earlier minimal list
6. `before_tool_batch` and `after_tool_batch` are retired as public names; the canonical public phase names are `before_tool_call` and `after_tool_call`
7. `compaction` success and failure are both first-class replayable runtime facts
8. any fatal run error after `RunStarted` must end in `RunFailed`

## Design Goals

This convergence pass optimizes for these goals in order:

1. complete and replayable runtime truth
2. one write path for state and canonical runtime facts
3. deterministic public hook semantics
4. one projection path for replayable trace and stream facts
5. strong behavior-level acceptance tests that prevent architectural regression

## Target Runtime Responsibilities

### `RunLoopOrchestrator`

`RunLoopOrchestrator` remains the only execution owner for a single run. It owns:

1. phase progression
2. calling hooks and runtime policies
3. deciding whether execution continues, pauses, or finishes
4. deciding when a failure is fatal to the run
5. delegating all runtime-truth writes to `RunStateWriter`

`RunLoopOrchestrator` must not:

1. mutate committed runtime state directly
2. append `RunLog` entries directly
3. publish replayable stream events directly
4. update trace state directly

### `RunStateWriter`

`RunStateWriter` is the only mutable write path for runtime truth.

Runtime truth means:

1. committed in-memory runtime state
2. canonical typed `RunLog` entries

`RunStateWriter` owns:

1. run lifecycle writes
2. message rebuild writes
3. committed step writes
4. termination writes
5. `compaction` success and failure writes
6. `step-back` writes
7. hook failure writes
8. keeping in-memory committed state and appended `RunLog` entries consistent
9. returning the committed entries for downstream projection

`RunStateWriter` does not own trace or stream building.

### `HookRegistry`

`HookRegistry` remains the runtime hook dispatcher. Its contract is tightened so that:

1. every public phase is explicit
2. every phase has an explicit payload contract
3. every transform-capable phase has an explicit field allowlist
4. hook execution order is deterministic across handler groups
5. invalid `critical` usage is rejected at registration time
6. hook failures are recorded through the canonical write path

### `RunPolicies`

`termination`, `compaction`, and `step-back` stay as named runtime policies.

Policies may:

1. inspect runtime state
2. produce structured decisions or outcomes

Policies may not:

1. mutate committed runtime state directly
2. append `RunLog` entries directly
3. publish trace or stream outputs directly

### `TraceBuilder`

`TraceBuilder` consumes committed `RunLog` entries and builds:

1. live trace updates from newly committed entries
2. replayed traces from stored entries

Live trace updates and replayed traces must use the same entry-to-span projection rules.

### `StreamBuilder`

`StreamBuilder` consumes committed `RunLog` entries and builds replayable public stream items.

The only live stream exception outside canonical replay is:

1. `StepDeltaEvent`

All other public replayable stream items must be derived from committed `RunLog` entries only.

## Public Runtime Phases

The canonical public runtime phases are:

1. `prepare`
2. `assemble_context`
3. `before_llm`
4. `after_llm`
5. `before_tool_call`
6. `after_tool_call`
7. `before_compaction`
8. `after_compaction`
9. `compaction_failed`
10. `before_review`
11. `after_step_back`
12. `before_termination`
13. `after_termination`
14. `after_step_commit`
15. `run_finalized`
16. `memory_persist`

The earlier public names `before_tool_batch` and `after_tool_batch` are retired. This is not just a naming cleanup. The public contract is formally per-tool-call because the current runtime behavior is per-tool-call and that is the better boundary.

## Hook Contract

### Hook Capabilities

Hooks must declare one of:

1. `observe_only`
2. `transform`
3. `decision_support`

Capability meaning:

1. `observe_only` may inspect the phase payload and return nothing meaningful
2. `transform` may only modify fields explicitly allowed by that phase
3. `decision_support` may only append structured advice fields defined by that phase

Hooks cannot:

1. mutate internal runtime state directly
2. commit steps directly
3. append `RunLog` entries directly
4. publish stream events directly
5. write trace spans directly
6. set termination state directly

### Hook Groups And Ordering

Hooks execute in this order:

1. `system`
2. `runtime_adapter`
3. `user`

Within each group, hooks execute by:

1. explicit numeric `order`
2. stable registration order as the final tie-breaker

This ordering is part of the public contract and must be behavior-tested.

### Critical Hooks

`critical=True` is allowed only in:

1. `prepare`
2. `assemble_context`
3. `before_llm`
4. `before_tool_call`

All other phases must reject critical hook registration.

### Hook Failure Policy

The default policy remains isolate-by-default:

1. non-critical hook failures are caught
2. a `HookFailed` runtime fact is recorded
3. execution continues

Critical hook failures:

1. must still record `HookFailed`
2. are then surfaced back to `RunLoopOrchestrator`
3. must end in `RunFailed` if the run has already started

### Phase Payload Rules

The runtime must stop accepting arbitrary payload mutations.

The canonical rules are:

1. `prepare`
   - transform allowlist: `prelude_text`
   - decision-support fields: none
2. `assemble_context`
   - transform allowlist: `memories`, `context_additions`
   - decision-support fields: none
3. `before_llm`
   - transform allowlist: `messages`, `model_settings_override`
   - decision-support fields: `llm_advice`
4. `after_llm`
   - observe-only only
5. `before_tool_call`
   - transform allowlist: `parameters`
   - decision-support fields: `tool_advice`
6. `after_tool_call`
   - observe-only only
7. `before_compaction`
   - decision-support fields: `compaction_advice`
8. `after_compaction`
   - observe-only only
9. `compaction_failed`
   - observe-only only
10. `before_review`
   - decision-support fields: `step-back_advice`
11. `after_step_back`
   - observe-only only
12. `before_termination`
   - decision-support fields: `termination_advice`
13. `after_termination`
   - observe-only only
14. `after_step_commit`
   - observe-only only
15. `run_finalized`
   - observe-only only
16. `memory_persist`
   - observe-only only

If a hook returns fields outside the phase allowlist, the runtime must reject that mutation rather than silently merge it.

## Runtime Truth And `RunLog`

`RunLog` remains the only persisted runtime source of truth and is tightened further in this pass.

### Completeness Rule

After `RunStarted` is committed, the run must always end in exactly one terminal lifecycle entry:

1. `RunFinished`
2. `RunFailed`

The runtime must not leave a started run without a terminal lifecycle fact.

### Entry Families

This convergence pass keeps the existing entry families and adds missing strictness:

1. run lifecycle
   - `RunStarted`
   - `RunFinished`
   - `RunFailed`
2. context and request
   - `ContextAssembled`
   - `MessagesRebuilt`
   - `LLMCallStarted`
   - `LLMCallCompleted`
3. committed steps
   - `UserStepCommitted`
   - `AssistantStepCommitted`
   - `ToolStepCommitted`
4. runtime decisions
   - `TerminationDecided`
   - `CompactionApplied`
   - `CompactionFailed`
   - `StepBackApplied`
   - `RunRolledBack`
5. runtime health
   - `HookFailed`
6. content rewrite
   - `StepCondensedContentUpdated`

### `compaction` Failure

`compaction` failure becomes a first-class replayable runtime fact through a new typed entry:

1. `CompactionFailed`

It must record:

1. the triggering run and session identity
2. the failure sequence
3. the error message
4. the retry attempt ordinal
5. the configured retry limit
6. whether the failure caused terminal run behavior

`CompactionApplied` remains success-only. Success and failure are separate facts because they represent different runtime semantics and query use cases.

## Canonical Runtime Data Flow

### Run Start

1. orchestrator enters `prepare`
2. orchestrator delegates run start commit to `RunStateWriter`
3. if any fatal failure occurs after run start, orchestrator must delegate terminal failure commit to `RunStateWriter`

### Context Assembly

1. orchestrator executes `assemble_context`
2. any committed message history rewrite must go through `RunStateWriter`
3. direct `replace_messages(...)` calls outside the write path are not allowed for committed runtime truth

### LLM Turn

1. orchestrator runs `before_llm`
2. writer records `LLMCallStarted`
3. provider deltas may publish live `StepDeltaEvent`
4. final assistant output is committed through `RunStateWriter`
5. replayable stream and trace projections consume the returned committed entries

### Tool Call

1. each tool call runs through `before_tool_call`
2. tool result observation runs through `after_tool_call`
3. committed tool steps are written only through `RunStateWriter`
4. tool-driven termination hints become canonical only after orchestrator delegates `TerminationDecided` to the writer

### `compaction`

1. orchestrator decides whether `compaction` is attempted
2. policy returns `not_needed`, `applied`, or `failed`
3. writer commits either:
   - `MessagesRebuilt` + `CompactionApplied`
   - `CompactionFailed`
4. any derived in-memory `compaction` counters are updated only through the writer

### `step-back`

1. policy/executor returns a structured step-back outcome
2. any committed message rewrite and replayable runtime fact is written only through `RunStateWriter`

### Termination

1. only orchestrator decides termination
2. `TerminationDecided` is written only through `RunStateWriter`
3. normal completion then writes `RunFinished`
4. fatal failure writes `RunFailed`

## Trace And Stream Rules

### Replayable Stream Contract

All replayable public stream items must come from committed `RunLog` entries only.

That includes:

1. run lifecycle events
2. committed-step completion events
3. messages rebuilt events
4. runtime decision events

The only live transport exception is:

1. `StepDeltaEvent`

`StepDeltaEvent` is not part of canonical replay and does not weaken the rule above for all other public stream facts.

### Trace Contract

Live trace updates and replayed traces must both use the same entry-to-span projection logic.

Direct runtime callbacks that construct trace state independently of committed `RunLog` entries are not allowed in the final design.

## Error Handling

The runtime must classify failures into three categories:

1. non-critical hook failure
   - record `HookFailed`
   - continue execution
2. recoverable runtime policy failure
   - record the matching typed runtime fact, such as `CompactionFailed`
   - continue or terminate according to orchestrator policy
3. fatal run failure
   - record `RunFailed`
   - stop execution

This guarantees:

1. every runtime failure of architectural interest becomes a typed runtime fact
2. every fatal run failure after `RunStarted` becomes `RunFailed`

## Acceptance Criteria

This convergence work is complete only when all of the following are true:

1. every run that commits `RunStarted` commits exactly one of `RunFinished` or `RunFailed`
2. `prepare` and `assemble_context` fatal failures still produce `RunFailed`
3. illegal hook registrations are rejected by contract
4. illegal hook transform fields are rejected by contract
5. hook execution order is deterministic and behavior-tested
6. every `compaction` failure produces `CompactionFailed`
7. latest `compaction`, `step-back`, `termination`, and rollback state remain queryable from replay helpers
8. replayable live stream output matches replayed stream output from `RunLog`
9. live trace structure matches replayed trace structure from `RunLog`
10. production code no longer mutates runtime truth outside `RunStateWriter`

## Testing Strategy

The runtime must be covered at four layers:

1. unit tests
   - `HookRegistry`
   - `RunStateWriter`
   - `TraceBuilder`
   - `StreamBuilder`
2. runtime integration tests
   - root runs
   - child runs
   - early failure paths
   - tool runs
   - `compaction`
   - `step-back`
3. live-versus-replay parity tests
   - stream parity
   - trace parity
4. guardrail tests
   - contract tests that fail if production code bypasses `RunStateWriter`

## Migration Notes

This pass is intentionally strict. It does not permit a permanent internal dual architecture.

Short-lived mechanical refactors during implementation are acceptable only if:

1. they are removed before completion of the implementation plan
2. the final shipped state satisfies the strict acceptance criteria above

The intended end state is:

1. orchestrator decides
2. writer commits runtime truth
3. builders project replayable views

That is the canonical runtime contract for the next implementation phase.
