# Agent Runtime Refactor Design

**Goal:** Rebuild the core `agiwo.agent` runtime so that execution flow is easier to reason about, hooks are deterministic and robust, runtime facts have a single source of truth, and later scheduler/console migrations can build on a cleaner runtime protocol without preserving legacy internals.

**Architecture:** Replace the current `run_loop + Run + StepRecord + AgentHooks` execution model with a `RunEngine` centered runtime. In the shipped code, `RunEngine` maps to `RunLoopOrchestrator` as the canonical runtime-facing name. The new runtime writes strongly typed `RunLog` entries as the only persisted source of truth. `termination`, `compaction`, and `step-back` are first-class runtime decisions. `trace` and `stream` are replayable views built from `RunLog`, not peer decision layers.

**Tech Stack:** Python 3.10+, existing `Model` and `BaseTool` abstractions, existing scheduler/console integration surfaces, new `RunLogStorage` facade replacing `RunStepStorage`

## Scope

This design covers the runtime model for the whole execution chain, but the first implementation project is limited to replacing the `agiwo.agent` core runtime.

It includes:

1. replacing the current `run_loop.py`-centered execution model with `RunEngine`
2. deleting `AgentHooks` and replacing it with a new phase-based hook system
3. deleting `Run` and `StepRecord` as persisted source models
4. introducing strongly typed `RunLog` entries as the only runtime source of truth
5. replacing `RunStepStorage` with `RunLogStorage`
6. making `termination`, `compaction`, and `step-back` first-class runtime records
7. rebuilding `trace` and `stream` publication from `RunLog`
8. defining migration stages for scheduler and console

It does not include:

1. preserving backward compatibility for `AgentHooks`
2. preserving `Run` or `StepRecord` as internal canonical models
3. keeping old storage/query semantics as the long-term design target
4. implementing the scheduler and console migrations in the first execution phase
5. replaying token-by-token provider stream output as canonical persisted history

## Current State

The current agent runtime already has some useful separation:

1. `Agent` is mostly a facade
2. `RunContext` separates identity from mutable ledger state
3. `SessionRuntime` owns session-scoped stream publication and storage handles
4. `compaction` and `step-back` are already split out of the main loop into dedicated modules

The remaining complexity is mostly caused by execution ownership and data ownership still being spread across multiple modules:

1. `run_loop.py` still coordinates most execution phases directly
2. hooks are called from multiple places such as `run_loop.py`, `tool_executor.py`, and `step_committer.py`
3. `Run`, `StepRecord`, ledger state, stream events, and trace callbacks together form more than one "truth" about what happened
4. `compaction`, `step-back`, and `termination` affect execution semantics, but `trace` and `stream` learn about them only indirectly
5. the current callback-slot `AgentHooks` model does not express phase ordering, mutation boundaries, or default failure isolation well enough

This makes the runtime harder to change safely, and it increases the coupling cost for scheduler and console.

## Product Decisions

These decisions are fixed for this refactor:

1. keep the names `termination`, `compaction`, and `step-back`
2. delete `AgentHooks` rather than preserve compatibility
3. delete `Run` and `StepRecord` as canonical persisted models
4. keep `Agent.start()`, `Agent.run()`, `Agent.run_stream()`, `Agent.run_child()`, and `Agent.create_child_agent()` as the public execution surfaces
5. make `RunLog` the only persisted runtime source of truth
6. treat "committed step" as one family of `RunLog` entries, not as a separate top-level model
7. make `trace` and `stream` consumers of `RunLog`, not execution peers of `termination`, `compaction`, or `step-back`
8. allow no legacy compatibility layer to become a permanent second architecture

## Design Goals

The refactor should optimize for these goals in order:

1. deterministic execution phases and clear state mutation boundaries
2. hook isolation and fault containment by default
3. extensibility through multiple handlers after phase semantics are stable
4. a single replayable runtime record that can support scheduler, console, and observability
5. reduced coupling between execution logic and storage, trace, and stream publication

## Target Runtime Model

The new runtime is organized around these responsibilities:

### `Agent`

`Agent` remains the public facade. It owns:

1. public construction and execution entrypoints
2. root and child runtime bootstrap
3. tool assembly and workspace assembly
4. passing a stable configuration into the runtime

`Agent` no longer owns the detailed execution lifecycle.

### `RunEngine`

`RunEngine` is the only execution owner for a single run. It owns:

Implementation mapping:
- `HookDispatcher` in this document maps to the shipped `agiwo.agent.hooks.HookRegistry` phase registry plus its dispatch helpers.
- `RunLogWriter` in this document maps to `SessionRuntime.append_run_log_entries(...)` and the typed entry builders in `agiwo.agent.runtime.state_writer`.
- `RunEngine` remains the single authoritative phase-decider in the implementation, with `RunLoopOrchestrator` kept as the canonical runtime-facing name.

1. phase progression
2. calling `termination`, `compaction`, and `step-back` policies
3. deciding when the run continues, pauses, or finishes
4. invoking `HookDispatcher`
5. sending all state changes through `RunStateWriter`
6. writing canonical runtime facts through `RunLogWriter`

Only `RunEngine` may decide what the next runtime phase is.

### `HookDispatcher`

`HookDispatcher` owns:

1. hook registration and ordering
2. phase-specific hook invocation
3. capability enforcement for each hook
4. default failure isolation
5. recording hook failures into `RunLog`

Hooks do not call runtime internals directly.

### `RunStateWriter`

`RunStateWriter` is the only mutable state write path. It owns:

1. rebuilding messages
2. applying committed runtime changes
3. writing committed-step entries
4. updating derived in-memory run state
5. ensuring state transitions and `RunLog` entries stay consistent

No other component may directly mutate message history, committed response state, or termination state.

### `RunPolicies`

`RunPolicies` is the runtime rule layer for:

1. `termination`
2. `compaction`
3. `step-back`

Each policy returns a structured decision. Policies do not publish streams, write traces, or mutate state directly.

### `RunLogWriter`

`RunLogWriter` owns appending strongly typed `RunLog` entries to storage.

### `TraceBuilder`, `StreamBuilder`, `MetricsBuilder`

These are view builders. They consume `RunLog` and build:

1. trace trees
2. outward streaming events
3. aggregated run metrics and run/session views

They do not decide runtime behavior.

## Runtime Phases

The first version of the runtime should expose only a fixed set of phases:

1. `prepare`
2. `assemble_context`
3. `before_llm`
4. `after_llm`
5. `before_tool_batch`
6. `after_tool_batch`
7. `before_compaction`
8. `after_compaction`
9. `before_review`
10. `after_step_back`
11. `before_termination`
12. `after_termination`
13. `after_step_commit`
14. `finalize`

The design intentionally avoids a general event-bus model. The runtime should first make the main execution path explicit and deterministic.

## Hook System

The new hook system replaces `AgentHooks` entirely.

### Hook Capabilities

Hooks must declare one of three capabilities:

1. `observe_only`
2. `transform`
3. `decision_support`

Capability meanings:

1. `observe_only` can inspect the phase payload but cannot modify execution input
2. `transform` may return a modified payload only for fields explicitly allowed by that phase
3. `decision_support` may attach structured advice for the engine, but it cannot directly terminate or mutate run state

Hooks cannot:

1. mutate internal run state directly
2. commit steps directly
3. publish stream events directly
4. write trace spans directly
5. set termination state directly

### Hook Ordering

Within a phase, hooks execute in this order:

1. system hooks
2. runtime adapter hooks
3. user hooks

Within each group, hooks execute by explicit numeric `order`, defaulting to `100`.

This gives deterministic execution before it adds richer composition.

### Hook Failure Policy

The default hook behavior is `isolate-by-default`.

That means:

1. hook failures are caught
2. failures are recorded as `RunLog` entries
3. execution continues unless the hook was explicitly declared `critical`

Critical hooks are allowed only in these phases:

1. `prepare`
2. `assemble_context`
3. `before_llm`
4. `before_tool_batch`

All `after_*` phases default to non-critical behavior.

## RunLog

`RunLog` becomes the only persisted runtime source of truth.

### Core Requirements

`RunLog` must be:

1. strongly typed
2. append-only at the source-of-truth level
3. replayable in execution order
4. queryable without forcing all callers to manually replay full history
5. rich enough to explain both committed steps and runtime decisions

`RunLog` must not become a generic unstructured event blob store.

### Ordering

Every `RunLog` entry should carry a monotonic sequence in session order. This preserves the current ability to reason about a shared session timeline, including nested runs, `compaction` ranges, and `step-back` impact ranges.

Committed-step views should retain this ordering model so existing user-facing sequence semantics remain understandable.

### Entry Families

The first implementation must include at least these entry families:

1. run lifecycle entries
   - `RunStarted`
   - `RunFinished`
   - `RunFailed`
2. context and request entries
   - `ContextAssembled`
   - `MessagesRebuilt`
   - `LLMCallStarted`
   - `LLMCallCompleted`
3. committed-step entries
   - `UserStepCommitted`
   - `AssistantStepCommitted`
   - `ToolStepCommitted`
4. policy decision entries
   - `CompactionApplied`
   - `StepBackApplied`
   - `TerminationDecided`
5. runtime health entries
   - `HookFailed`

Additional entry types are acceptable only if they clarify execution semantics rather than recreate an unfocused event bus.

### Relation To Old `StepRecord`

`StepRecord` is deleted.

Its useful semantics move into committed-step `RunLog` entries:

1. user-visible message content
2. assistant content and tool calls
3. tool results and `content_for_user`
4. timing and token metrics tied to committed output
5. sequence-based ordering

The key difference is that a committed step is no longer a separate top-level source model. It is one family inside the unified `RunLog`.

### Relation To Old `Run`

`Run` is deleted.

Its old responsibilities split into:

1. `RunIdentity`: runtime identity only, not a persisted source model
2. lifecycle `RunLog` entries such as `RunStarted`, `TerminationDecided`, `RunFinished`, and `RunFailed`
3. projected read models such as `RunOutput`, `RunView`, and `RunMetrics`

This removes the second source-of-truth problem where run status and response were stored both as runtime facts and as a separate aggregate object.

## `termination`, `compaction`, and `step-back`

These three remain named exactly as they are today, but their runtime role changes.

### `termination`

`termination` becomes a first-class engine decision. It should no longer be treated as a field set "at the end" of the loop.

It must record:

1. the termination reason
2. the phase that triggered it
3. the evidence or policy source that caused it

### `compaction`

`compaction` remains named `compaction`.

It must record:

1. the affected sequence range
2. any persisted transcript or summary references
3. before/after message reconstruction facts
4. whether the compaction attempt succeeded or failed

`compaction` is an engine decision plus state rewrite, not a hidden message mutation.

### `step-back`

`step-back` remains named `step-back`.

It must record:

1. the affected committed-step range or tool-result range
2. the condensed replacement or reference
3. the reason it was triggered
4. how message history was rewritten afterward

`step-back` is not just tool-result post-processing. It is a first-class context rewrite decision.

## Trace And Stream Publication

`trace` and `stream` are intentionally not modeled at the same layer as `termination`, `compaction`, or `step-back`.

The rule is:

1. runtime decisions happen first
2. `RunLog` records those facts
3. trace and stream builders consume the resulting facts

### Replay Model

The design guarantees replay at the phase and committed-step level.

That means:

1. committed step history is replayable
2. `termination`, `compaction`, and `step-back` decisions are replayable
3. trace trees can be rebuilt from `RunLog`
4. outward stream timelines can be rebuilt from `RunLog`

The design does not require token-perfect replay of provider stream chunks.

Provider token deltas may still be published live for UX, but they are treated as transport-level deltas rather than canonical persisted runtime facts.

## Storage And Query Boundary

`RunStepStorage` is replaced by `RunLogStorage`.

### Write Interface

`RunLogStorage` must support:

1. appending one or more `RunLog` entries
2. allocating monotonic session-order sequence values
3. storing any compacted transcript or large payload references needed by `compaction` and `step-back`

### Read Interface

The stable query surface should return read models, not raw persistence assumptions.

Required query capabilities:

1. replay a run or session timeline in order
2. list committed-step views for a session, run, or agent
3. return latest run view for a session
4. return latest `compaction`, `step-back`, and `termination` state for a run or session
5. return summary counts needed by console session list views

### Query Model

The storage layer may maintain projections or indexes internally. Callers should not be forced to replay a full `RunLog` on every query.

The intended public read models are:

1. `RunView`
2. `StepView`
3. replayed `AgentStreamItem`
4. explicit runtime-decision views for latest `termination`, `compaction`, `step-back`, and rollback state
5. `RunMetrics`
6. `RunOutput`

Console and scheduler should consume these views through a stable facade rather than learn `RunLog` layout details themselves.

## Public API Impact

The public execution surfaces should remain recognizable:

1. `Agent.start(...)`
2. `Agent.run(...)`
3. `Agent.run_stream(...)`
4. `Agent.run_child(...)`
5. `Agent.create_child_agent(...)`

The public extension surface changes:

1. `AgentHooks` is removed
2. `Agent(..., hooks=...)` stays as the constructor surface, but `hooks` now accepts phase-based hook handlers or a hook registry rather than an `AgentHooks` dataclass

The public result surface remains view-oriented:

1. `RunOutput` stays as the return model for synchronous waiting
2. `AgentStreamItem` stays as the outward stream item protocol
3. console/API-facing run and step responses are rebuilt from `RunLog` views rather than SDK source models

## Migration Strategy

The architecture is defined once, but delivery is staged.

### Phase 0: Design Lock

Before implementation starts:

1. finalize `RunLog` entry families
2. finalize hook phase payloads and allowed transforms
3. finalize `RunLogStorage` read/write contracts
4. define migration acceptance tests for agent, scheduler, and console

### Phase 1: Replace `agiwo.agent` Runtime

This is the first implementation project.

It includes:

1. introducing `RunEngine`
2. introducing `HookDispatcher`
3. introducing `RunStateWriter`
4. introducing `RunLogWriter`
5. introducing `RunLogStorage`
6. deleting `Run`
7. deleting `StepRecord`
8. deleting `AgentHooks`
9. moving trace and stream publication to `RunLog`-driven builders

Implementation mapping:

- `RunEngine` maps to the shipped `RunLoopOrchestrator`
- `HookDispatcher` maps to `agiwo.agent.hooks.HookRegistry`
- `RunLogWriter` maps to `SessionRuntime.append_run_log_entries(...)` plus typed entry builders
- the shipped stable read surface is `RunView`, `StepView`, replayed `AgentStreamItem`, and runtime-decision views, not a separate `TimelineView`

Completion criteria:

1. root and child agent execution run fully on the new runtime
2. `termination`, `compaction`, and `step-back` are stored as first-class `RunLog` entries
3. `trace` and `stream` can be rebuilt from `RunLog`
4. existing agent-level behavior is re-covered by tests using the new model

### Phase 2: Scheduler Migration

After the agent runtime is stable:

1. migrate scheduler runtime integration to consume `RunLog` views and replay
2. remove direct dependence on old run/step storage semantics
3. align root/child/steer/wait/cancel semantics with the new runtime facts

### Phase 3: Console Migration

After scheduler migration:

1. migrate console queries and SSE serialization to `RunLog`-backed views
2. remove direct assumptions about persisted `Run` or `StepRecord` models
3. align session summary and timeline views with the new query facade

## Testing Strategy

The refactor requires stronger behavior-level tests than the current structure-only checks.

At minimum, the new runtime test plan should cover:

1. phase ordering and hook ordering
2. hook capability enforcement
3. hook failure isolation and critical failure behavior
4. committed-step generation from `RunLog`
5. `termination`, `compaction`, and `step-back` replay behavior
6. trace rebuild from `RunLog`
7. stream rebuild from `RunLog`
8. root and child run execution correctness
9. console-facing query summaries built from `RunLog`
10. scheduler-facing timeline and wait/cancel behavior built from `RunLog`

## Risks And Guardrails

### Risk: `RunLog` Becomes Too Generic

Guardrail:

1. keep entry families strongly typed
2. reject generic "freeform event payload" growth
3. require each new entry type to explain why it cannot be represented as an existing typed fact

### Risk: Query Cost Regresses

Guardrail:

1. build `RunLogStorage` with explicit query/index expectations
2. keep session summary, latest run view, and committed-step views as first-class read paths

### Risk: Migration Scope Expands Too Early

Guardrail:

1. implement only `agiwo.agent` in the first execution project
2. define scheduler and console migration contracts now, but migrate them in later phases

### Risk: Hook System Reintroduces Hidden Mutation

Guardrail:

1. enforce capability limits in `HookDispatcher`
2. keep `RunStateWriter` as the only state mutation path

## Acceptance Criteria

1. The new design has one persisted source of truth: `RunLog`.
2. `Run`, `StepRecord`, and `AgentHooks` are removed from the runtime design.
3. `termination`, `compaction`, and `step-back` remain named as-is and become first-class runtime records.
4. `trace` and `stream` are modeled as replayable views that consume `RunLog`.
5. The `Agent` public execution surfaces remain stable enough for staged migration.
6. The design supports a first implementation phase focused only on replacing `agiwo.agent` internals without forcing scheduler and console to be rewritten in the same change.
