# Agent Runtime Mainline Cleanup Design

**Date:** 2026-04-23

**Status:** Draft for review

## Problem

The single-run agent runtime has a clear implementation owner in `agiwo/agent/run_loop.py`, but `agiwo/agent/runtime/` still contains compatibility shells and thin wrappers that make the mainline harder to read. `runtime/run_engine.py` re-exports the actual owner, `runtime/step_committer.py` wraps a small execution pipeline, and `runtime/__init__.py` exposes mixed responsibilities through a broad aggregate surface.

This creates a misleading structure: readers cannot immediately tell whether single-run execution is owned by `run_loop.py`, `runtime/run_engine.py`, or the runtime package facade.

## Goals

1. Make the single-run execution mainline obvious to a new reader.
2. Ensure each runtime file has one clear responsibility.
3. Remove aliases, forwarding modules, and compatibility shells that obscure ownership.
4. Preserve external behavior and the public `agiwo.agent` facade.
5. Keep run-log committed entries as the canonical runtime truth.

## Non-Goals

This project does not redesign the whole `agiwo.agent` package, change scheduler or console architecture, change run-log schemas, add migrations, or alter public `Agent.start`, `Agent.run`, or `Agent.run_stream` behavior.

## Recommended Approach

Use `agiwo/agent/run_loop.py` as the only single-run execution owner and make `agiwo/agent/runtime/` a support layer.

This fits the current implementation and the existing repository boundary documented in `AGENTS.md`: top-level agent modules contain stable entries and core orchestrators, while `runtime/` contains run/session context, state helpers, `RunStateWriter`, and session projection support.

## Module Boundaries

### `agiwo/agent/run_loop.py`

`run_loop.py` owns the single-run state machine. `RunLoopOrchestrator` remains the only phase decider for one run. It coordinates bootstrap, compaction, assistant turns, tool execution, termination decisions, finalization, and failure handling.

Readers should be able to start here and understand the run lifecycle without first understanding runtime package internals.

### `agiwo/agent/runtime/context.py`

`context.py` owns run-scoped containers:

- `RunContext`: immutable run identity plus mutable ledger and session access.
- `RunRuntime`: ephemeral runtime dependencies for a single run.

It must not contain phase decisions or execution flow.

### `agiwo/agent/runtime/state_writer.py`

`state_writer.py` remains the single committed-state write coordinator. It mutates the runtime ledger and appends typed `RunLog` entries together.

It should not own projection or hook orchestration. Those are execution pipeline concerns owned by `RunLoopOrchestrator` or session projection code.

### `agiwo/agent/runtime/session.py`

`session.py` remains the session-level owner for sequence allocation, run-log append, stream projection, trace projection, and replayable session views.

It does not decide run phases.

### `agiwo/agent/runtime/state_ops.py`

`state_ops.py` may remain as low-level ledger mutation helpers, but only as implementation support for `RunStateWriter`. Other modules should not use it to bypass the writer for committed runtime truth.

## File-Level Changes

### Delete `runtime/run_engine.py`

This file only re-exports `RunEngine` and `RunLoopOrchestrator` from `run_loop.py`. It creates a second apparent execution entry and should be removed.

The `RunEngine = RunLoopOrchestrator` alias should also be removed. Internal code and tests should import `RunLoopOrchestrator` from `agiwo.agent.run_loop`.

### Fold and Delete `runtime/step_committer.py`

`step_committer.py` currently performs a small execution pipeline:

1. construct `RunStateWriter`
2. commit the step
3. project committed entries
4. trigger `on_step`

That pipeline belongs to the single-run execution owner, not to a separate runtime concept. Move this flow into a private `RunLoopOrchestrator._commit_step(...)` method.

Helpers that need to commit steps, such as compaction, termination summary, and tool-batch execution, should receive an explicit commit callback or a narrowly scoped execution service from `RunLoopOrchestrator`. They should not import a runtime-level `commit_step` facade.

### Delete `runtime/hook_dispatcher.py`

`hook_dispatcher.py` is a thin wrapper around `HookRegistry._dispatch(...)`. If no production caller needs it, delete it. If a real dispatch abstraction is needed, it should live in `HookRegistry` as a proper public method rather than in the runtime package as a shell.

### Tighten `runtime/__init__.py`

`runtime/__init__.py` should expose only stable runtime support types:

- `RunContext`
- `RunRuntime`
- `SessionRuntime`
- `RunStateWriter`

It should not expose execution owners, commit-step wrappers, `state_ops` helpers, or `build_*_entry` helpers. Internal modules should import from concrete modules instead of the aggregate runtime package.

## Execution Flow

`RunLoopOrchestrator.execute_run(...)` should present the lifecycle as a short sequence of high-level phases:

1. Start the run and write `RunStarted`.
2. Bootstrap the run context with messages, memory, tool schemas, and the user step.
3. Resume pending tool calls when present, otherwise enter the normal run loop.
4. Loop until terminal by checking limits, running compaction, calling the model, handling assistant output, executing tools, and recording termination decisions.
5. Finalize with termination summary, hooks, memory write, and `RunFinished`.
6. On errors, write the correct termination decision and `RunFailed`.

Private methods that define phase control should remain on `RunLoopOrchestrator`, including `_run_loop`, `_run_loop_iteration`, `_run_compaction_cycle`, `_run_assistant_turn`, `_handle_assistant_turn_result`, `_execute_tool_calls`, `_set_termination_reason`, `_finalize_run`, and `_fail_run`.

Domain helpers such as `run_bootstrap.py`, `run_tool_batch.py`, `llm_caller.py`, `compaction.py`, `termination/*`, and `review/*` can remain separate because they implement phase work, not phase ownership.

## Correctness Requirements

The refactor must preserve these behaviors:

- `Agent.start`, `Agent.run`, and `Agent.run_stream` external behavior.
- `agiwo.agent` public facade exports.
- Run-log entries as the canonical source of runtime truth.
- Trace and replayable stream projection from committed `RunLog` entries.
- `StepDeltaEvent` as the only live-only stream exception.
- Hook order and hook phases, especially `on_step`, `after_run`, and `memory_write`.
- Step commit behavior for normal assistant turns, tool results, compaction, step-back, and termination summary.
- Cancel and error terminal handling, including termination decisions and `RunFailed`.

## Risks

The main risk is losing side effects currently hidden behind `commit_step`: run-log append, projection, and `on_step` hook dispatch must move together. The new owner method should make those side effects explicit and should be covered by tests before deleting the old module.

Another risk is replacing old thin wrappers with new thin wrappers under different names. New helper boundaries are only acceptable when they have an independent responsibility that makes the mainline easier to read.

## Implementation Order

1. Capture a baseline by running agent runtime tests.
2. Add or confirm coverage for step commit side effects: log append, projection, and `on_step`.
3. Move the step commit pipeline into `RunLoopOrchestrator`.
4. Update compaction, termination summary, and tool-batch helpers to commit through the orchestrator-owned path.
5. Delete `runtime/step_committer.py` and update imports.
6. Delete `runtime/run_engine.py`, remove the `RunEngine` alias, and update imports/tests/docs.
7. Delete `runtime/hook_dispatcher.py` or replace it with a real `HookRegistry` method if needed.
8. Tighten `runtime/__init__.py` exports and update concrete imports.
9. Update `AGENTS.md` to document the sharper boundary.
10. Run `uv run python scripts/lint.py ci` and `uv run pytest tests/agent -v`.

## Acceptance Criteria

- A new reader can identify `agiwo/agent/run_loop.py` as the only single-run execution mainline.
- `agiwo/agent/runtime/` contains support objects, not alternate execution entries.
- No internal production code imports `RunEngine`, `runtime.run_engine`, or runtime-level `commit_step`.
- Step commits still append run logs, project trace/stream views, and trigger hooks.
- Existing public `agiwo.agent` imports continue to work.
- Agent tests and lint pass.
