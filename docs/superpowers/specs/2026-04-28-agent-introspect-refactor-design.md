# Agent Introspect Refactor Design

## Context

The current goal-directed review implementation grew through several rounds of
runtime, replay, trace, and Console changes. It now works through a mix of
`ReviewBatch`, direct storage writes, ad hoc ledger message edits, RunLog facts,
trace spans, and Console text parsing.

This design replaces the current `agiwo.agent.review` subsystem with a cleaner
`agiwo.agent.introspect` subsystem. The refactor keeps the feature semantics but
does not preserve old data compatibility. Existing development data may be
deleted.

## Goals

- Make first-class RunLog facts the only canonical source for goal,
  introspection, and context repair state.
- Separate the domain concepts:
  - goal contract: what the Agent has committed to;
  - trajectory introspection: how runtime checks whether execution stays aligned;
  - context repair: how off-track tool output is condensed after misalignment.
- Remove the overloaded `ReviewBatch` lifecycle object.
- Remove Console parsing of `<system-review>` as a data protocol.
- Keep hooks as extension points, not as the implementation mechanism.
- Keep `run_tool_batch.py` as the tool-batch execution owner while moving domain
  rules into focused `introspect` modules.

## Non-Goals

- No storage migration.
- No compatibility wrapper for `agiwo.agent.review`.
- No legacy Console review-cycle fallback for old text-only traces.
- No broad redesign of scheduler runtime tools beyond names/imports needed for
  the introspect model.

## Package Layout

Create:

```text
agiwo/agent/introspect/
  __init__.py
  models.py
  goal.py
  trajectory.py
  repair.py
  apply.py
  replay.py
```

Delete after migration:

```text
agiwo/agent/review/
agiwo/agent/models/review.py
```

`models.py` contains pure data structures: `Milestone`, `GoalState`,
`IntrospectionState`, `IntrospectionCheckpoint`,
`PendingIntrospectionNotice`, `GoalUpdate`, `IntrospectionNotice`,
`IntrospectionOutcome`, `ContextRepairPlan`, and `ContentUpdate`.

`goal.py` owns milestone parsing, validation, and state transitions. It handles
`declare_milestones` output, enforces unique milestone ids, enforces at most one
active milestone, and reports whether a milestone transition should trigger
introspection.

`trajectory.py` owns introspection trigger and outcome rules. It tracks tool
counts, consecutive errors, milestone-switch triggers, `<system-review>` notice
text, and `review_trajectory` output parsing.

`repair.py` owns pure context repair planning. It identifies prompt-visible
review notices to clean, review metadata steps to hide, and off-track tool
results to condense.

`apply.py` is the commit boundary for introspect effects. It mutates live ledger
messages as needed, commits facts through `RunStateWriter`, and projects every
committed entry through `SessionRuntime.project_run_log_entries(...)`.

`replay.py` rebuilds `GoalState` and `IntrospectionState` from committed RunLog
facts.

## Runtime Flow

`run_tool_batch.py` remains the explicit execution sequence owner. It should not
contain goal, introspection, or repair rules.

For each tool result:

1. Run `after_tool_call` hooks as today.
2. Allocate sequence and build an uncommitted `StepView.tool(...)`.
3. Call `goal.handle_goal_tool_result(...)` to prepare any goal update.
4. Call `trajectory.maybe_build_introspection_notice(...)`.
5. If a notice is returned:
   - call `hooks.before_review(...)` for optional advice;
   - append the `<system-review>` notice to the uncommitted tool step content.
6. Commit the final `ToolStepCommitted` step.
7. If a goal update was prepared, call `apply.commit_goal_update(...)` with the
   committed step id.
8. If a notice was prepared, call `apply.commit_introspection_trigger(...)`
   with the committed step id.
9. If the tool result is `review_trajectory`, call
   `trajectory.parse_introspection_outcome(...)` and remember the outcome for
   batch finalization with the committed review tool step id.

After the batch:

1. If there is no introspection outcome, stop.
2. Build a repair plan with `repair.build_context_repair_plan(...)`.
3. Commit hidden metadata steps, cleaned notice updates, condensed content
   updates, introspection outcome facts, checkpoint facts, and context repair
   facts through `apply.py`.
4. Call `hooks.after_step_back(...)` only when a context repair was actually
   applied.

There is no `IntrospectionCoordinator` or replacement for `ReviewBatch`.
The flow is explicit in `run_tool_batch.py`, and each called function has one
clear purpose.

## State Model

`GoalState`:

```text
milestones: list[Milestone]
active_milestone_id: str | None
```

`IntrospectionState`:

```text
review_count_since_boundary: int
consecutive_errors: int
pending_trigger: PendingIntrospectionNotice | None
last_boundary_seq: int
latest_aligned_checkpoint: IntrospectionCheckpoint | None
pending_milestone_switch: bool
```

`last_boundary_seq` is the canonical boundary for both review counting and
future context repair. Every accepted `review_trajectory` outcome advances this
boundary, including `aligned=False` and malformed/unknown outcomes. This avoids
re-condensing old off-track tool results during consecutive misaligned reviews.

`latest_aligned_checkpoint` only records the most recent aligned checkpoint. It
is not used as the cleanup boundary after a misaligned review.

## RunLog Facts

Rename review facts to introspect facts:

```text
GoalMilestonesUpdated
IntrospectionTriggered
IntrospectionCheckpointRecorded
IntrospectionOutcomeRecorded
ContextRepairApplied
```

`GoalMilestonesUpdated`:

```text
milestones
active_milestone_id
source_tool_call_id
source_step_id
reason
```

`IntrospectionTriggered`:

```text
trigger_reason
active_milestone_id
review_count_since_boundary
trigger_tool_call_id
trigger_tool_step_id
notice_step_id
```

`IntrospectionCheckpointRecorded`:

```text
checkpoint_seq
milestone_id
review_tool_call_id
review_step_id
```

`IntrospectionOutcomeRecorded`:

```text
aligned
mode
experience
active_milestone_id
review_tool_call_id
review_step_id
hidden_step_ids
notice_cleaned_step_ids
condensed_step_ids
boundary_seq
repair_start_seq
repair_end_seq
```

`ContextRepairApplied`:

```text
mode = "step_back"
affected_count
start_seq
end_seq
experience
```

`IntrospectionOutcomeRecorded(mode="step_back")` and
`ContextRepairApplied(mode="step_back")` intentionally overlap. The first is
the introspection conclusion; the second is the runtime context-repair action.

`StepCondensedContentUpdated` and `ContextStepsHidden` remain generic runtime
facts and are reused by context repair.

## Hooks

Hooks remain extension points:

- `before_review` can provide optional review advice before a notice is
  committed.
- `after_step_back` observes an applied context repair.

Hooks must not implement canonical introspection behavior, write storage
directly, or become required during replay.

## Prompt Notice Semantics

`<system-review>` remains a prompt-control mechanism only.

- Runtime may append it to a tool result to force `review_trajectory`.
- Context repair may clean it from tool results after the review outcome.
- Console, replay, and trace views must not parse it as authoritative state.

## Console And Trace

Trace projection consumes the new RunLog facts and emits runtime spans for:

- goal milestone updates;
- introspection triggers;
- introspection checkpoints;
- introspection outcomes;
- context repair.

Console observability consumes these spans/facts only. Remove
`parse_system_review_notice` and related text parsing for review cycles.

## Validation Rules

`declare_milestones` fails fast when:

- the payload is not a non-empty list;
- any milestone id or description is empty;
- the same milestone id appears more than once in one payload;
- more than one milestone is marked active.

The goal layer does not silently merge duplicate ids in a single declaration.
Existing milestones can still be updated by a later valid declaration.

## AGENTS.md Update

Update the agent section to say that goal declaration, trajectory
introspection, and context repair live under `agiwo.agent.introspect`.
Remove the old statement that goal-directed review and step-back optimization
are owned by `agiwo.agent.review`.

## Test Plan

Goal tests:

- valid milestone declaration;
- duplicate ids fail;
- multiple active milestones fail;
- milestone updates preserve declaration metadata where appropriate;
- active milestone switch sets the introspection trigger condition.

Trajectory tests:

- non-introspection tool steps increment `review_count_since_boundary`;
- `review_trajectory` does not increment the counter;
- step interval triggers a notice;
- consecutive errors trigger a notice;
- milestone switch triggers a notice;
- malformed outcomes still advance `last_boundary_seq`.

Repair tests:

- aligned outcomes clean prior prompt-visible review notices;
- misaligned outcomes condense tool results after the previous boundary and
  before the review result;
- consecutive misaligned reviews do not repair already-repaired old steps;
- review assistant/tool metadata steps are hidden.

RunLog and replay tests:

- all new facts serialize and deserialize;
- `replay.py` restores `GoalState` and `IntrospectionState`;
- `last_boundary_seq` is restored from outcomes;
- hidden steps and condensed content affect replayed `StepView` context.

Integration and Console tests:

- `execute_tool_batch_cycle` commits and projects all introspect facts;
- trace writer projects new introspect facts;
- Console review cycles are built from new facts only;
- removed text-parser paths have no remaining tests or exports.

## Implementation Order

1. Add `agiwo/agent/introspect/models.py`.
2. Add pure `goal.py`, `trajectory.py`, and `repair.py` with focused tests.
3. Add new RunLog fact dataclasses and storage serialization.
4. Add `RunStateWriter` commit methods for introspect facts and condensed
   content updates.
5. Add `apply.py` commit helpers that always project returned entries.
6. Update `run_tool_batch.py` to call the new focused functions.
7. Update `run_bootstrap.py` to restore goal and introspection state.
8. Update trace writer and Console observability for new facts.
9. Delete `agiwo/agent/review/` and `agiwo/agent/models/review.py`.
10. Update AGENTS.md.
11. Run SDK lint/tests and Console backend tests.
