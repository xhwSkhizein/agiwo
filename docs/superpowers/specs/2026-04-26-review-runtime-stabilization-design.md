# Review Runtime Stabilization Design

## Goal

Stabilize goal-directed review by making runtime state and observability depend on first-class RunLog facts instead of prompt-visible review metadata or tool-span inference.

This design covers:

- `review_step_interval` semantics based on non-`review_trajectory` tool results.
- Structured milestone, review-trigger, checkpoint, and review-outcome RunLog facts.
- Replay, Console, and trace views built from those facts.
- One-shot prompt-visible `<system-review>` lifecycle.

This design does not cover:

- Review backoff, per-run budgets, or token/cost-based throttling.
- Soft review or heuristic review that avoids an LLM round trip.
- A major redesign of the `review_trajectory` tool schema.
- Data migration for historical sessions.

## Current Problem

The current implementation has two different sources of truth:

- Runtime review decisions use `RunLedger.review`, a live in-memory `ReviewState`.
- Console and trace views infer milestone and review data from `declare_milestones`, `review_trajectory`, and `<system-review>` text in tool spans.

This creates unstable behavior:

- `review_step_interval` is implemented as `current_seq - last_review_seq`, where `sequence` includes assistant, tool, review, and other committed runtime steps. The configuration reads like "every N tool calls" but behaves like "every N global sequence positions."
- Milestone state is not a first-class persisted fact. It is mostly reconstructed from tool payloads.
- Review checkpoints and outcomes are not first-class persisted facts. Observability has to infer them from tool calls and notice text.
- Prompt-visible `<system-review>` is a control notice, but if it remains visible after review completion it can retrigger review.

The desired shape is: runtime writes structured facts, replay reads structured facts, observability reads structured facts, and prompt-visible notices remain one-shot instructions only.

## Design Summary

Use first-class RunLog entries as the canonical source for milestone and review state.

Add these RunLog facts:

- `ReviewMilestonesUpdated`
- `ReviewTriggerDecided`
- `ReviewCheckpointRecorded`
- `ReviewOutcomeRecorded`

`ReviewState` remains the live runtime state used during a run, but it must be replayable from these facts. Console and trace review views must use these facts as the authoritative source. Tool spans stay available for debug, but they are no longer the status source for milestone boards or review cycles.

Historical sessions without these facts are not migrated. Views for those sessions may show an empty review board or an explicit legacy-unavailable state.

## Data Model

### ReviewState

Extend `ReviewState` with explicit review interval state:

```python
@dataclass
class ReviewState:
    milestones: list[Milestone] = field(default_factory=list)
    latest_checkpoint: ReviewCheckpoint | None = None
    consecutive_errors: int = 0
    pending_review_reason: Literal["milestone_switch"] | None = None
    review_count_since_checkpoint: int = 0
    pending_review_notice: PendingReviewNotice | None = None
```

`last_review_seq` should stop being the interval cursor. It may remain temporarily during implementation if needed for compatibility inside a staged patch, but new logic should use `review_count_since_checkpoint`.

`review_count_since_checkpoint` increments for every completed tool result except `review_trajectory`. This includes `declare_milestones` and scheduler control tools.

### PendingReviewNotice

Add a small runtime model for an outstanding prompt-visible notice:

```python
@dataclass
class PendingReviewNotice:
    trigger_reason: str
    active_milestone_id: str | None
    review_count_since_checkpoint: int
    trigger_tool_call_id: str | None
    trigger_tool_step_id: str | None
    notice_step_id: str | None
```

This state exists to prevent nested or duplicate review requests. It does not need to be exposed as public API.

### RunLog Facts

#### ReviewMilestonesUpdated

Records the complete milestone board after a successful milestone declaration or status update.

Fields:

- `milestones: list[Milestone]`
- `active_milestone_id: str | None`
- `source_tool_call_id: str | None`
- `source_step_id: str | None`
- `reason: Literal["declared", "updated", "completed", "activated"]`

The fact stores the full board, not only the delta. This keeps replay simple and avoids reconstructing state from multiple partial updates.

#### ReviewTriggerDecided

Records that runtime decided to request a review.

Fields:

- `trigger_reason: Literal["step_interval", "consecutive_errors", "milestone_switch"]`
- `active_milestone_id: str | None`
- `review_count_since_checkpoint: int`
- `trigger_tool_call_id: str | None`
- `trigger_tool_step_id: str | None`
- `notice_step_id: str | None`

There must be at most one `ReviewTriggerDecided` per tool batch.

#### ReviewCheckpointRecorded

Records an aligned checkpoint.

Fields:

- `checkpoint_seq: int`
- `milestone_id: str | None`
- `review_tool_call_id: str | None`
- `review_step_id: str | None`

Only `aligned=true` records a new checkpoint in this design. `aligned=false` keeps the previous aligned checkpoint as the rollback/step-back base.

#### ReviewOutcomeRecorded

Records the result of processing a review.

Fields:

- `aligned: bool | None`
- `mode: Literal["metadata_only", "step_back"]`
- `experience: str | None`
- `active_milestone_id: str | None`
- `review_tool_call_id: str | None`
- `review_step_id: str | None`
- `hidden_step_ids: list[str]`
- `notice_cleaned_step_ids: list[str]`
- `condensed_step_ids: list[str]`

`aligned=None` represents malformed review output that was still consumed to avoid blocking the run.

## Runtime Flow

### Milestone Declaration

1. Agent calls `declare_milestones`.
2. The tool validates parameters and returns normalized milestone data.
3. `ReviewBatch` parses the tool output and updates live `ReviewState` through `goal_manager.declare_milestones`.
4. Runtime appends `ReviewMilestonesUpdated` with the complete milestone board.
5. The milestone tool result is still committed as a normal tool step for debug and trace history.

If the tool succeeds but the payload cannot be parsed into valid milestones, runtime does not write `ReviewMilestonesUpdated`.

### Tool Result Counting

For each successful or failed tool result:

1. If `tool_name == "review_trajectory"`, do not increment `review_count_since_checkpoint`.
2. Otherwise, increment `review_count_since_checkpoint`.
3. Update `consecutive_errors` using the existing success/error semantics.
4. Evaluate review triggers.

`declare_milestones` and scheduler control tools count because they are still tool turns consumed by the agent.

### Trigger Decision

Runtime may request review when:

- `review_count_since_checkpoint >= review_step_interval`
- `review_on_error` is enabled and consecutive tool errors reach the configured threshold
- `pending_review_reason == "milestone_switch"`

Before injecting a notice, runtime must check:

- no review notice has already been requested in this tool batch
- no pending review notice is already present in `ReviewState`

When a review is requested:

1. Runtime appends `ReviewTriggerDecided`.
2. Runtime appends one `<system-review>` block to the trigger tool result.
3. Runtime stores `PendingReviewNotice` in live `ReviewState`.

The prompt-visible `<system-review>` block is only a one-shot control notice. It is not persistent review state.

### Review Completion

When `review_trajectory` succeeds:

1. Runtime reads `aligned` and `experience` from tool output.
2. Runtime hides the review assistant/tool metadata steps.
3. Runtime clears the pending review notice.
4. Runtime removes `<system-review>` from the trigger tool result using targeted content replacement.
5. Runtime writes `ReviewOutcomeRecorded`.

If `aligned is True`:

- write `ReviewCheckpointRecorded`
- set `review_count_since_checkpoint = 0`
- clear `pending_review_reason`

If `aligned is False`:

- do not write a new checkpoint
- execute step-back from the latest aligned checkpoint
- write `ReviewOutcomeRecorded(mode="step_back", experience=...)`
- keep the previous checkpoint as the base for future step-back
- set `review_count_since_checkpoint = 0` after the review is consumed

If `aligned` is missing or invalid after a successful tool result:

- write `ReviewOutcomeRecorded(aligned=None, mode="metadata_only")`
- clear pending notice
- set `review_count_since_checkpoint = 0`
- do not write `ReviewCheckpointRecorded`

### Prompt-Visible Context

The runtime must preserve KV-cache-friendly message shape:

- Do not delete or reorder business messages.
- Hide only temporary review metadata steps.
- Use targeted content replacement to remove `<system-review>`.
- Use targeted content replacement for `[EXPERIENCE] ...` when step-back applies.

Replay must not resurrect a completed `<system-review>` notice.

## Replay

Replay must derive review state from RunLog facts:

- latest `ReviewMilestonesUpdated` gives the milestone board
- `ReviewMilestonesUpdated` also derives `pending_review_reason="milestone_switch"` when replay observes an active milestone transition, a completed milestone status change, or a `reason` of `completed` / `activated`; a later `ReviewTriggerDecided` or `ReviewOutcomeRecorded` clears that transient pending reason. This matches `agiwo/agent/review/replay.py`, which reads milestone facts instead of tool payloads.
- latest `ReviewCheckpointRecorded` gives the checkpoint
- `ReviewOutcomeRecorded` clears pending review state and resets count after consumed reviews
- `ReviewTriggerDecided` can set pending review state only until a matching outcome is recorded
- tool committed facts update `review_count_since_checkpoint` for non-`review_trajectory` tools when needed during replay
- `consecutive_errors` is intentionally runtime-transient and is not reconstructed from RunLog facts.

Replay should not parse `declare_milestones` or `review_trajectory` tool payloads to reconstruct authoritative milestone/review state.

## Console And Trace Views

Console and trace review views must read first-class facts.

Milestone board:

- source: latest `ReviewMilestonesUpdated`
- active milestone: fact field, not description matching from a `<system-review>` block
- latest checkpoint: latest `ReviewCheckpointRecorded`
- latest outcome: latest `ReviewOutcomeRecorded`

Review cycles:

- trigger: `ReviewTriggerDecided`
- outcome: matching or next `ReviewOutcomeRecorded`
- checkpoint: matching `ReviewCheckpointRecorded` when aligned
- step-back status: existing step-back facts plus `ReviewOutcomeRecorded.condensed_step_ids`

Tool spans remain useful for raw debugging but must not be the authoritative source.

Historical sessions without review facts should show no structured review board or a clear legacy-unavailable message. They should not silently reconstruct authoritative state from old tool spans.

## Error Handling

- Invalid `declare_milestones` output: no milestone fact is written.
- Failed `declare_milestones` tool result: no milestone fact is written.
- Failed `review_trajectory`: no outcome is recorded; it remains an ordinary tool error and can contribute to error-triggered review rules.
- Successful malformed `review_trajectory`: write `ReviewOutcomeRecorded(aligned=None, mode="metadata_only")` and clear pending review state.
- Empty `experience` for `aligned=false`: tool layer should continue to fail validation, so no step-back is applied.
- Missing storage while a fact must be persisted is a runtime error. Review state and committed facts must not silently diverge.

## Tradeoffs

This design intentionally chooses a stronger data boundary over backward compatibility.

Benefits:

- Runtime, replay, Console, and trace use one structured source of truth.
- `review_step_interval` matches user expectations better.
- Prompt-visible review notices become one-shot instructions instead of state.
- Future optimizations such as budget, backoff, or soft review can build on stable facts.

Costs:

- More RunLog models and serialization code.
- Console review views for old sessions degrade because no migration is performed.
- Initial implementation touches SDK runtime, storage serialization, observability, and tests.
- `declare_milestones` and scheduler control tools count toward interval, which can trigger review during planning-heavy work. This is intentional for the first stable version because it follows the selected rule: all tools except `review_trajectory` count.

## Testing Requirements

### Review Interval

- Multiple non-`review_trajectory` tool results trigger review after `review_step_interval`.
- `review_trajectory` does not increment the interval counter.
- `declare_milestones` increments the interval counter.
- Scheduler control tools increment the interval counter.
- A single tool batch can produce at most one review trigger.

### RunLog Facts

- Successful milestone declaration writes `ReviewMilestonesUpdated`.
- Review trigger writes `ReviewTriggerDecided`.
- `aligned=true` writes `ReviewCheckpointRecorded` and `ReviewOutcomeRecorded`.
- `aligned=false` writes `ReviewOutcomeRecorded` and step-back facts, but not a new checkpoint.
- Successful malformed review writes `ReviewOutcomeRecorded(aligned=None)`.

### Replay

- Review state can be rebuilt from facts.
- Hidden review metadata does not remove checkpoint or outcome facts.
- Completed review notices do not reappear after replay.

### Prompt Context

- `<system-review>` is removed after review completion.
- aligned=false leaves `[EXPERIENCE] ...` in prompt-visible tool result content.
- Business messages are not deleted or reordered.

### Console And Trace

- Milestone board is built from first-class facts.
- Review cycles are built from first-class facts.
- Legacy sessions without facts do not use old tool-span inference as an authoritative fallback.

### Required Commands

For SDK runtime changes:

```bash
uv run pytest tests/agent/ -v
uv run python scripts/lint.py ci
```

For Console backend observability changes:

```bash
uv run python scripts/check.py console-tests
```

## Open Follow-Ups Outside This Spec

- Add review backoff after repeated `aligned=true`.
- Add per-run review budget or cost budget.
- Consider a soft review path that does not require a dedicated LLM turn.
- Simplify `review_trajectory` so `aligned=true` responses are minimal and `experience` is only meaningful for `aligned=false`.
