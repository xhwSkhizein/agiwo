# Context Rollback & Step-Back Predecessor Design

> Archived predecessor note. This document captures an earlier context-optimization direction that predated the shipped goal-directed review / step-back design.

## What The Earlier Proposal Tried To Solve

The original discussion grouped two different problems together:

1. Periodic scheduler wake-ups that produced no progress but still consumed context.
2. Large or low-value tool outputs that crowded the prompt window even when the only useful takeaway was the lesson learned.

## Ideas That Carried Forward

- Scheduler `no_progress` rollback remained a real feature.
- Original tool output should stay in storage, while prompt-visible context can become shorter.
- The agent should keep the exploration trail, but future context should emphasize the distilled lesson instead of verbose raw output.

## Ideas That Were Replaced

The earlier design relied on threshold-driven tool-result self-review and a dedicated rewrite tool. That approach was later replaced by the shipped goal-directed review flow:

- the system injects milestone-aware `<system-review>` checkpoints
- the agent answers with `review_trajectory`
- the runtime performs step-back by replacing off-track tool-result content with `[EXPERIENCE] ...`
- temporary review metadata stays observable in run-log history but is hidden from future prompt rebuilds

## Why This File Still Exists

This file is kept only as historical context for how the current design evolved. It is not a source of truth for implementation, configuration, or public API.

Use instead:

- `docs/guides/context-optimization.md`
- `docs/superpowers/specs/2026-04-24-goal-directed-review-and-stepback-design.md`
