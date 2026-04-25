# Goal-Directed Review And Stepback Implementation Plan

> Historical note: this plan was superseded after later broad renames made many legacy/new symbols indistinguishable. Do not use this file as executable implementation guidance.

This path is intentionally kept as a lightweight stub so existing references still land somewhere understandable, but the detailed task breakdown that used to live here is no longer trustworthy.

Use these documents instead:

- `docs/superpowers/specs/2026-04-24-goal-directed-review-and-stepback-design.md`
- `docs/superpowers/plans/2026-04-25-goal-directed-review-cleanup-and-closure.md`
- `docs/superpowers/specs/2026-04-25-console-trace-runtime-observability-design.md`

Scope clarification:

- The canonical terminology is `review`, `step_back`, `ReviewBatch`, `StepBackApplied`, and `agiwo/agent/review/`.
- Legacy `retrospect` names are retained only where a document explicitly discusses historical predecessors.
- If you need the exact shipped behavior, verify it against the current code, not against the removed task list from the old plan.
