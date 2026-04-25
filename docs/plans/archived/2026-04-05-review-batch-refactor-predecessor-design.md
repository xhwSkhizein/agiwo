# Review Batch Refactor Predecessor Design

> Archived predecessor note. This document records an earlier attempt to pull tool-result review cleanup out of `run_loop.py` and into a dedicated batch abstraction.

## What The Earlier Proposal Got Right

- A single batch object should own the per-tool-batch review lifecycle.
- `run_loop.py` should coordinate high-level phases, not manually juggle review state variables.
- Storage writes, prompt-visible content changes, and review cleanup should return a structured outcome instead of being hidden behind scattered side effects.

## What Changed In The Shipped Design

The shipped runtime kept the batch-ownership idea but moved it onto the current goal-directed review model:

- `ReviewBatch` lives in `agiwo.agent.review`
- milestone state is tracked through `ReviewState`
- review cleanup returns structured `StepBackOutcome`
- prompt-visible cleanup happens through targeted `content_updates` plus hidden-from-context facts
- current review metadata is removed from future prompt assembly without rewriting the whole message list

## Why This Predecessor Is Archived

The earlier draft assumed a threshold-driven tool-result rewrite flow and an internal package layout that no longer exists. Those details were intentionally replaced once goal-directed review and step-back became the canonical design.

Use instead:

- `docs/guides/context-optimization.md`
- `docs/superpowers/specs/2026-04-24-goal-directed-review-and-stepback-design.md`
