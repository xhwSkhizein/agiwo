# Objective Observability

How to explain an Objective’s status, cost, and pauses without treating metrics as a second domain truth.

## Truth sources

| Question | Source |
| --- | --- |
| Current status, timeline, delivery | `ObjectiveView` from ObjectiveLog |
| Budget limit / used / remaining | `ObjectiveBudget` on that view |
| Aggregated counts and elapsed time | `GET /api/objectives/{id}/metrics` → `project_objective_metrics` |
| Per-run LLM tokens, tool spans, review latency | RunLog / Trace for runs under the Objective |
| Scheduler tree identity | Scheduler AgentState (not ObjectiveBudget) |

Metrics APIs **replay committed facts**. Replaying the same ObjectiveLog must not invent new counts.

## Objective metrics fields

`ObjectiveMetrics` (SDK: `agiwo.objective.project_objective_metrics`) exposes:

- status / terminal flag / delivery_count
- assignment counts by kind (intake / work / verification)
- user_input / artifact / timeline / system_fault counts
- budget dimensions (handoffs, verification_attempts, llm_cost_usd, active_seconds)
- timestamps and total_elapsed_seconds

`truth_source` is always `"ObjectiveLog via ObjectiveView"`.

## Trajectory review

`enable_trajectory_review` remains the AgentConfig default. Review tool spans may attach `review_latency_ms` on Trace review cycles. Those numbers are **experimental** compaction-adjacent metadata:

- Prefer measuring extra latency/tokens from Trace/RunLog, not from ObjectiveLog.
- Scores must not drive deterministic message deletion thresholds.
- This is not a general benchmark framework (`docs/eval-draft` is out of scope).

## Prefix cache

Adjacent Assignment stable-prefix reuse is observed via RunLog/LLM cache_read tokens and compaction/rebuild facts. Loss of prefix should map to an explicit rebuild/compaction reason, not silent rewrite.

## Logging

Use structured `logger.{level}("event_name", key=value, ...)`. Do not put prompt/content into ordinary metrics labels.
