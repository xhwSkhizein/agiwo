# Agent introspection

Trajectory review and RunPlan introspection for the agent runtime.

## Modules

| Module | Responsibility |
| --- | --- |
| `models.py` | `IntrospectionState`, `IntrospectionOutcome`, usefulness scores. |
| `trajectory.py` | `<system-review>` trigger detection, notice rendering, outcome parsing. |
| `tool.py` | `ReviewTrajectoryTool` system tool (append-only review metadata). |
| `window.py` | Build review windows from committed tool results. |
| `apply.py` | Append-only outcome application and RunLog fact writes. |
| `replay.py` | Rebuild introspection state from committed RunLog facts. |
| `compaction_hints.py` | Optional experimental usefulness hints for compaction. |

## Tool ownership

`update_plan` and `review_trajectory` are agent-owned system tools assembled in
`Agent._owned_system_tools()` when enabled. Scheduler runtime tools no longer
include `review_trajectory`.

## Configuration

```text
context.config.enable_trajectory_review
and "review_trajectory" in runtime.tools_map
```

## Append-only review semantics

`review_trajectory` records alignment, optional experience, and per-tool
usefulness scores. It does not hide steps, rewrite tool results, or remove
review metadata. Review tool calls, results, notices, and prior messages stay
in prompt-visible history.

## RunLog facts

| Fact | Writer | Consumers |
| --- | --- | --- |
| `IntrospectionTriggered` | `run_tool_batch.py` | Replay, trace, Console review cycles. |
| `IntrospectionCheckpointRecorded` | `apply.py` | Milestone board checkpoints. |
| `IntrospectionOutcomeRecorded` | `apply.py` | Replay, trace, Console review cycles. |

`IntrospectionOutcomeRecorded` stores `tool_usefulness` entries with scores
`0-3` or `null` for unknown tools in the review window.
