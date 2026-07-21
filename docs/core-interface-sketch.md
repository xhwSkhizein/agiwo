# Core interface sketch (ADR 0048)

Documentation-level contract for the minimal core. Implementation may rename helpers but must preserve these seams and invariants. Vocabulary: **Session**, **Run** (root/child), **RunLog**, **Scheduler**, **委派**.

## Session

Owner of conversation identity and user-visible history. Single user entry for Console/channels.

```text
submit_user_message(session_id, user_input: UserMessage, *, idempotency_key?) -> RootRunHandle
  invariants:
    - append user_input to Session history exactly once (is_user_provided=true)
    - if a root Run for this session is RUNNING: inject/steer that run; do not start a second root
    - else: start one new root Run via Scheduler (user_input=None on dispatch; history already has the text)
    - never create Objective / Turn aggregate / second user ledger

get_history(session_id, ...) -> list[StepView | message projection]
get_current_root_run(session_id) -> run_id | None
```

`RootRunHandle` exposes wait/stream/cancel over the root execution tree; it is not a Turn type.

## Scheduler

Owner of child delegation tree and waitset. Does not own cross-run task ledgers.

```text
# Session-facing (one root start verb)
dispatch_execution(request: SchedulerExecutionRequest) -> None
  # request carries preallocated or generated run_id, agent, optional thin system notice
  # Session path sets user_input=None when history already holds the user text

inject_user_message(run_id, user_input) -> None   # RUNNING root only
steer(...) / cancel(...) / wait_for(...)
list_execution_tree(root_run_id) -> depth-1 root + direct children
get_run_view / get_run_status / list_run_log_entries   # read bridges over agent RunLog
request_recoverable_pause / prepare_resume / release_resume_barrier  # Run-level PAUSED

# Agent-facing system tools (unchanged responsibility)
spawn_child_agent / fork_child_agent / sleep_and_wait(waitset|timer|...)
```

Invariants:

- `WAITING` on AgentState means waitset/timer/events — not "waiting for user on a task ledger"
- One public path for "start this root Run"; mailbox `route_root_input` may remain for SDK/debug but Console must not use a second task plane

## Agent + RunLog

```text
Agent.start / start_prevalidated / run_stream
RunStateWriter — sole write path for RunLog facts
RunStatus: RUNNING | PAUSED | COMPLETED | INTERRUPTED | FAILED
```

Invariants:

- RunLog is the only execution truth; Session history and traces project from it
- Scheduler does not write RunLog (except documented fork/rollback edge cases)
- No ObjectiveLog dual-write

## Explicit non-goals (this sketch)

- `submit_turn` / `TurnHandle`
- `ObjectiveService`, `upgrade_from_plain_run`, `RootRunRequested`
- Automatic verification / handoff peer root from a control plane
