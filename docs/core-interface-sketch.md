# Core interface sketch (ADR 0048 + ADR 0049)

Documentation-level contract for the minimal core. Implementation may rename helpers but must preserve these seams and invariants. Vocabulary: **Session**, **MainAgent**, **Run**, **RunLog**, **Scheduler**, **Worker**, **委派**.

## Session

Owner of conversation identity. User-visible history is a **RunLog projection**. Single user entry for Console/channels.

```text
MainAgent.accept(user_message) -> None
  invariants:
    - append user_message to RunLog history exactly once (is_user_provided=true)
    - append SessionIntent user entry (D1 full text)
    - if Main Loop busy: enqueue onto the shared Loop queue
    - else: open one new root Run
    - never create Objective / Turn aggregate / second user ledger
    - never call Scheduler.route_root_input / dispatch_execution / inject_user_message / steer

subscribe() / wait_for_current_run() / cancel(...)
```

Console `SessionGateway` is the HTTP/channel adapter over `accept`.

## Scheduler

Owner of child / Worker waitset and cancel subtree. Does **not** own Session chat entry or user-visible history.

```text
enqueue_input(state_id, user_input, *, agent?) -> None
  # persistent root only:
  #   IDLE/FAILED → pending_input / QUEUED
  #   RUNNING → live handle enqueue_message
  #   WAITING/QUEUED → USER_HINT mailbox (WAITING urgent)

register_worker_parent(...)
cancel(...) / shutdown(...) / wait_for(...)
list_states / list_events / get_stats / rebind_agent
spawn_child_agent / fork_child_agent / sleep_and_wait  # system tools
```

Invariants:

- `WAITING` on AgentState means waitset/timer/events — not a Session user-entry facade
- Session product input is `MainAgent.accept` only
- Deleted Session facade: `route_root_input`, `dispatch_execution`, `inject_user_message`, `steer`

## Agent + RunLog

```text
Agent.start / handle.enqueue_message / handle.stream / handle.wait
RunLog append-only facts → RunView / StepView / stream / trace projections
SessionRuntime: one pending-input queue (enqueue_message / peek / ack)
```

One Loop queue. No public `enqueue_steer` / `enqueue_inject` / dual drain.
