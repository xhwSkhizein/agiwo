# Session + Run + Waitset is the minimal core

Status: accepted

Agiwo's Objective layer (and the Assignment era before it) stacked a second work-progress state machine on top of Session, Run, and Scheduler waitset. That made maintenance cost grow with every "simplification." We choose a Pi-style minimal core instead: **Session** owns conversation and user-visible history; **Run** (root vs child) is the only execution unit; **RunLog** is the only execution ledger; **Scheduler** owns the child delegation tree and waitset. There is no cross-run task ledger in core.

We do **not** introduce Turn as an aggregate or API—user submit starts a root Run. Cross-run "long tasks," budgets, verification handoffs, and Objective SSE are out of core; if needed later they must be observers of RunLog or explicit additional root Runs, never a parallel status machine.

This decision supersedes ADR 0003, 0018, 0021, 0022, 0034, 0035, 0044, 0046, 0047, and the broader Objective/Assignment control chain listed in `docs/adr/README.md`. Switching is breaking: clear development databases; no migration.

## Considered options

- Keep on-demand Objective escalation (0047) and keep shrinking it — rejected; the second progress plane remained the root of accidental complexity.
- Introduce Turn as Session-level aggregate wrapping root Run — rejected; 1:1 with root Run, adds vocabulary without depth.
- Minimal Session + Run + waitset core (chosen).
