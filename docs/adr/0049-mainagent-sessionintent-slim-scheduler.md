# MainAgent, SessionIntent, and a slim Scheduler

Status: accepted

We keep ADR 0048's ban on a cross-run **task status machine** (Objective), but the Session + Run + waitset core needs sharper seams for maintainability. This ADR records the **target domain shape** agreed in design: a long-lived **MainAgent** per Session, a lean **SessionIntent** for cross-run alignment, **Worker** for one-shot delegation, **RunPlan** as Run-scoped structured state beside RunLog, and a **Scheduler** reduced to waitset + cancel-subtree mechanics. Implementation may still expose a fatter Scheduler facade; new work must move toward this shape, not enlarge the facade.

## Decisions

### MainAgent is the Session-scoped executor (T2)

**AgentSpec** (reusable config/template: model, tools, skills, options) is separate from **MainAgent** (Session-bound live executor). Session holds a MainAgent; many Sessions may bind the same spec. Do not keep a parallel non-Session `Agent.run` entry beside MainAgent (M3).

MainAgent outlives individual Runs (idle ≠ destroyed). External user input enters through **`accept`**. Internally, user mid-run input, completion-gate feedback, and Worker reports share **one enqueue primitive and one Loop message queue**—no separate steer vs inject APIs. If the Main Loop is running, enqueue; otherwise `accept` opens a new Run. Prefer the name MainAgent over ExecutionHandle / SessionRuntime in domain language.

### Session history and RunLog share one ledger

User-visible Session history is **not** a separate message table. Canonical facts live in RunLog (`run_log_entries`); UI history and model `messages` are projections. Console `SessionStore` holds Session metadata only. Do not invent a second user message store for history. (SessionIntent may deliberately denormalize user full text for alignment—see below—without replacing RunLog.)

### SessionIntent (not Objective)

**SessionIntent** is a Session-level alignment view: full user inputs (no summary) plus summarized Run reports, updated (a) on new user input and (b) when a Run ends. Optional `last_run_plan` snapshot after a Run. First version does **not** LLM-extract `core_goal` / roadmap / decisions (model distortion).

**Persistence (I-S1):** store SessionIntent in its **own table/document** keyed by `session_id` (not a column on `console_session`, not a pure RunLog projection). D1 denormalization stands: user full text is copied into Intent on `accept` while RunLog remains the execution/UI ledger. SessionIntent is a **compass**, not an engine or progress state machine: no ObjectiveStatus, no cross-run dispatch ledger. When **semantic completion gates** are enabled (I2), SessionIntent is a primary check standard alongside the current Run’s facts.

### RunPlan beside RunLog

**RunPlan** / milestones are structured state owned by the **current Run**, not Session-scoped todos and not reconstructed by parsing tool-output text. RunLog remains the full event ledger used to build model messages and to drive introspect/compaction. Plan changes may also append first-class RunLog facts for audit/replay. A new Run starts with an empty plan; on successful completion the plan freezes (snapshot may land on SessionIntent).

### Worker delegation (distinct from Agent-as-Tool)

Only MainAgent may spawn **Workers**. A Worker has an isolated context, produces a final **report**, cannot be reused or restarted, and cannot spawn further Workers. Sync: Main waits via Scheduler waitset. Async: Main continues; on Worker completion the system notifies Main by **enqueue** onto the same Loop queue as user input / gate feedback and writes the report into the main context; if Main Loop already finished, **resume the same Run**. Cancel Main cancels its Workers.

**Agent-as-Tool** (`Agent` exposed as a normal tool) is **not** a Worker and is **not** part of the Scheduler delegation tree. It is a functional tool: nested model+tool work happens inside a tool call, subject to tool rules (`allowed_tools`, depth limits, abort with the parent tool batch). It does not use waitset/Worker lifecycle, does not produce Session-level Worker reports, and must not be redescribed as “another child agent.” **Scheduler Workers and Agent-as-Tool coexist** as separate mechanisms; M3 removes the *confusion* and dual *delegation* semantics, not Agent-as-Tool itself.


### Slim Scheduler

Scheduler’s irreplaceable core is **waitset + cancel subtree** (and registering wait edges for Workers). It must not own user-visible history semantics, interpret RunPlan, or be the sole product facade for “run the session.” User entry / accept semantics belong to Session (Gateway) + MainAgent.accept; opening Loop and appending RunLog belong to MainAgent / Run runtime.

### Completion gates

Before Main may stop a Run: **mechanical gates** always apply (e.g. unfinished Workers, open RunPlan milestones). **Semantic gates** are optional, off by default, enabled by simple complexity signals (e.g. milestones existed, model-call rounds ≥ N, Workers spawned). When on, they use SessionIntent as an important criterion (I2). Gates only read facts / Intent and emit Continue(feedback) or AllowComplete—feedback re-enters the unified Loop queue; they must not become a second progress plane.

**First delivery (G-v1a):** ship **mechanical gates only**; keep the semantic-gate seam but do not turn it on in v1. Rejected for the first cut: shipping semantic gates in the same wave (G-v1b) or deferring all gates past the M3 cut (G-v1c).

### Worker ledger isolation (L1-a)

Workers share the Session’s RunLog store and `session_id` but use a **distinct `agent_id`** (and their own `run_id` for one-shot execution). Main Loop builds messages with `session_id + main agent_id` so prior main Runs remain visible; **do not** key main context solely by `run_id` (that drops earlier-turn history). Worker step traces stay out of the main projection; only the final report is enqueued onto Main’s shared queue / written into the main context.

### Messages

Model `messages` are built from RunLog (plus optional short renders such as current RunPlan or SessionIntent excerpts). Introspect and compaction operate on RunLog-derived views.

## Considered options

- Keep Scheduler as the Session runtime facade (route/steer/history-adjacent)—rejected as the main source of accidental complexity; slim to waitset/cancel.
- Session-scoped live RunPlan / LLM-maintained core_goal as end conditions—rejected for v1; reintroduces Objective-shaped drift and distortion.
- Drop SessionIntent and rely only on RunLog projections—rejected; cross-run alignment needs an explicit, small read model (user full text + report summaries).
- New Run when an async Worker finishes after Main stopped—rejected; resume the same Run.
- Pure waitset-only module with cancel-subtree owned elsewhere—deferred; cancel-with-parent stays with Scheduler for one clear owner.
- Treat Agent-as-Tool as a second Worker/child-agent protocol—rejected; Agent-as-Tool is a functional tool, independent of Worker/waitset.

## Migration strategy

**M3 — full package cut, no legacy facade.** Refactor to the seams in this ADR in one coherent redesign (not a thin MainAgent wrapper over today’s Scheduler). After the cut: no parallel root-entry APIs, no dual “SessionRuntime” names for the live executor, no second *delegation* model (Agent-as-Tool remains as a tool). **Data is fail-closed:** no schema migration; clear development/persistent stores. Old RunLog / AgentState shapes may be rejected outright.

**Delivery waves (P2):** first mergeable vertical slice is **A+B together**—AgentSpec + MainAgent + unified queue + Gateway/`accept` + RunLog write path + slim Scheduler *interfaces* enough to run a Session turn—then stack **C** SessionIntent (I-S1), **D** Worker (L1-a), **E** mechanical gates (G-v1a), **F** delete leftover names/APIs. Rejected: P1 (too fragmented before a working path) and P3 (single-bang A–F).

Rejected as end-state strategy: M1 (MainAgent facade over fat Scheduler) and M2 (slim Scheduler first, accept later)—they preserve two mental models and prolong the debt this ADR exists to end.

## Consequences

- ADR 0048 remains in force: no Objective-like cross-run task status machine. SessionIntent must not grow into one.
- Implementation work is a breaking redesign of agent/scheduler/console wiring toward MainAgent / SessionIntent / Worker / slim Scheduler; do not keep the fat Scheduler facade as a supported end state.
- SessionIntent storage and D1 denormalization land in the new shape; dual-writing to obsolete tables is unnecessary if those tables are deleted.
- Truth order unchanged: CONTEXT.md → this ADR with 0048 → source.
- Historical M3 task breakdown (complete; not live specs): [`trash/docs-historical-2026-07-23/superpowers/plans/2026-07-23-mainagent-m3-refactor/`](../../trash/docs-historical-2026-07-23/superpowers/plans/2026-07-23-mainagent-m3-refactor/).
