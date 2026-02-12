---
name: making-dynamic-plan
description: Create adaptive, living plans using OODA-COP methodology. Two modes — (1) help a human plan a project, or (2) plan the agent's own multi-step task. Use when the task involves complex multi-step work, significant unknowns, evolving requirements, or any scenario where a static checklist is insufficient. Triggers on "help me plan", "make a plan for", "how should I approach", "I need a strategy for", or when self-planning a complex task.
---

# Making Dynamic Plans (OODA-COP)

A dynamic plan is a living system that senses context, self-corrects, and adapts — not a static checklist. Follow the three core engines below **in order**, then apply the appropriate **adaptation mode**. Revisit earlier engines whenever new information invalidates prior assumptions.

## Two Modes

This skill operates in two modes. Choose based on context:

| Mode | When | Adaptation Guide |
|------|------|------------------|
| **For Human** | User asks you to help plan a project, strategy, or initiative | [for_human/adaptation.md](for_human/adaptation.md) |
| **For AI** | You need to plan your own multi-step task execution | [for_ai/adaptation.md](for_ai/adaptation.md) |

## Engine 1: Diagnose — Sense the Terrain

Classify the problem's uncertainty level to select the right planning strategy:

| Uncertainty | Strategy | Signal |
|-------------|----------|--------|
| **Low** | Plan-driven — heavy upfront design | Requirements are clear, domain is well-known, few unknowns |
| **Medium** | Value-driven — prioritize by impact | Some known-unknowns, scope may shift, partial information |
| **High** | Hypothesis-driven — plan = experiment list | Unknown-unknowns dominate, novel domain, high ambiguity |

**How to diagnose:**

1. Extract from available context: goal, constraints, deadline, executor, known risks
2. If critical information is missing (goal unclear, no success criteria, unknown executor), **ask specific clarifying questions** — do not proceed with assumptions on critical unknowns
3. Classify into three buckets: **knowns** / **known-unknowns** / **suspected unknown-unknowns**
4. Select the matching strategy from the table above
5. Produce a **Situation Diagnosis**: complexity level + chosen strategy + top risks with hedges

## Engine 2: Structure — Decompose into a Task Graph

### Step 1: Anchor scope with PSA

- **P (Problem)**: Whose pain are we solving? What is the core need?
- **S (Success)**: Define 2–3 measurable success criteria. If the executor cannot state them, help derive them from the goal.
- **A (Anti-scope)**: Explicitly list what is OUT of scope. This prevents scope creep.

### Step 2: Model the problem type and decompose

| Problem type | Signal | Decomposition method |
|--------------|--------|----------------------|
| **Linear** | Clear cause-effect, sequential steps | Critical-path analysis — identify which delays cascade |
| **Complex** | Multiple interacting parts, emergent behavior | Probe-sense-respond — start with small experiments, expand what works |
| **Chaotic** | No clear cause-effect, crisis mode | Stabilize first (contain damage), then analyze |

### Step 3: Calibrate granularity

The right level of detail depends on who executes and how coupled the tasks are:

| Context | Granularity |
|---------|-------------|
| Expert executor, loosely coupled tasks | Goals + constraints + key milestones |
| Mixed team or moderate coupling | Phase-level with clear deliverables per phase |
| Novice executor, tightly coupled tasks | Next-action level (each step completable in ≤15 min) |
| Unknown executor | Default to **phase-level**, let executor refine |

**Output**: a **Dynamic Work Breakdown** — a task graph with dependencies, not a flat list.

## Engine 3: Evolve — Embed Feedback & Checkpoints

A plan without feedback sensors is dead on arrival. This engine ensures the plan **monitors itself**.

### Checkpoint types

Every plan must embed two types of checkpoints:

- **Single-loop** (progress check): "Are we on track against the plan?" → correct action drift
- **Double-loop** (assumption check): "Are the premises behind this plan still valid?" → correct goal drift or trigger re-planning

### Pre-embed checkpoint definitions

For each checkpoint in the plan, explicitly define:

1. **Trigger**: when does this checkpoint fire? (after phase X / after N days / on event Y)
2. **Check type**: single-loop or double-loop
3. **Questions to answer**: specific questions, not vague "how's it going"
4. **Decision rule**: what action to take based on the answer (proceed / adjust / re-plan / escalate)

### Rhythm calibration

Match checkpoint frequency to the current phase:

| Phase | Cadence | Rationale |
|-------|---------|-----------|
| Exploration | Every 1–2 cycles (short) | Fast feedback on hypotheses |
| Execution | Every 3–5 cycles (medium) | Steady progress tracking |
| Wrap-up | End-of-phase retrospective | Capture lessons, release resources |

**Principle**: if the environment changes faster than your checkpoints fire, you are flying blind. Increase frequency.

### Buffer management

Allocate slack as a **centralized buffer pool** rather than padding each task:

- Estimate total buffer based on uncertainty level (high uncertainty → larger buffer)
- Track buffer consumption across the plan
- **Alert threshold**: when 50% of buffer is consumed before 50% of work is done, trigger a double-loop checkpoint

## Adaptation

After completing the three engines above, apply the mode-specific adaptation:

- **For Human**: read [for_human/adaptation.md](for_human/adaptation.md) — covers audience style matching, autonomy, and meaning framing
- **For AI**: read [for_ai/adaptation.md](for_ai/adaptation.md) — covers self-planning heuristics, tool usage, and progress tracking

## Output Templates

Use the appropriate template based on mode:

- **For Human**: [for_human/templates.md](for_human/templates.md) — project plans, strategic plans, learning paths
- **For AI**: [for_ai/templates.md](for_ai/templates.md) — task execution plans with tool integration

## When to Re-plan

Re-enter at Engine 1 whenever:

- A double-loop checkpoint reveals invalid assumptions
- Major scope change, new constraint, or new stakeholder appears
- Buffer pool crosses the alert threshold
- The chosen strategy no longer fits (e.g., uncertainty dropped from high to low)

## Full Methodology

For the complete OODA-COP theoretical framework, see [references/ooda-cop.md](references/ooda-cop.md).
