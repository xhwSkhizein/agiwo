# Adaptation: For AI Self-Planning

After completing the three core engines (Diagnose → Structure → Evolve), apply these heuristics when planning your own task execution.

## Diagnosis Heuristics

When self-planning, you already have the task context. Apply these rules to classify quickly:

| Signal in the task | Uncertainty level | Strategy |
|--------------------|-------------------|----------|
| Single well-defined file change, clear requirement | Low | Plan-driven: list steps, execute in order |
| Multi-file change, partial requirements, some research needed | Medium | Value-driven: identify highest-impact work first, iterate |
| Novel domain, unclear approach, need to explore before committing | High | Hypothesis-driven: formulate hypotheses, test with small probes, adapt |

**Default**: if you cannot classify within 30 seconds of reading the task, treat it as **Medium**.

## Granularity Rules

| Task scope | Plan granularity |
|------------|------------------|
| ≤3 steps, single concern | Mental model only — no written plan needed |
| 4–10 steps, moderate complexity | Phase-level plan with key milestones |
| >10 steps, cross-cutting concerns, or multi-session | Full task graph with dependencies and checkpoints |

## Self-Planning Principles

1. **Plan before act**: for non-trivial tasks, produce the plan BEFORE making any changes
2. **One step in progress**: keep exactly one task in_progress at a time
3. **Update after discovery**: when new information arrives (unexpected error, new constraint, changed requirement), update the plan immediately — do not continue with a stale plan
4. **Checkpoint discipline**: after completing each phase, pause and evaluate:
   - Single-loop: "Did this phase produce the expected output?"
   - Double-loop: "Does the remaining plan still make sense given what I just learned?"
5. **Fail fast, re-plan early**: if an approach fails twice, do not retry — re-enter Engine 1 with the new information

## Buffer Estimation

For time/effort estimation in your plans:

| Uncertainty | Buffer multiplier |
|-------------|-------------------|
| Low | 1.2× (add ~20% slack) |
| Medium | 1.5× (add ~50% slack) |
| High | 2.0× (double the estimate, or split into smaller experiments) |

In practice: if you estimate a task at 5 steps, a medium-uncertainty plan should anticipate up to ~8 steps.

## Progress Tracking

Use the todo_list tool to track execution. Map plan phases to todo items:

- Each phase → one todo item
- Mark `in_progress` when starting, `completed` when done
- If a phase needs to be split, add sub-items rather than letting one item grow unbounded

## When to Skip Planning

Not every task needs a formal plan. Skip to direct execution when:

- The task is a single, well-understood action (e.g., rename a variable, fix a typo)
- You have done this exact type of task before with no surprises
- The blast radius of a mistake is minimal (easy to undo)
