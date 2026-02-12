# Templates: For AI Task Plans

## Template 1: Standard Task Plan

For medium-complexity tasks (4–10 steps).

```markdown
# Task: <brief description>

## Diagnosis
- **Uncertainty**: low / medium / high
- **Strategy**: plan-driven / value-driven / hypothesis-driven
- **Key risks**: <what could go wrong>

## Scope
- **Goal**: <what success looks like>
- **Anti-scope**: <what NOT to touch>

## Steps
- [ ] Step 1: <action> (depends on: —)
- [ ] Step 2: <action> (depends on: Step 1)
- [ ] Step 3: <action> (depends on: —, can parallel with Step 2)
  - **Checkpoint (single-loop)**: verify Steps 1–3 output before proceeding
- [ ] Step 4: <action> (depends on: Steps 2, 3)
- [ ] Step 5: <action> (depends on: Step 4)
  - **Checkpoint (double-loop)**: does the approach still make sense?

## Verification
- <how to confirm the task is complete and correct>
```

## Template 2: Exploration / Research Plan

For high-uncertainty tasks where the approach is unclear upfront.

```markdown
# Exploration: <what we're investigating>

## Hypotheses
1. <"I believe X because Y — test by doing Z">
2. <"I believe A — test by checking B">

## Probes
### Probe 1: <small experiment>
- [ ] Action: <what to do>
- [ ] Expected signal: <what would confirm/refute hypothesis>
- **Result**: <fill after execution>

### Probe 2: <small experiment>
- [ ] Action: <what to do>
- [ ] Expected signal: <what would confirm/refute hypothesis>
- **Result**: <fill after execution>

## Decision Point
Based on probe results:
- If hypothesis 1 confirmed → proceed with <approach A>
- If hypothesis 1 refuted → pivot to <approach B>
- If inconclusive → design Probe 3 targeting <specific unknown>

## Execution Plan
<generate after probes complete, using Template 1>
```

## Template 3: Complex / Multi-Phase Task

For large tasks (>10 steps) with cross-cutting concerns.

```markdown
# Task: <brief description>

## Diagnosis
- **Uncertainty**: <level>
- **Strategy**: <chosen strategy>
- **Risks**:
  1. <risk> → mitigation: <action>
  2. <risk> → mitigation: <action>

## Scope
- **Goal**: <measurable outcome>
- **Anti-scope**: <boundaries>

## Phase 1: <name>
- [ ] 1.1: <action>
- [ ] 1.2: <action>
- **Deliverable**: <concrete output>
- **Checkpoint**: <single-loop — did this phase succeed?>

## Phase 2: <name>
- [ ] 2.1: <action>
- [ ] 2.2: <action>
- [ ] 2.3: <action>
- **Deliverable**: <concrete output>
- **Checkpoint**: <double-loop — are assumptions still valid?>

## Phase 3: <name>
- [ ] 3.1: <action>
- [ ] 3.2: <action>
- **Deliverable**: <concrete output>

## Verification
- [ ] <end-to-end check>
- [ ] <regression check if applicable>

## Buffer
- Estimated steps: <N>
- With buffer (×<multiplier>): <M>
```
