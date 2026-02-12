# Templates: For Human Plans

Choose the template that best fits the planning scenario. Adapt sections as needed — these are skeletons, not rigid forms.

## Template 1: Project Plan (General Purpose)

```markdown
# Plan: <title>

## Situation Diagnosis
- **Complexity**: low / medium / high
- **Strategy**: plan-driven / value-driven / hypothesis-driven
- **Key risks**:
  1. <risk> → hedge: <mitigation>
  2. <risk> → hedge: <mitigation>

## Scope (PSA)
- **Problem**: <who has what pain>
- **Success criteria**:
  1. <measurable outcome>
  2. <measurable outcome>
  3. <measurable outcome>
- **Anti-scope**: <what we will NOT do>

## Task Graph

### Phase 1: <name>
- [ ] Task 1.1: <description> (depends on: —)
- [ ] Task 1.2: <description> (depends on: 1.1)
- **Deliverable**: <what this phase produces>

### Phase 2: <name>
- [ ] Task 2.1: <description> (depends on: Phase 1)
- [ ] Task 2.2: <description> (depends on: 2.1)
- **Deliverable**: <what this phase produces>

## Checkpoints
| ID | Trigger | Type | Key Questions | Decision Rule |
|----|---------|------|---------------|---------------|
| CP-1 | After Phase 1 | Single-loop | Did Phase 1 deliverable meet criteria? | Proceed / adjust Phase 2 |
| CP-2 | After Phase 2 | Double-loop | Are original assumptions still valid? | Proceed / re-plan from Engine 1 |

## Buffer Pool
- **Total buffer**: <X days/hours>
- **Alert threshold**: 50% consumed before 50% complete → trigger double-loop review
```

## Template 2: Strategic Initiative

```markdown
# Strategy: <initiative name>

## Context & Diagnosis
- **Current state**: <where we are>
- **Desired state**: <where we want to be>
- **Uncertainty level**: low / medium / high
- **Strategy**: plan-driven / value-driven / hypothesis-driven
- **Key assumptions** (to be validated):
  1. <assumption>
  2. <assumption>

## Scope (PSA)
- **Problem**: <strategic gap>
- **Success criteria**: <2–3 north-star metrics>
- **Anti-scope**: <strategic boundaries>

## Workstreams

### Workstream A: <name>
- **Owner**: <role/team>
- **Milestones**:
  1. <milestone> — target: <date/condition>
  2. <milestone> — target: <date/condition>

### Workstream B: <name>
- **Owner**: <role/team>
- **Milestones**:
  1. <milestone> — target: <date/condition>

## Dependencies
- Workstream B.1 depends on Workstream A.2

## Checkpoints
| ID | Trigger | Type | Key Questions | Decision Rule |
|----|---------|------|---------------|---------------|
| CP-1 | <when> | Single-loop | <specific question> | <action> |
| CP-2 | <when> | Double-loop | Are key assumptions still valid? | Continue / pivot / stop |

## Buffer & Contingency
- **Buffer**: <amount>
- **Contingency plan**: if <risk event>, then <response>
```

## Template 3: Learning / Exploration Path

```markdown
# Learning Plan: <topic>

## Diagnosis
- **Current knowledge**: <what I already know>
- **Target competency**: <what "done" looks like>
- **Uncertainty**: high (exploration — hypothesis-driven)

## Hypotheses to Test
1. <"I believe learning X will enable Y">
2. <"I believe approach A is better than B for my context">

## Phases

### Phase 1: Orientation (<timeframe>)
- [ ] Survey the landscape: <resources to review>
- [ ] Identify key concepts and skill gaps
- **Checkpoint**: Can I articulate the top 3 things I need to learn?

### Phase 2: Focused Practice (<timeframe>)
- [ ] Deep-dive into <topic A>
- [ ] Build <small project / exercise>
- **Checkpoint**: Can I apply concept X without reference material?

### Phase 3: Integration (<timeframe>)
- [ ] Apply to real problem: <specific use case>
- [ ] Seek feedback from <source>
- **Checkpoint (double-loop)**: Was my learning hypothesis correct? Adjust path if not.

## Success Criteria
1. <concrete demonstration of competency>
2. <practical output produced>
```
