# 2026-03-26 Console Remote Workspace Refactor Design

## Summary

This design reframes the current Console refactor as a structural product alignment effort rather than a feature expansion effort. The immediate goal is to make the Console viable for SDK developers who use it as a remote operations workspace for agents. The core problem is not missing surface features; it is that the current Console code structure and interaction model do not cleanly match the intended product semantics.

The refactor therefore focuses on two problems first:

1. clarifying the state model around session, task, and run
2. unifying Console and Feishu behind the same interaction semantics

The result should be a Console that acts as a thin product-facing projection layer over SDK execution facts, not as a second execution system with its own parallel truth.

## Problem Statement

The repository currently contains an agent SDK plus a Console control plane, with Feishu support as a remote interaction channel. The basic capabilities already exist, but the current implementation is still too close to a demo:

- the Console interaction model is not yet cleanly aligned with actual usage
- the Web experience is not yet strong enough to feel like a durable developer workspace
- Console and Feishu expose overlapping capabilities through uneven abstractions
- the current Console implementation has too much complexity relative to the clarity of the underlying product model

The most urgent need is not adding more functionality. It is reducing Console-side complexity and re-aligning the implementation with the intended remote-operations product model.

## Goals

### Primary Goal

Refactor the Console into a maintainable remote workspace for SDK developers, with a product model centered on sessions and tasks, and a runtime model that consistently routes execution through the scheduler.

### User

The primary user for this phase is the SDK developer, using the Console to:

- interact with agents remotely
- inspect what happened during execution
- switch between ongoing work contexts
- continue work through either Console or Feishu

### In-Scope Outcomes

- define a single interaction model shared by Console and Feishu
- make Session / Task / Run semantics explicit and consistent
- make the Console a projection over SDK-layer execution facts
- reduce architectural complexity in the current Console implementation
- preserve support for remote execution and later observability improvements

### Out of Scope

- broad feature expansion unrelated to the refactor
- redesigning the SDK execution model as part of this spec
- deciding in this document whether SDK Trace should be fully removed
- exposing low-level execution detail in the default main workflow

## Product Model

### Core Domain Objects

#### Session

A Session is one complete conversation context. It is the primary container in the remote workspace model. Users should be able to create, switch, and resume sessions across entrypoints.

#### Task

A Task is a unit of work inside a Session. Task is a first-class domain object, but it is not intended to be a first-class manual action in the default workflow.

In the normal case:

- a user enters or switches to a session
- the user sends the first message
- the system implicitly creates the current task for that session
- subsequent messages append to the active task

The default mental model is therefore simple even though Task remains explicit in the data model.

#### Run

A Run is an execution-level realization of task work. The Console should treat Run as a derived execution view backed by SDK facts rather than as an independently authored business object.

## Default Interaction Semantics

### Session-First Workflow

The unified workflow for Console and Feishu should be:

1. select or create a session
2. send a message within that session
3. implicitly create a task if the session does not already have the relevant active task
4. route execution through the scheduler
5. project resulting SDK execution facts back into the UI as task/run views

### One Session, One Task by Default

The default expectation should be:

- one session maps to one task

This keeps the product mental model simple and aligns with most real usage.

### Multiple Tasks Only for Strongly Related Serial Work

A session may contain multiple tasks only when the work is strongly sequential and contextually continuous. This should be an exception, not the main model.

The refactor should avoid forcing users to think in workflow-engine terms during ordinary use.

## Fork Model

Fork is a key product capability.

When a user is working on task A and discovers an adjacent but independent task B, the system should support forking the current context into a new session rather than encouraging unrelated work to accumulate inside one session.

### Fork Behavior

- create a new session
- create a new task in that new session
- copy forward the chosen context from the source conversation
- keep the new session operationally independent from the source session
- preserve only weak references between source and forked lineage

### Weak Association Rules

Fork should preserve lineage metadata such as:

- source session id
- source task id
- optional summary or selected context snapshot used during fork

Fork should not create shared live runtime state across sessions. It is a context copy, not a runtime merge.

## Unified Entry Architecture

The target runtime interaction path is:

**Console / Feishu → Session application layer → Scheduler → Agent**

### Architectural Meaning

This implies:

- Console and Feishu are both entry adapters
- product semantics are not defined by the entrypoint
- scheduler-mediated execution becomes the default and consistent execution path
- the Console should no longer preserve a separate direct-to-agent product path as its primary semantics

This unification is necessary because the current inconsistency between Console and Feishu contributes directly to the state-model confusion and implementation complexity.

## Projection Boundary

A critical design rule for the refactor is:

**The Console constructs views over SDK-layer facts; it does not create a second execution truth model.**

### SDK Fact Sources

The current relevant SDK execution facts are:

- RunStep
- Trace

For this refactor, the Console should be designed to project from SDK-provided execution records rather than invent new authoritative runtime objects.

### RunStep-First Direction

The current design should prefer a RunStep-first projection model for Console views.

That means:

- replay-oriented message/timeline views should primarily be built from RunStep-backed data
- task and run summaries should be projections over SDK execution state
- low-level execution details should remain secondary and optional in the product surface

### Trace Decision Deferred

There is a real possibility that Trace overlaps enough with RunStep that Trace may eventually be reduced or removed from the SDK-side observability model.

However, this document does not lock in that conclusion. Instead, it records the following decision:

- this Console refactor must not depend on Trace being the long-term primary source of truth
- a separate technical investigation should determine whether RunStep can fully subsume current Trace usage

This keeps the refactor aligned with likely simplification without turning an open technical question into a premature architecture commitment.

## UI Exposure Principles

The default user-facing surface should emphasize the product model rather than the execution internals.

### Default View

Users should primarily see:

- sessions
- the current task in a session
- task status
- current or recent result summaries
- the ability to continue work or fork to a new session

### Secondary Debug View

Users should only see detailed execution mechanics when they explicitly ask for them, such as:

- run timelines
- step sequences
- tool activity
- trace-like diagnostic detail

This preserves power for SDK developers while keeping the primary workflow legible and lightweight.

## Service Boundary Direction

The refactor should move toward a smaller set of explicit Console-side service boundaries.

### Target Responsibilities

#### SessionService

Own session lifecycle concerns such as create, switch, resume, and lookup.

#### ConversationService

Accept user input within a session context, determine the relevant task semantics, and route execution through the scheduler.

#### TaskProjectionService

Build user-facing task views from SDK execution facts and session/task state.

#### RunProjectionService

Build run-oriented replay and diagnostic views without becoming a second execution state machine.

#### ChannelAdapter

Represent Console and Feishu as separate entry adapters that translate entrypoint-specific protocol and identity concerns into the shared application model.

### Boundary Constraints

- entry adapters must not own product semantics
- projection services must not become authoritative runtime owners
- scheduler remains the orchestration entrypoint
- Console should stay aligned to SDK contracts rather than re-encoding execution logic in parallel

## Implications for Existing Console Structure

The refactor should deliberately move away from a structure where channel-specific code paths, routing decisions, and execution semantics are spread across multiple partially overlapping services.

The desired effect is not merely moving files. It is making the following true:

- there is one clear path for session-based remote interaction
- Console and Feishu differ only at adapter/protocol edges
- Session / Task / Run are represented consistently from API layer to projection layer
- execution truth stays in SDK-layer data and orchestration components

## Testing Strategy

The refactor should be validated at three levels.

### 1. Domain and Service Semantics

Tests should verify:

- session switching behavior
- implicit task creation behavior
- one-session-one-task default behavior
- fork lineage behavior and independence
- routing through the scheduler as the default execution path

### 2. Projection Correctness

Tests should verify:

- task views are correctly projected from SDK execution facts
- run replay views remain consistent with RunStep-backed execution data
- detailed views remain optional and do not leak into the default workflow model

### 3. Channel Consistency

Tests should verify:

- Console and Feishu produce equivalent session/task semantics
- channel-specific behavior remains confined to adapter-level logic
- identity and session binding stay correct when switching sessions or continuing work

## Risks and Tradeoffs

### Risk: Over-modeling Task in the UI

Because Task is a first-class domain object, there is a temptation to expose it too aggressively in the product workflow.

Mitigation:

- keep task explicit in the model
- keep task mostly implicit in the default interaction flow
- expose stronger task controls only where they materially help debugging or workflow control

### Risk: Refactor Recreates a Parallel Runtime Model

If Console-side projection logic starts carrying authoritative execution state, the refactor will reproduce the very complexity it is trying to remove.

Mitigation:

- treat SDK execution records as the source of truth
- make all Console-side run/task views clearly derived
- keep execution orchestration centered in scheduler and SDK runtime

### Risk: Premature Trace Removal Decision

Folding Trace into RunStep may ultimately be correct, but locking that in before evidence could distort the refactor.

Mitigation:

- make the Console refactor compatible with a RunStep-first future
- isolate the Trace vs. RunStep decision into a dedicated investigation

## Recommended Implementation Sequence

1. codify the Session / Task / Run semantics in Console domain models and service contracts
2. unify Console and Feishu entry semantics around session-first, scheduler-mediated interaction
3. simplify routing so ConversationService owns the main input-to-scheduler path
4. introduce projection-oriented task/run view building over SDK facts
5. add fork support as an explicit cross-session workflow capability
6. tighten tests around session/task semantics and channel consistency
7. separately investigate whether Trace can be retired in favor of RunStep-only projections

## Final Design Decision

This refactor should be treated as a product-model alignment refactor, not a feature sprint.

The Console becomes a remote workspace built for SDK developers, centered on session-first interaction, implicit task creation, scheduler-mediated execution, fork-based branching into new sessions, and projection-only views over SDK execution facts.
