# Skill Discovery And Dynamic Activation Design

**Date:** 2026-04-16

## Goal

Stop skill discovery from inflating the agent system prompt while preserving the ability to use skills when they are actually needed.

## Scope

This design covers the first-stage skill discovery redesign for the SDK.

It includes:

1. shrinking the skill-related system prompt surface
2. keeping a small statically configured default skill set in prompt
3. extending the existing `skill` tool to support `search` and `activate`
4. adding runtime skill recommendation based on embedding TopK plus LLM judgment
5. keeping all filtering constrained by `allowed_skills`

It does not include:

- schema migration
- persistent vector indexes
- telemetry-driven default skill selection
- automatic runtime injection of hidden skill recommendations without a tool call
- changes to the core allowlist semantics

## Current Problem

Today `[build_system_prompt()](/home/hongv/workspace/agiwo/agiwo/agent/prompt.py)` asks `SkillManager` to render a full skills section into the system prompt. `[SkillPromptCatalog.render_section()](/home/hongv/workspace/agiwo/agiwo/skill/prompt_catalog.py)` emits one prompt entry for every discovered skill, including its name, description, and location.

This creates a linear prompt growth problem:

- every discovered skill increases prompt size
- large `skills_dirs` make prompt cost and latency scale with repository size rather than user need
- most requests do not require any skill at all
- the current design pays prompt cost before the model has decided whether a skill is useful

The existing `[SkillTool](/home/hongv/workspace/agiwo/agiwo/skill/skill_tool.py)` already loads full `SKILL.md` content on demand. The missing piece is dynamic discovery, not dynamic activation.

## Decision

Adopt a two-layer design:

1. prompt-time skill visibility becomes small and static
2. runtime skill discovery becomes explicit through the existing `skill` tool

The `skill` tool remains the only skill-facing tool, but its protocol expands to support:

- `search`: recommend a skill or explicitly recommend none
- `activate`: load the full `SKILL.md` content for a chosen skill

The model should not be pushed toward skills by default. "Do not use a skill" must remain a normal, successful result.

## Prompt Design

The system prompt should no longer list every discovered skill.

Instead, the skill section should contain:

- a short instruction that skills are optional and should be used only when clearly helpful
- a small statically configured `default_prompt_skills` subset
- brief instructions to use `skill.search` before `skill.activate` when the model is unsure

`default_prompt_skills` is SDK global configuration. It is validated as a list of explicit skill names during startup and must fail fast for unknown names.

Prompt rendering must apply the same allowlist boundary as runtime search:

- if `allowed_skills is None`, the prompt may show all configured default prompt skills
- if `allowed_skills` is a list, the prompt may only show `default_prompt_skills ∩ allowed_skills`
- if `allowed_skills == []`, skills are disabled and no skill section is rendered

This keeps prompt size stable while preserving a small high-signal set of commonly useful skills.

## Tool Protocol

The existing `skill` tool becomes a dual-mode tool.

### `skill.search`

Input schema:

```json
{
  "mode": "search",
  "query": "original user request"
}
```

Behavior:

- `query` is the raw user request text as seen by the model
- the tool does not activate any skill
- the tool returns a structured recommendation only

Output shape:

```json
{
  "decision": "recommend",
  "skill_name": "brainstorming",
  "reason": "The request is about exploring approaches before implementation."
}
```

or:

```json
{
  "decision": "no_recommendation",
  "reason": "No available skill is necessary or clearly matched for this request."
}
```

### `skill.activate`

Input schema:

```json
{
  "mode": "activate",
  "skill_name": "brainstorming"
}
```

Behavior:

- preserves the current activation semantics
- loads the full `SKILL.md` body for the requested skill
- fails if the skill does not exist or is not allowed

The tool name stays `skill` so the tool surface remains simple.

## Search Pipeline

`skill.search` should run the following steps in order.

### 1. Candidate Filtering

Start from the discovered skill metadata cache in `SkillManager`.

Apply allowlist semantics first:

- `allowed_skills is None`: all discovered skills are eligible
- `allowed_skills == []`: return `no_recommendation`
- explicit list: only those skills are eligible

This preserves the current security and boundary semantics. An agent must never discover or activate a skill outside its allowed set.

### 2. Embedding Recall

Build an in-memory searchable text for each skill from existing metadata:

- `name`
- `description`
- optional `metadata` entries when present

No migration or new persisted index is introduced in this phase.

At runtime:

- embed the user query
- embed the filtered skill corpus
- compute similarity
- take TopK candidates, default `K = 6`

The searchable corpus can be rebuilt in memory when skills are initialized or reloaded. This keeps the first version simple and consistent with the existing metadata cache design.

### 3. LLM Judgment

Pass only the following to the judge model:

- the raw user query
- the TopK candidate skill metadata

The judge must choose one of two valid outcomes:

1. recommend exactly one specific skill from the candidate set
2. recommend no skill

This judgment is recommendation-only. It does not activate the skill automatically.

If the judge returns an unknown skill or a skill outside the candidate set, the runtime must treat that as invalid output and downgrade to `no_recommendation`.

## Failure Semantics

`skill.search` is advisory and must not block the main task.

The first-stage behavior should be:

- if skills are disabled, the `skill` tool is not registered
- if the candidate set is empty, return `no_recommendation`
- if embedding is unavailable or fails, return `no_recommendation`
- if the judge LLM fails, times out, or returns invalid structured output, return `no_recommendation`

`skill.activate` keeps strict failure behavior:

- unknown skill -> failed tool result
- disallowed skill -> failed tool result
- unreadable or invalid `SKILL.md` -> failed tool result

This asymmetric design keeps discovery cheap and safe while preserving explicit failure for actual activation.

## Module Boundaries

The implementation should stay mostly inside `agiwo/skill/`.

### Existing Modules To Change

- `[agiwo/agent/prompt.py](/home/hongv/workspace/agiwo/agiwo/agent/prompt.py)`
  render only the reduced prompt-time skill section
- `[agiwo/skill/manager.py](/home/hongv/workspace/agiwo/agiwo/skill/manager.py)`
  remain the facade for discovery, prompt rendering, and search entrypoints
- `[agiwo/skill/skill_tool.py](/home/hongv/workspace/agiwo/agiwo/skill/skill_tool.py)`
  parse `mode`, call search or activate, and return `ToolResult`

### New Module

Add a focused runtime search module:

- `agiwo/skill/search.py`

Responsibilities:

- convert skill metadata into searchable text
- manage in-memory search corpus
- run embedding TopK recall
- run LLM judgment
- return structured recommendation results

`ToolManager` should remain unaware of search internals. Its only job is still to create `SkillTool` when skills are enabled.

## Configuration

Add SDK-level configuration in `[agiwo/config/settings.py](/home/hongv/workspace/agiwo/agiwo/config/settings.py)` for the first-stage rollout:

- `default_prompt_skills: list[str] = []`
- `skill_search_enabled: bool = True`
- `skill_search_top_k: int = 6`

The first version should not add a separate persisted config model or migration path.

Judge model configuration should stay simple:

- the first version reuses the existing runtime model capability instead of adding separate judge-model settings
- no dedicated persisted or console-level judge configuration is introduced in this phase

This keeps the first implementation focused on prompt reduction and dynamic discovery rather than provider plumbing.

## Testing Strategy

Add tests in three layers.

### 1. Prompt Construction Tests

Verify that:

- full discovered skills are no longer rendered into the system prompt
- only configured default prompt skills appear
- rendered prompt skills still respect `allowed_skills`
- disabling skills removes the prompt section entirely

### 2. Skill Search Tests

Verify that:

- `skill.search` only considers skills inside the allowed set
- search returns TopK candidates before judge evaluation
- judge can recommend exactly one candidate
- judge can return `no_recommendation`
- invalid judge output downgrades to `no_recommendation`
- embedding or judge failure downgrades to `no_recommendation`

### 3. Activation And Integration Tests

Verify that:

- `skill.activate` keeps current successful activation behavior
- disallowed activation still fails
- a normal non-skill task can proceed without activating any skill
- a clear skill-matching task can call `search` first and `activate` second

## Rollout Plan

The first rollout should be intentionally narrow:

1. stop injecting the full skill catalog into the system prompt
2. render only `default_prompt_skills`
3. add `skill.search`
4. keep activation explicit through `skill.activate`
5. keep all search state in memory

The implementation should not add:

- migration logic
- usage-frequency ranking
- hidden auto-routing
- persisted embedding caches

## Acceptance Criteria

The change is complete when:

1. system prompt size no longer scales with the total number of discovered skills
2. only statically configured default prompt skills are rendered, subject to `allowed_skills`
3. the `skill` tool supports both `search` and `activate`
4. `skill.search` can return either a concrete recommendation or `no_recommendation`
5. search can never reveal or recommend a skill outside the agent's allowed set
6. the first implementation works without schema migration or persisted search indexes
