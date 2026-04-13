# Agiwo Repository Overview Generation Design

## Summary

The current public site for `docs.agiwo.o-ai.tech` provides an SEO homepage and a small curated docs set, but its architecture and repository explanations are partly hand-maintained and can drift from the codebase.

This design adds a code-tree-driven repository overview page to the public site. The overview will be generated manually from stable repository signals rather than written entirely by hand. The primary fact source will be the repository structure itself, with `README.md` and `AGENTS.md` used as supporting inputs. Before generation, both documents must be aligned with the current codebase so they can safely serve as narrative supplements instead of stale truth.

## Goals

- Add a public repository overview page that feels closer to a code-aware explainer than a static marketing page.
- Generate the overview from stable repository signals instead of maintaining it manually.
- Keep the first version architecture-level, focused on directory responsibilities and public runtime surfaces.
- Align `README.md` and `AGENTS.md` with the current codebase so they remain useful supporting sources.
- Keep the generation workflow explicit and reviewable by committing generated output.

## Non-Goals

- Full API reference generation from source code.
- LLM-driven summarization during build time.
- Deep class-by-class or function-by-function code explanation in the first phase.
- Automatic regeneration on every push.
- Replacing `README.md` or `AGENTS.md` as human-readable project documentation.

## Constraints And Assumptions

- Public docs are served through the existing `website/` Astro + Starlight project.
- The repository overview should be static content checked into git, not generated on-the-fly at request time.
- The first version should prefer stable and explainable rules over ambitious coverage.
- `AGENTS.md` is useful because it already captures directory responsibilities and architecture boundaries, but it is not authoritative if it diverges from code.
- `README.md` should continue to optimize for external discoverability and first-use clarity, not become an exhaustive architecture reference.

## Recommended Approach

Implement a manual generator that scans the repository tree and selected supporting documents, then writes a structured JSON file consumed by the public docs site.

Why this approach:

- It keeps generation deterministic and reviewable.
- It avoids putting repo-scanning complexity inside the Astro build.
- It allows manual review of generated output before publishing.
- It supports progressive refinement of rules without locking the site into hand-maintained pages.

Rejected alternatives:

- Manual-only public overview page: too easy to drift.
- LLM-generated overview: harder to keep stable and defensible.
- Build-time scanning directly inside Astro: mixes content generation with presentation and makes local debugging harder.

## Fact Sources

### Primary Sources

- Repository directory structure
- Package/module entry points
- Existing public documentation headings

### Supporting Sources

- `README.md`
- `AGENTS.md`

### Source Priority

1. Code tree and stable package structure
2. `AGENTS.md` responsibility and boundary statements
3. `README.md` high-level external framing
4. Existing docs headings and links as navigation/context helpers

## Scope Of The Generated Overview

The first version should cover architecture-level repository understanding only.

Included:

- High-level repository summary
- SDK top-level directories and responsibilities
- Console top-level directories and responsibilities
- Supporting directories and their roles
- Key public entry surfaces such as `agiwo.agent`, `agiwo.scheduler`, `agiwo.tool`, and `agiwo.llm`
- Architecture boundary notes sourced from stable rules
- Pointers to relevant source directories and docs pages

Excluded in phase one:

- Symbol-level inventories
- Class inheritance maps
- Per-file detailed summaries across the whole repo
- Automated code examples synthesized from source

## Public Site Output

Add a new public docs page:

- `/docs/repo-overview/`

This page should contain five sections:

1. What Agiwo Contains
2. Repository Layout
3. Core Runtime Surfaces
4. Architectural Boundaries
5. Source Pointers

The page should render from generated structured data rather than inline hand-written explanations.

## Generator Output Format

Generate a structured JSON artifact, for example:

- `website/src/generated/repo-overview.json`

The JSON should contain explicit fields for:

- repository summary
- section groups
- path responsibilities
- public runtime surfaces
- boundary notes
- related docs/source links

The format should be simple enough for both the generator and the rendering component to remain understandable.

## Repository Changes

### Document Calibration

Before generation logic is trusted, align these files with the codebase:

- `README.md`
- `AGENTS.md`

Calibration should focus on:

- correcting stale statements
- removing obsolete claims
- tightening wording to current boundaries
- preserving each document’s role instead of turning either into a file dump

### Generator Layer

Add a manual generation script, preferably:

- `scripts/generate_repo_overview.py`

Responsibilities:

- scan selected repository directories
- detect top-level package/module surfaces
- read supporting docs
- merge facts according to source priority
- emit structured JSON for the public site

### Public Site Rendering

Add:

- a generated data directory in `website/src/generated/`
- one or more rendering components for overview tables/sections
- a public docs page under the Starlight docs tree

The rendering layer should not rescan the repository. It should only consume the generated JSON.

## Generation Rules

### Directory Responsibilities

Responsibilities should be resolved using this order:

1. Explicit mapping rules from the generator for known top-level directories
2. `AGENTS.md` responsibility text when available and still consistent
3. Simple fallback phrasing derived from path names

### Public Runtime Surfaces

The generator should identify and present only public-facing or high-signal package entry surfaces.

Examples:

- `agiwo.agent`
- `agiwo.scheduler`
- `agiwo.tool`
- `agiwo.llm`
- selected `console/server` top-level service boundaries

Do not attempt full symbol extraction in phase one.

### Architecture Boundary Notes

Boundary notes should prefer stable rules, such as:

- scheduler depends on agent
- console should use facades instead of store internals
- public types should enter through the correct facade

These are better suited for public explanation than volatile implementation details.

## Manual Workflow

The intended workflow is:

1. Update code
2. If needed, align `README.md` and `AGENTS.md`
3. Run the generator script manually
4. Review the generated JSON diff
5. Build the public site and review the rendered page
6. Commit both source and generated output

This keeps the process explicit while still reducing hand-maintained page drift.

## Testing And Validation

Implementation should validate:

- generator runs successfully from repo root
- generated JSON is deterministic for the same repository state
- public site builds successfully using generated output
- the overview page renders expected sections and path tables
- generated content links only to paths/docs that actually exist

Manual validation should check:

- obvious stale claims are gone from `README.md` and `AGENTS.md`
- overview page matches current repository structure
- narrative remains architecture-level and readable to external visitors

## Risks

- If the generator depends too heavily on heuristics, the overview can become vague or misleading.
- If `README.md` and `AGENTS.md` are not calibrated first, supporting text may reintroduce stale claims.
- If the scope expands into symbol-level extraction too early, complexity rises quickly without proportional public value.
- If generated output becomes too verbose, the page will feel like a file dump instead of a useful explainer.

## Success Criteria

- `README.md` and `AGENTS.md` are aligned with the current codebase.
- A manual generator can produce a structured repository overview artifact.
- The public docs site exposes `/docs/repo-overview/`.
- The overview is primarily derived from repository structure rather than hand-maintained prose.
- The resulting page gives external readers a clearer and more current architecture view than the existing hand-written docs alone.
