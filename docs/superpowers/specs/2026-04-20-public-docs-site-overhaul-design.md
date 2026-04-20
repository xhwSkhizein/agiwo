# Public Docs Site Overhaul Design

**Date:** 2026-04-20

## Goal

Rebuild the public `website/` into a balanced product site plus complete documentation hub: preserve and strengthen marketing + SEO value on the homepage, while turning `/docs/...` into a coherent, comprehensive public documentation system for Agiwo.

## Problem Statement

The current public site is structurally too thin:

- the homepage is a lightweight marketing page with limited depth
- the docs section is sparse and does not yet form a complete learning path
- content ownership between repository-native docs and website docs is unclear
- navigation is too shallow to support serious external developer evaluation
- SEO currently depends too much on a small number of pages instead of a topic cluster

The result is that the site neither fully sells Agiwo nor fully teaches it.

## Product Positioning

The site should move away from generic framework language such as “streaming-first Python AI agent framework” as its main story.

The new primary narrative is:

> **Runtime harness for orchestrated, self-improving agents**

This framing better matches the current and emerging direction of the project:

- explicit runtime boundaries
- scheduler-driven orchestration
- trace + persistence + operator visibility
- skills, memory, and tool control
- context optimization and retrospective feedback loops
- ongoing work around self-improvement and runtime evolution

The phrase “runtime harness” is the central differentiator. The public site should consistently explain it as the union of:

- runtime control
- orchestration
- observability
- optimization loops

The phrase “self-improving” is preferred over “self-evolving” for the top-level promise because it is ambitious without implying unconstrained autonomous self-modification.

## Audience

The site should serve two audiences, with priority on external developers:

1. external developers evaluating or adopting Agiwo
2. repository contributors and maintainers who still need structured reference material

This means the site must support:

- fast first impression and positioning
- a short path to first success
- deep technical understanding
- architecture trust-building
- operator-facing Console and deployment guidance

## Content Strategy

The content model should use a split-source strategy:

- repository `docs/` remains the engineering-facing and repository-native documentation space
- `website/src/content/docs/` becomes the formal public documentation product

The public docs are not a raw mirror of `docs/`. They are curated, rewritten, and restructured for external consumption.

### Repository `docs/` should continue to own

- engineering-facing reference notes
- architecture notes intended for contributors
- maintenance workflows
- historical plans/specs
- repo-native boundary documents

### `website/src/content/docs/` should own

- polished getting-started flows
- stable concepts pages
- guided tutorials
- public-facing API/reference pages
- public architecture explanations
- operator-facing Console and deployment docs
- compare pages
- FAQ

## Site Model

The public site should be a two-layer system:

- a marketing homepage outside `/docs`
- a structured docs application under `/docs`

The homepage and docs must feel like one product, not two unrelated surfaces.

## Information Architecture

The docs app should be reorganized into the following top-level structure:

- `Docs`
  - landing page
  - installation
  - getting started
  - first agent
- `Guides`
  - custom tools
  - multi-agent
  - streaming
  - storage
  - skills
  - hooks
  - context optimization
- `Concepts`
  - agent
  - model
  - tool
  - scheduler
  - memory
  - runtime harness / runtime boundaries
- `Reference`
  - model API
  - tool API
  - scheduler API
  - console overview
  - console API
  - console docker / deployment
  - configuration and environment variables
- `Architecture`
  - architecture overview
  - memory system
  - repo overview
  - selected runtime design pages that are suitable for public consumption
- `Compare`
  - Agiwo vs LangGraph / OpenAI Agents / AutoGen
  - future framework comparisons
- `FAQ`

This structure should be explicit in Starlight navigation rather than relying on one broad auto-generated bucket.

## Homepage Design Role

The homepage should remain marketing-capable and SEO-capable, but become more systematic.

It should answer:

- what Agiwo is
- who it is for
- what makes it different
- how quickly someone can start
- where to go next

### Recommended homepage sections

1. Hero
   - positioning line built around “runtime harness”
   - concise value statement
   - strong CTAs to docs, compare, and GitHub
2. Why Agiwo
   - differentiation pillars
   - clear reasons Agiwo is not “just another agent framework”
3. Quickstart Preview
   - minimal working example
4. Choose Your Path
   - first agent
   - multi-agent / orchestration
   - console / operator path
5. Comparison Entry
   - visible link into framework comparison
6. Architecture Snapshot
   - runtime boundaries and system layers
7. FAQ
   - external evaluator questions
8. Final CTA
   - start docs
   - compare frameworks
   - view GitHub

## Narrative System

The site should use a unified vocabulary.

### Preferred top-level language

- runtime harness
- orchestrated agents
- self-improving agents
- control plane
- runtime boundaries
- traceable execution
- persistent orchestration

### Language to demote from headline position

- generic “AI agent framework”
- generic “streaming-first” framing without broader context

“Streaming-first” should still appear, but as one capability in the larger harness story rather than the main identity.

## SEO Strategy

SEO must be preserved and strengthened, not sacrificed during reorganization.

### Core principle

SEO should be distributed across a topic cluster, not concentrated in the homepage alone.

### SEO page roles

- homepage
  - brand + broad category queries
- getting started / installation / first agent
  - high-intent onboarding queries
- guides
  - “how to” queries
- concepts and architecture
  - conceptual and systems-design queries
- compare pages
  - framework comparison and alternative-evaluation queries
- Console deployment/API pages
  - operator and deployment queries
- FAQ
  - long-tail question queries

### SEO implementation requirements

- maintain stable URLs where practical
- avoid removing current public routes unless a redirect strategy exists
- ensure every public page has:
  - unique title
  - unique description
  - canonical URL
  - meaningful H1/H2 hierarchy
- avoid duplicative metadata across many docs pages
- ensure homepage and comparison pages remain strong internal-link hubs

## Marketing Requirements

Marketing strength must not regress.

The homepage should still:

- communicate differentiation within seconds
- support GitHub conversion
- provide immediate “start here” CTAs
- frame Agiwo as a serious systems product, not a toy SDK

The docs structure itself should also support marketing by building trust:

- architecture pages prove design quality
- comparison pages frame category position
- Console pages prove operator maturity
- API/reference pages prove implementation seriousness

## Implementation Architecture

The implementation should stay within the current Astro + Starlight stack.

### Keep

- Astro site shell
- Starlight docs integration
- custom homepage outside `/docs`

### Change

- Starlight sidebar and docs organization
- homepage content system and section composition
- public docs content inventory
- metadata discipline and internal linking

No CMS or remote content pipeline should be introduced in this phase.

## Migration Strategy

Use incremental migration rather than a destructive rewrite.

1. preserve current public URLs where feasible
2. rebuild homepage messaging and section system
3. reorganize docs navigation and landing pages
4. add missing high-value public docs pages
5. migrate or rewrite selected repository docs into polished website docs
6. validate SEO-sensitive routes and internal links

The migration should favor compatibility over novelty.

## Scope

### In scope

- homepage overhaul
- docs information architecture rebuild
- content stratification between `docs/` and `website/src/content/docs/`
- addition of core missing public docs pages
- stronger comparison, architecture, Console, and FAQ coverage
- stronger SEO metadata and site navigation

### Out of scope

- a full standalone brand site with many marketing microsites
- Console application changes
- wholesale migration of all repository docs into the website
- adding a CMS
- reworking the runtime product itself to match the new messaging

## Success Criteria

The overhaul succeeds if:

- the homepage clearly differentiates Agiwo using the runtime-harness narrative
- the docs form a complete external-developer path from onboarding to deep reference
- the docs navigation is intelligible at a glance
- marketing capability and SEO are preserved or improved
- content ownership between repository docs and website docs becomes clear
- the public site feels like a mature documentation product rather than a thin wrapper around a repo

## Risks

### Risk: positioning overpromises on self-improvement

Mitigation:

- use “self-improving” as the umbrella framing
- ground every claim in existing or clearly emerging runtime capabilities
- explain the harness concept concretely in docs and architecture pages

### Risk: docs scope balloons into a full content rewrite

Mitigation:

- prioritize a core page set first
- reuse and refine good existing content where possible
- separate repository docs cleanup from website public-docs completion

### Risk: SEO regressions from URL or metadata churn

Mitigation:

- preserve current routes where practical
- keep comparison and getting-started pages stable
- explicitly review metadata and internal-link hubs during implementation

## Recommended Execution Order

1. define the final docs IA and sidebar
2. redesign homepage around the new narrative
3. create the missing public docs skeleton
4. rewrite key pages for completeness and consistency
5. refine SEO metadata and cross-linking
6. validate build, navigation, and route integrity
