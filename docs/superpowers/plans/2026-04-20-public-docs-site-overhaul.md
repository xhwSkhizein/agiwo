# Public Docs Site Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the public `website/` into a stronger marketing homepage plus a complete, externally consumable documentation hub without regressing SEO or existing high-value public routes.

**Architecture:** Keep Astro + Starlight as the delivery stack, but replace the current shallow docs structure with an explicit information architecture and a richer homepage. Public docs live in `website/src/content/docs/docs/`, while repository `docs/` remains engineering-facing source material and reference context.

**Tech Stack:** Astro, Starlight, MDX, CSS, structured metadata

---

### Task 1: Rebuild Site Navigation And Metadata

**Files:**
- Modify: `website/astro.config.mjs`
- Modify: `website/src/layouts/MarketingLayout.astro`
- Modify: `website/src/pages/404.astro`
- Test: `website/package.json`

- [ ] **Step 1: Replace the single auto-generated sidebar with an explicit public IA**

Use a sidebar structure like:

```js
sidebar: [
  {
    label: "Start Here",
    items: [
      { label: "Documentation", link: "/docs/" },
      { label: "Installation", link: "/docs/installation/" },
      { label: "Getting Started", link: "/docs/getting-started/" },
      { label: "First Agent", link: "/docs/first-agent/" },
    ],
  },
  {
    label: "Guides",
    items: [
      { label: "Custom Tools", link: "/docs/guides/custom-tools/" },
      { label: "Multi-Agent", link: "/docs/guides/multi-agent/" },
      { label: "Streaming", link: "/docs/guides/streaming/" },
      { label: "Storage", link: "/docs/guides/storage/" },
      { label: "Skills", link: "/docs/guides/skills/" },
      { label: "Hooks", link: "/docs/guides/hooks/" },
      { label: "Context Optimization", link: "/docs/guides/context-optimization/" },
    ],
  },
]
```

- [ ] **Step 2: Strengthen global site metadata around the new narrative**

Update the Starlight config description to center the harness framing:

```js
description:
  "Agiwo is a runtime harness for orchestrated, self-improving agents in Python, with tools, scheduler orchestration, persistence, tracing, and a control plane.",
```

Also add or verify `title`, GitHub social link, and any stable canonical-bearing site values already used by Astro.

- [ ] **Step 3: Make the marketing layout reusable for stronger per-page SEO**

Extend the layout props to support explicit OG image and canonical override:

```astro
interface Props {
  title: string;
  description: string;
  canonicalPath?: string;
  socialImagePath?: string;
}

const {
  title,
  description,
  canonicalPath = "/",
  socialImagePath = "/social-card.svg",
} = Astro.props;
```

- [ ] **Step 4: Update the 404 page CTA to point into the new docs structure**

Ensure the 404 action points to `/docs/` and includes a second CTA for the homepage:

```astro
<a class="button button--primary" href="/docs/">Go to docs</a>
<a class="button button--secondary" href="/">Back to homepage</a>
```

- [ ] **Step 5: Verify config and route integrity**

Run: `cd website && npm run check`
Expected: Astro/Starlight config validates with no schema or route errors.

### Task 2: Redesign The Homepage As A Product Entry Surface

**Files:**
- Modify: `website/src/pages/index.astro`
- Modify: `website/src/components/Hero.astro`
- Modify: `website/src/components/FeatureGrid.astro`
- Modify: `website/src/components/CodePreview.astro`
- Modify: `website/src/components/ArchitectureSnapshot.astro`
- Modify: `website/src/components/FAQList.astro`
- Create: `website/src/components/PathChooser.astro`
- Create: `website/src/components/WhyAgiwo.astro`
- Create: `website/src/components/FinalCTA.astro`
- Modify: `website/src/styles/site.css`

- [ ] **Step 1: Replace the homepage narrative with the harness positioning**

Update the page-level metadata in `website/src/pages/index.astro`:

```astro
const title = "Agiwo: Runtime Harness for Orchestrated, Self-Improving Agents";
const description =
  "Build orchestrated, tool-using, traceable AI systems in Python with Agiwo. Explore the runtime harness, scheduler orchestration, persistence, Console, and public docs.";
```

Use section order:

```astro
<Hero />
<WhyAgiwo />
<CodePreview />
<PathChooser />
<ArchitectureSnapshot />
<FeatureGrid />
<section class="comparison-entry">...</section>
<FAQList />
<FinalCTA />
```

- [ ] **Step 2: Rewrite the hero for differentiation instead of generic framework language**

Use copy in this direction:

```astro
<p class="eyebrow">Runtime harness for AI systems</p>
<h1>Build orchestrated, self-improving agents with explicit runtime control.</h1>
<p class="hero__copy">
  Agiwo combines agent execution, tool contracts, scheduler orchestration,
  persistence, tracing, and a control plane into one Python runtime harness.
</p>
```

- [ ] **Step 3: Add a “Why Agiwo” section that frames the category position**

Create `website/src/components/WhyAgiwo.astro` with 4 cards:

```astro
---
const pillars = [
  {
    title: "Runtime harness, not just prompts",
    body: "Agiwo centers execution control, orchestration, observability, and optimization loops rather than treating the runtime as an implementation detail.",
  },
  {
    title: "Explicit orchestration",
    body: "Agents, tools, scheduler, and Console stay separate enough to inspect, debug, and evolve independently.",
  },
]
---
```

- [ ] **Step 4: Add a “Choose Your Path” section to improve conversion into docs**

Create `website/src/components/PathChooser.astro` with 3 entry cards:

```astro
---
const paths = [
  {
    title: "Start your first agent",
    body: "Install Agiwo, configure a model, and run a minimal agent.",
    href: "/docs/getting-started/",
  },
  {
    title: "Build orchestration",
    body: "Learn scheduler patterns, multi-agent composition, and runtime control.",
    href: "/docs/guides/multi-agent/",
  },
  {
    title: "Operate with Console",
    body: "Understand sessions, scheduler state, traces, and Docker deployment.",
    href: "/docs/reference/console/overview/",
  },
];
---
```

- [ ] **Step 5: Update the code preview and FAQ to match current product language**

Change the code preview model to current examples:

```python
model=OpenAIModel(name="gpt-5.4")
```

And update FAQ prompts to include:

- what is a runtime harness
- when to choose Agiwo over graph-first or provider-first frameworks
- whether Agiwo includes a control plane

- [ ] **Step 6: Strengthen homepage styling without breaking the current look**

Add section-level styling for the new components:

```css
.why-agiwo,
.path-chooser,
.final-cta {
  margin-top: 36px;
}

.path-grid,
.pillar-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 16px;
}
```

- [ ] **Step 7: Verify homepage rendering**

Run: `cd website && npm run build`
Expected: homepage builds with the new sections and no missing component imports.

### Task 3: Rebuild The Docs Landing And Onboarding Path

**Files:**
- Modify: `website/src/content/docs/docs/index.mdx`
- Modify: `website/src/content/docs/docs/getting-started.mdx`
- Create: `website/src/content/docs/docs/installation.mdx`
- Create: `website/src/content/docs/docs/first-agent.mdx`

- [ ] **Step 1: Rewrite the docs landing page as a true navigation hub**

Replace the splash copy with the new narrative and explicit pathing:

```mdx
---
title: Agiwo Docs
description: Learn the Agiwo runtime harness for orchestrated, self-improving agents.
template: splash
hero:
  title: Agiwo documentation
  tagline: Build Python AI systems with explicit runtime control, orchestration, tracing, and operator visibility.
  actions:
    - text: Start with installation
      link: /docs/installation/
      icon: right-arrow
    - text: Read the comparison
      link: /docs/compare/agiwo-vs-langgraph-openai-agents-autogen/
      variant: minimal
---
```

- [ ] **Step 2: Split installation out of getting-started**

Create `website/src/content/docs/docs/installation.mdx` with:

```mdx
---
title: Installation
description: Install Agiwo and Agiwo Console from source or pip.
---

# Installation

## SDK

```bash
pip install agiwo
```

## Console

```bash
pip install agiwo-console
```
```

- [ ] **Step 3: Create a focused first-agent page**

Create `website/src/content/docs/docs/first-agent.mdx`:

```mdx
---
title: First Agent
description: Run your first Agiwo agent in Python.
---

# First Agent

```python
import asyncio

from agiwo.agent import Agent, AgentConfig
from agiwo.llm import OpenAIModel

async def main() -> None:
    agent = Agent(
        AgentConfig(
            name="assistant",
            system_prompt="Answer concisely.",
        ),
        model=OpenAIModel(name="gpt-5.4"),
    )

    result = await agent.run("What is 2 + 2?")
    print(result.response)
    await agent.close()

asyncio.run(main())
```
```

- [ ] **Step 4: Rewrite getting-started as the path overview instead of a monolith**

Refocus `website/src/content/docs/docs/getting-started.mdx` so it links to:

- installation
- first agent
- tools
- scheduler
- Console

Add a section like:

```mdx
## Choose Your Next Step

- [Install Agiwo](/docs/installation/)
- [Run your first agent](/docs/first-agent/)
- [Build custom tools](/docs/guides/custom-tools/)
- [Orchestrate multiple agents](/docs/guides/multi-agent/)
- [Understand the architecture](/docs/architecture/overview/)
```

- [ ] **Step 5: Verify onboarding links**

Run: `cd website && npm run build`
Expected: no broken links across `/docs/`, `/docs/installation/`, `/docs/getting-started/`, and `/docs/first-agent/`.

### Task 4: Expand Guides And Concepts Into A Coherent Public Learning Path

**Files:**
- Modify: `website/src/content/docs/docs/guides/custom-tools.mdx`
- Modify: `website/src/content/docs/docs/guides/multi-agent.mdx`
- Create: `website/src/content/docs/docs/guides/streaming.mdx`
- Create: `website/src/content/docs/docs/guides/skills.mdx`
- Create: `website/src/content/docs/docs/guides/hooks.mdx`
- Create: `website/src/content/docs/docs/guides/context-optimization.mdx`
- Create: `website/src/content/docs/docs/concepts/agent.mdx`
- Create: `website/src/content/docs/docs/concepts/model.mdx`
- Create: `website/src/content/docs/docs/concepts/tool.mdx`
- Create: `website/src/content/docs/docs/concepts/scheduler.mdx`
- Create: `website/src/content/docs/docs/concepts/memory.mdx`
- Create: `website/src/content/docs/docs/concepts/runtime-harness.mdx`

- [ ] **Step 1: Bring guides coverage in line with the public IA**

Create guide pages with concise public-facing structure:

```mdx
---
title: Streaming
description: Stream agent output from run start through final completion.
---

# Streaming

## `run_stream()`

```python
async for event in agent.run_stream("Explain recursion in one sentence."):
    if event.type == "step_delta" and event.delta.content:
        print(event.delta.content, end="", flush=True)
```
```

- [ ] **Step 2: Add concepts pages that explain the runtime model**

Create `concepts/runtime-harness.mdx` with the key framing:

```mdx
---
title: Runtime Harness
description: Understand Agiwo's runtime harness model for orchestrated, self-improving agents.
---

# Runtime Harness

Agiwo is not just a prompt wrapper or graph builder. It is a runtime harness that combines:

- agent execution
- tool contracts
- scheduler orchestration
- persistence and traces
- control-plane projection
- optimization loops such as context rollback and tool-result retrospect
```

- [ ] **Step 3: Make multi-agent and custom-tools pages match current code reality**

Update examples to use:

```python
OpenAIModel(name="gpt-5.4")
```

and current scheduler/agent terminology such as:

- `Agent.as_tool()`
- `Scheduler.submit(...)`
- `Scheduler.wait_for(...)`

- [ ] **Step 4: Cross-link guides and concepts**

At the bottom of each new guide/concept page, add explicit “Next” links:

```mdx
## Next

- [Understand the scheduler](/docs/concepts/scheduler/)
- [Build multi-agent flows](/docs/guides/multi-agent/)
```

- [ ] **Step 5: Verify concept/guide navigation**

Run: `cd website && npm run build`
Expected: all guide and concept routes resolve and appear in the sidebar.

### Task 5: Build Out Public Reference And Console Documentation

**Files:**
- Create: `website/src/content/docs/docs/reference/api/model.mdx`
- Create: `website/src/content/docs/docs/reference/api/tool.mdx`
- Create: `website/src/content/docs/docs/reference/api/scheduler.mdx`
- Create: `website/src/content/docs/docs/reference/console/overview.mdx`
- Create: `website/src/content/docs/docs/reference/console/api.mdx`
- Create: `website/src/content/docs/docs/reference/console/docker.mdx`
- Create: `website/src/content/docs/docs/reference/configuration.mdx`

- [ ] **Step 1: Add public API reference pages derived from the now-correct repo docs**

Example `reference/api/model.mdx` skeleton:

```mdx
---
title: Model API
description: Reference for Agiwo's model abstractions, providers, and factory helpers.
---

# Model API

## Public exports

- `LLMConfig`
- `Model`
- `StreamChunk`
- `ModelSpec`
- `create_model(...)`
- `create_model_from_dict(...)`
```

- [ ] **Step 2: Add Console public reference pages with session-first semantics**

Example `reference/console/overview.mdx`:

```mdx
---
title: Console Overview
description: Understand Agiwo Console as a session-first control plane.
---

# Console Overview

The Console is Agiwo's operator-facing control plane. It combines:

- agent registry
- session routing
- scheduler state inspection
- trace visibility
- deployment and channel integrations
```

- [ ] **Step 3: Add a public configuration page**

Create `reference/configuration.mdx` with sections for:

- provider credentials
- `AGIWO_*`
- `AGIWO_CONSOLE_*`
- compatible-provider `base_url` / `api_key_env_name`

Use a table like:

```mdx
| Variable | Scope | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | provider | OpenAI credentials |
| `AGIWO_SKILLS_DIRS` | SDK | skill discovery roots |
| `AGIWO_CONSOLE_PORT` | Console | bind port |
```

- [ ] **Step 4: Add public reference links back into guides and concepts**

Link examples:

```mdx
## See also

- [Model API](/docs/reference/api/model/)
- [Console API](/docs/reference/console/api/)
```

- [ ] **Step 5: Verify reference coverage**

Run: `cd website && npm run build`
Expected: all new reference routes build and no sidebar link is orphaned.

### Task 6: Refresh Architecture, Comparison, And FAQ For Trust And SEO

**Files:**
- Modify: `website/src/content/docs/docs/architecture/overview.mdx`
- Create: `website/src/content/docs/docs/architecture/memory.mdx`
- Modify: `website/src/content/docs/docs/repo-overview.mdx`
- Modify: `website/src/content/docs/docs/compare/agiwo-vs-langgraph-openai-agents-autogen.mdx`
- Modify: `website/src/content/docs/docs/faq.mdx`

- [ ] **Step 1: Rewrite architecture overview around runtime boundaries and session-first Console routing**

Use text like:

```mdx
## Runtime Path

```text
User / Console / Channel -> Session runtime services -> Scheduler -> Agent -> Tools / Model / Storage
```
```

And explicitly explain:

- runtime harness boundary
- scheduler ownership
- Console as projection layer

- [ ] **Step 2: Add a public memory architecture page**

Create `architecture/memory.mdx`:

```mdx
---
title: Memory System
description: Understand Agiwo's hybrid MEMORY retrieval path.
---

# Memory System

Agiwo uses hybrid retrieval over `MEMORY/` files, combining:

- file syncing
- chunking
- embeddings when available
- BM25 fallback
- result merging
```

- [ ] **Step 3: Strengthen the comparison page around category differentiation**

Rewrite comparison framing to emphasize:

- runtime harness vs graph-first abstraction
- explicit orchestration vs provider-native flows
- operator visibility and persistence
- control plane separation

Add a summary table:

```mdx
| Framework | Primary center of gravity | Best fit |
| --- | --- | --- |
| Agiwo | Runtime harness + control plane | Teams that want explicit runtime control and operator visibility |
```

- [ ] **Step 4: Rewrite FAQ around evaluation questions**

Ensure `faq.mdx` covers:

- what “runtime harness” means
- whether Agiwo is production-ready
- when to choose Agiwo instead of LangGraph or OpenAI Agents SDK
- whether Console is required

- [ ] **Step 5: Verify architecture and comparison pages as link hubs**

Run: `cd website && npm run build`
Expected: architecture, compare, and FAQ pages build and interlink cleanly.

### Task 7: Final QA For SEO, Navigation, And Build Stability

**Files:**
- Modify as needed: `website/src/pages/index.astro`
- Modify as needed: `website/astro.config.mjs`
- Verify: `website/src/content/docs/docs/**/*.mdx`

- [ ] **Step 1: Audit titles and descriptions across all public pages**

Check for duplicate or weak descriptions with:

```bash
rg -n '^title:|^description:' website/src/content/docs/docs website/src/pages/index.astro
```

Expected: each public page has a distinct, useful title/description pair.

- [ ] **Step 2: Validate route and navigation integrity**

Run:

```bash
cd website && npm run check
```

Expected: no content-schema or route errors.

- [ ] **Step 3: Build the full static site**

Run:

```bash
cd website && npm run build
```

Expected: the public site builds successfully with the new homepage and docs structure.

- [ ] **Step 4: Smoke-review the generated outputs**

Check key generated routes exist:

```bash
test -f website/dist/index.html
test -f website/dist/docs/index.html
test -f website/dist/docs/getting-started/index.html
test -f website/dist/docs/reference/console/docker/index.html
```

Expected: all commands succeed with no output.

- [ ] **Step 5: Commit**

```bash
git add website docs/superpowers/specs/2026-04-20-public-docs-site-overhaul-design.md docs/superpowers/plans/2026-04-20-public-docs-site-overhaul.md
git commit -m "docs: overhaul public docs site structure and content"
```
