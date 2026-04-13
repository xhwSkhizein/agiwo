# Agiwo Public Docs SEO Site Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a public static site for `docs.agiwo.o-ai.tech` with a strong SEO homepage, curated product docs, comparison content, and GitHub Pages deployment.

**Architecture:** Add a standalone `website/` Astro + Starlight project that is fully separate from the internal `console/web` app. The site will render a custom homepage at `/`, documentation content under `/docs/`, and comparison/FAQ content as static pages with per-page metadata, sitemap generation, robots support, and a GitHub Pages deployment workflow.

**Tech Stack:** Astro, Starlight, TypeScript, Markdown/MDX, GitHub Pages, GitHub Actions

---

## File Structure

### New files and directories

- Create: `website/package.json`
- Create: `website/tsconfig.json`
- Create: `website/astro.config.mjs`
- Create: `website/public/favicon.svg`
- Create: `website/public/social-card.svg`
- Create: `website/src/content.config.ts`
- Create: `website/src/content/docs/index.mdx`
- Create: `website/src/content/docs/getting-started.mdx`
- Create: `website/src/content/docs/architecture/overview.mdx`
- Create: `website/src/content/docs/guides/multi-agent.mdx`
- Create: `website/src/content/docs/guides/custom-tools.mdx`
- Create: `website/src/content/docs/guides/storage.mdx`
- Create: `website/src/content/docs/faq.mdx`
- Create: `website/src/content/docs/compare/agiwo-vs-langgraph-openai-agents-autogen.mdx`
- Create: `website/src/components/Hero.astro`
- Create: `website/src/components/FeatureGrid.astro`
- Create: `website/src/components/CodePreview.astro`
- Create: `website/src/components/ArchitectureSnapshot.astro`
- Create: `website/src/components/FAQList.astro`
- Create: `website/src/layouts/MarketingLayout.astro`
- Create: `website/src/pages/index.astro`
- Create: `website/src/pages/robots.txt.ts`
- Create: `website/src/styles/site.css`
- Create: `.github/workflows/public-docs.yml`
- Create: `docs/public-site-deploy.md`

### Existing files to modify

- Modify: `README.md`
- Modify: `docs/README.md`

### Tests and validation targets

- Validate: `npm --prefix website run build`
- Validate: `npm --prefix website run check`
- Validate: generated `website/dist/sitemap-index.xml` or `website/dist/sitemap.xml`
- Validate: generated `website/dist/robots.txt`

### Notes on boundaries

- Do not modify `console/web`
- Do not route public traffic through the internal Console
- Do not add Cloudflare automation in this phase

## Task 1: Scaffold the public docs site

**Files:**
- Create: `website/package.json`
- Create: `website/tsconfig.json`
- Create: `website/astro.config.mjs`
- Create: `website/src/content.config.ts`
- Create: `website/public/favicon.svg`
- Create: `website/public/social-card.svg`

- [ ] **Step 1: Create the site directory and initialize Astro + Starlight dependencies**

Run:

```bash
mkdir -p website/src website/public website/src/content/docs website/src/components website/src/layouts website/src/pages website/src/styles
cd website
npm init -y
npm install astro @astrojs/starlight @astrojs/sitemap typescript
```

Expected:

```text
added ... packages
found 0 vulnerabilities
```

- [ ] **Step 2: Replace `website/package.json` with the minimal site scripts**

Write:

```json
{
  "name": "agiwo-public-site",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "astro dev",
    "build": "astro build",
    "preview": "astro preview",
    "check": "astro check"
  },
  "dependencies": {
    "@astrojs/sitemap": "^3.2.1",
    "@astrojs/starlight": "^0.30.6",
    "astro": "^5.7.0"
  },
  "devDependencies": {
    "typescript": "^5.8.3"
  }
}
```

- [ ] **Step 3: Add TypeScript config for the site**

Write `website/tsconfig.json`:

```json
{
  "extends": "astro/tsconfigs/strict",
  "include": [".astro/types.d.ts", "**/*"],
  "exclude": ["dist"]
}
```

- [ ] **Step 4: Add the Astro config with site URL, Starlight, and sitemap support**

Write `website/astro.config.mjs`:

```js
import { defineConfig } from "astro/config";
import sitemap from "@astrojs/sitemap";
import starlight from "@astrojs/starlight";

export default defineConfig({
  site: "https://docs.agiwo.o-ai.tech",
  integrations: [
    sitemap(),
    starlight({
      title: "Agiwo Docs",
      description:
        "Open-source Python AI agent framework and control plane docs for streaming, tool use, orchestration, tracing, and persistence.",
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/xhwSkhizein/agiwo",
        },
      ],
      customCss: ["./src/styles/site.css"],
      sidebar: [
        {
          label: "Overview",
          items: [
            { label: "Docs Home", slug: "index" },
            { label: "Getting Started", slug: "getting-started" },
            { label: "FAQ", slug: "faq" },
          ],
        },
        {
          label: "Guides",
          items: [
            { label: "Architecture Overview", slug: "architecture/overview" },
            { label: "Multi-Agent", slug: "guides/multi-agent" },
            { label: "Custom Tools", slug: "guides/custom-tools" },
            { label: "Storage", slug: "guides/storage" },
            {
              label: "Agiwo vs Alternatives",
              slug: "compare/agiwo-vs-langgraph-openai-agents-autogen",
            },
          ],
        },
      ],
    }),
  ],
});
```

- [ ] **Step 5: Register the docs content collection**

Write `website/src/content.config.ts`:

```ts
import { defineCollection } from "astro:content";
import { docsLoader, docsSchema } from "@astrojs/starlight/loaders";
import { z } from "astro:content";

export const collections = {
  docs: defineCollection({
    loader: docsLoader(),
    schema: docsSchema({
      extend: z.object({
        seoTitle: z.string().optional(),
        seoDescription: z.string().optional(),
      }),
    }),
  }),
};
```

- [ ] **Step 6: Add the initial favicon and social card assets**

Write `website/public/favicon.svg`:

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" role="img" aria-label="Agiwo">
  <rect width="64" height="64" rx="14" fill="#0f172a" />
  <path d="M18 45L32 16l14 29h-8l-2.8-6H28.8L26 45h-8zm14.1-13h5.8L35 25.6 32.1 32z" fill="#f8fafc" />
</svg>
```

Write `website/public/social-card.svg`:

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 630">
  <rect width="1200" height="630" fill="#020617" />
  <rect x="48" y="48" width="1104" height="534" rx="24" fill="#0f172a" stroke="#334155" />
  <text x="90" y="220" fill="#f8fafc" font-size="72" font-family="ui-sans-serif, system-ui, sans-serif">Agiwo</text>
  <text x="90" y="310" fill="#cbd5e1" font-size="38" font-family="ui-sans-serif, system-ui, sans-serif">
    Open-source Python AI agent framework and control plane
  </text>
  <text x="90" y="390" fill="#94a3b8" font-size="28" font-family="ui-sans-serif, system-ui, sans-serif">
    Streaming, tool use, orchestration, tracing, persistence
  </text>
</svg>
```

- [ ] **Step 7: Run type and build validation for the empty shell**

Run:

```bash
npm --prefix website run check
npm --prefix website run build
```

Expected:

```text
Result (0 errors)
dist/ created
```

- [ ] **Step 8: Commit the scaffold**

Run:

```bash
git add website/package.json website/tsconfig.json website/astro.config.mjs website/src/content.config.ts website/public/favicon.svg website/public/social-card.svg
git commit -m "feat: scaffold public docs site"
```

## Task 2: Build the homepage shell and reusable marketing components

**Files:**
- Create: `website/src/layouts/MarketingLayout.astro`
- Create: `website/src/components/Hero.astro`
- Create: `website/src/components/FeatureGrid.astro`
- Create: `website/src/components/CodePreview.astro`
- Create: `website/src/components/ArchitectureSnapshot.astro`
- Create: `website/src/components/FAQList.astro`
- Create: `website/src/pages/index.astro`
- Create: `website/src/styles/site.css`

- [ ] **Step 1: Create the marketing layout with site-wide SEO metadata**

Write `website/src/layouts/MarketingLayout.astro`:

```astro
---
interface Props {
  title: string;
  description: string;
  canonicalPath?: string;
}

const {
  title,
  description,
  canonicalPath = "/",
} = Astro.props;

const canonicalUrl = new URL(canonicalPath, Astro.site);
const socialImage = new URL("/social-card.svg", Astro.site);
---

<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
    <meta name="description" content={description} />
    <link rel="canonical" href={canonicalUrl.toString()} />
    <meta property="og:type" content="website" />
    <meta property="og:title" content={title} />
    <meta property="og:description" content={description} />
    <meta property="og:url" content={canonicalUrl.toString()} />
    <meta property="og:image" content={socialImage.toString()} />
    <meta name="twitter:card" content="summary_large_image" />
    <meta name="twitter:title" content={title} />
    <meta name="twitter:description" content={description} />
    <meta name="twitter:image" content={socialImage.toString()} />
  </head>
  <body class="site-shell">
    <slot />
  </body>
</html>
```

- [ ] **Step 2: Create the hero component with SEO-focused messaging**

Write `website/src/components/Hero.astro`:

```astro
---
const links = {
  docs: "/docs/getting-started/",
  github: "https://github.com/xhwSkhizein/agiwo",
};
---

<section class="hero">
  <div class="hero__content">
    <p class="eyebrow">Open-source Python AI agent framework</p>
    <h1>Build streaming, tool-using, orchestrated LLM agents with Agiwo.</h1>
    <p class="hero__copy">
      Agiwo is a streaming-first Python AI agent SDK and control plane for teams
      building production-grade agents with tools, scheduling, tracing, and persistence.
    </p>
    <div class="hero__actions">
      <a class="button button--primary" href={links.docs}>Get Started</a>
      <a class="button button--secondary" href={links.github}>View on GitHub</a>
    </div>
  </div>
</section>
```

- [ ] **Step 3: Add the feature grid component**

Write `website/src/components/FeatureGrid.astro`:

```astro
---
const features = [
  {
    title: "Streaming-first runtime",
    body: "Run and stream through the same execution pipeline instead of separate code paths.",
  },
  {
    title: "Tool calling",
    body: "Combine builtin tools, custom tools, and agent-as-tool composition behind a stable interface.",
  },
  {
    title: "Scheduler orchestration",
    body: "Coordinate long-running roots, child agents, waiting, wakeups, and steering in one runtime.",
  },
  {
    title: "Tracing and persistence",
    body: "Persist runs and steps, inspect traces, and reason about execution after the fact.",
  },
];
---

<section class="feature-grid">
  {features.map((feature) => (
    <article class="feature-card">
      <h2>{feature.title}</h2>
      <p>{feature.body}</p>
    </article>
  ))}
</section>
```

- [ ] **Step 4: Add a code preview component for the minimal Python example**

Write `website/src/components/CodePreview.astro`:

```astro
---
const code = `import asyncio

from agiwo.agent import Agent, AgentConfig
from agiwo.llm import OpenAIModel


async def main() -> None:
    agent = Agent(
        AgentConfig(
            name="assistant",
            description="A helpful assistant",
            system_prompt="You are a concise assistant.",
        ),
        model=OpenAIModel(id="gpt-4o-mini"),
    )

    result = await agent.run("What is 2 + 2?")
    print(result.response)

    await agent.close()


asyncio.run(main())`;
---

<section class="code-preview">
  <div class="section-heading">
    <p class="eyebrow">Quick example</p>
    <h2>Start with a minimal Python agent.</h2>
  </div>
  <pre><code>{code}</code></pre>
</section>
```

- [ ] **Step 5: Add the architecture summary and FAQ components**

Write `website/src/components/ArchitectureSnapshot.astro`:

```astro
<section class="architecture-snapshot">
  <div class="section-heading">
    <p class="eyebrow">Architecture</p>
    <h2>Separate agent runtime, tools, scheduler, and control plane.</h2>
  </div>
  <div class="architecture-grid">
    <article><h3>Agent</h3><p>Owns execution, prompts, hooks, and run state.</p></article>
    <article><h3>Tool</h3><p>Defines stable contracts for builtin, custom, and nested-agent tools.</p></article>
    <article><h3>Scheduler</h3><p>Coordinates root and child execution, waiting, routing, and lifecycle.</p></article>
    <article><h3>Console</h3><p>Projects runtime state and traces without becoming the execution truth.</p></article>
  </div>
</section>
```

Write `website/src/components/FAQList.astro`:

```astro
---
const items = [
  {
    question: "What is Agiwo?",
    answer:
      "Agiwo is an open-source Python AI agent framework and control plane focused on streaming execution, tool use, orchestration, tracing, and persistence.",
  },
  {
    question: "Who is Agiwo for?",
    answer:
      "It is for developers building production-grade LLM agents and multi-agent systems in Python.",
  },
  {
    question: "Does Agiwo support multi-agent orchestration?",
    answer:
      "Yes. Agiwo supports agent-as-tool composition and scheduler-driven orchestration for longer-lived workflows.",
  },
];
---

<section class="faq-list">
  <div class="section-heading">
    <p class="eyebrow">FAQ</p>
    <h2>Common questions from first-time evaluators.</h2>
  </div>
  {items.map((item) => (
    <details class="faq-item">
      <summary>{item.question}</summary>
      <p>{item.answer}</p>
    </details>
  ))}
</section>
```

- [ ] **Step 6: Create the homepage and include structured data**

Write `website/src/pages/index.astro`:

```astro
---
import ArchitectureSnapshot from "../components/ArchitectureSnapshot.astro";
import CodePreview from "../components/CodePreview.astro";
import FAQList from "../components/FAQList.astro";
import FeatureGrid from "../components/FeatureGrid.astro";
import Hero from "../components/Hero.astro";
import MarketingLayout from "../layouts/MarketingLayout.astro";

const title = "Agiwo: Open-source Python AI Agent Framework and Control Plane";
const description =
  "Build streaming, tool-using, orchestrated LLM agents in Python with Agiwo. Explore docs for scheduling, tracing, persistence, and multi-agent execution.";
const structuredData = {
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  name: "Agiwo",
  applicationCategory: "DeveloperApplication",
  operatingSystem: "Cross-platform",
  url: "https://docs.agiwo.o-ai.tech/",
  description,
  softwareHelp: "https://docs.agiwo.o-ai.tech/docs/",
  codeRepository: "https://github.com/xhwSkhizein/agiwo",
};
---

<MarketingLayout title={title} description={description}>
  <main class="home-page">
    <Hero />
    <FeatureGrid />
    <CodePreview />
    <ArchitectureSnapshot />
    <section class="comparison-entry">
      <div class="section-heading">
        <p class="eyebrow">Comparison</p>
        <h2>Compare Agiwo with other agent frameworks.</h2>
        <p>
          Evaluate runtime boundaries, orchestration model, and observability tradeoffs
          before choosing a Python AI agent stack.
        </p>
        <a class="button button--secondary" href="/docs/compare/agiwo-vs-langgraph-openai-agents-autogen/">
          Read the comparison
        </a>
      </div>
    </section>
    <FAQList />
  </main>
  <script type="application/ld+json" set:html={JSON.stringify(structuredData)} />
</MarketingLayout>
```

- [ ] **Step 7: Add the shared site styles**

Write `website/src/styles/site.css`:

```css
:root {
  --bg: #020617;
  --panel: #0f172a;
  --panel-2: #111827;
  --border: #334155;
  --text: #e2e8f0;
  --muted: #94a3b8;
  --accent: #38bdf8;
  --accent-2: #f59e0b;
}

html,
body {
  margin: 0;
  background:
    radial-gradient(circle at top left, rgba(56, 189, 248, 0.14), transparent 32%),
    linear-gradient(180deg, #020617 0%, #08111f 100%);
  color: var(--text);
}

.site-shell {
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
}

.home-page {
  width: min(1120px, calc(100% - 32px));
  margin: 0 auto;
  padding: 48px 0 96px;
}

.hero,
.feature-grid,
.code-preview,
.architecture-snapshot,
.comparison-entry,
.faq-list {
  margin-top: 40px;
}

.hero__content,
.code-preview,
.architecture-snapshot,
.comparison-entry,
.faq-item,
.feature-card {
  border: 1px solid var(--border);
  background: rgba(15, 23, 42, 0.8);
  border-radius: 24px;
}

.hero__content {
  padding: 40px;
}

.eyebrow {
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 0.8rem;
}

.hero h1,
.section-heading h2 {
  margin: 0.2rem 0 0.75rem;
  line-height: 1.1;
}

.hero h1 {
  font-size: clamp(2.5rem, 6vw, 4.5rem);
}

.hero__copy,
.section-heading p,
.feature-card p,
.architecture-grid p,
.faq-item p {
  color: var(--muted);
}

.hero__actions {
  display: flex;
  gap: 12px;
  margin-top: 24px;
}

.button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 12px 18px;
  border-radius: 999px;
  text-decoration: none;
  font-weight: 600;
}

.button--primary {
  background: var(--accent);
  color: #082f49;
}

.button--secondary {
  border: 1px solid var(--border);
  color: var(--text);
}

.feature-grid,
.architecture-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 16px;
}

.feature-card,
.architecture-grid article,
.code-preview,
.comparison-entry {
  padding: 24px;
}

.code-preview pre {
  overflow: auto;
  padding: 16px;
  border-radius: 16px;
  background: #020617;
}

.faq-item {
  margin-top: 12px;
  padding: 20px 24px;
}

.faq-item summary {
  cursor: pointer;
  font-weight: 600;
}

@media (max-width: 640px) {
  .hero__actions {
    flex-direction: column;
  }
}
```

- [ ] **Step 8: Run the site locally and confirm the homepage is server-rendered**

Run:

```bash
npm --prefix website run build
sed -n '1,200p' website/dist/index.html
```

Expected:

```text
<title>Agiwo: Open-source Python AI Agent Framework and Control Plane</title>
<meta name="description" ...
<h1>Build streaming, tool-using, orchestrated LLM agents with Agiwo.</h1>
```

- [ ] **Step 9: Commit the marketing shell**

Run:

```bash
git add website/src/layouts/MarketingLayout.astro website/src/components/Hero.astro website/src/components/FeatureGrid.astro website/src/components/CodePreview.astro website/src/components/ArchitectureSnapshot.astro website/src/components/FAQList.astro website/src/pages/index.astro website/src/styles/site.css
git commit -m "feat: add public homepage and marketing components"
```

## Task 3: Add curated docs, FAQ, and comparison content

**Files:**
- Create: `website/src/content/docs/index.mdx`
- Create: `website/src/content/docs/getting-started.mdx`
- Create: `website/src/content/docs/architecture/overview.mdx`
- Create: `website/src/content/docs/guides/multi-agent.mdx`
- Create: `website/src/content/docs/guides/custom-tools.mdx`
- Create: `website/src/content/docs/guides/storage.mdx`
- Create: `website/src/content/docs/faq.mdx`
- Create: `website/src/content/docs/compare/agiwo-vs-langgraph-openai-agents-autogen.mdx`
- Modify: `docs/README.md`

- [ ] **Step 1: Create the docs landing page**

Write `website/src/content/docs/index.mdx`:

```mdx
---
title: Agiwo Docs
description: Start here for Agiwo installation, architecture, orchestration, tools, and runtime concepts.
template: splash
hero:
  title: Agiwo documentation
  tagline: Learn how to build streaming, tool-using, orchestrated AI agents in Python.
  actions:
    - text: Getting Started
      link: /docs/getting-started/
    - text: Compare Agiwo
      link: /docs/compare/agiwo-vs-langgraph-openai-agents-autogen/
---
```

- [ ] **Step 2: Copy and normalize the first doc set with frontmatter**

Run:

```bash
cp docs/getting-started.md website/src/content/docs/getting-started.mdx
mkdir -p website/src/content/docs/architecture website/src/content/docs/guides
cp docs/architecture/overview.md website/src/content/docs/architecture/overview.mdx
cp docs/guides/multi-agent.md website/src/content/docs/guides/multi-agent.mdx
cp docs/guides/custom-tools.md website/src/content/docs/guides/custom-tools.mdx
cp docs/guides/storage.md website/src/content/docs/guides/storage.mdx
```

Expected:

```text
five markdown files copied into website/src/content/docs/
```

- [ ] **Step 3: Add frontmatter to each migrated doc**

Insert at the top of `website/src/content/docs/getting-started.mdx`:

```md
---
title: Getting Started
description: Install Agiwo, configure providers, and run your first Python AI agent.
seoTitle: Getting Started with Agiwo
seoDescription: Install Agiwo and build your first streaming Python AI agent with tools and provider configuration.
---
```

Insert at the top of `website/src/content/docs/architecture/overview.mdx`:

```md
---
title: Architecture Overview
description: Understand the runtime boundaries between Agent, Tool, Scheduler, and Console in Agiwo.
seoTitle: Agiwo Architecture Overview
seoDescription: Learn how Agiwo separates agent runtime, tools, scheduler orchestration, and control-plane projections.
---
```

Insert at the top of `website/src/content/docs/guides/multi-agent.mdx`:

```md
---
title: Multi-Agent
description: Compose agents with agent-as-tool patterns and scheduler-based orchestration.
seoTitle: Agiwo Multi-Agent Orchestration
seoDescription: Explore agent-as-tool composition and scheduler-based multi-agent workflows in Agiwo.
---
```

Insert at the top of `website/src/content/docs/guides/custom-tools.mdx`:

```md
---
title: Custom Tools
description: Build custom tools on top of Agiwo's stable tool interfaces.
seoTitle: Build Custom Tools in Agiwo
seoDescription: Learn how to define tool parameters, execution behavior, and user-facing output in Agiwo.
---
```

Insert at the top of `website/src/content/docs/guides/storage.mdx`:

```md
---
title: Storage and Observability
description: Persist runs, steps, and traces while keeping execution and projections separate.
seoTitle: Agiwo Storage and Observability
seoDescription: Understand run persistence, step storage, and trace collection in Agiwo.
---
```

- [ ] **Step 4: Add a dedicated FAQ page**

Write `website/src/content/docs/faq.mdx`:

```mdx
---
title: FAQ
description: Common questions about Agiwo's runtime model, orchestration, and observability.
seoTitle: Agiwo FAQ
seoDescription: Answers to common questions about Agiwo, including orchestration, tracing, and Python AI agent workflows.
---

## What is Agiwo?

Agiwo is an open-source Python AI agent framework and control plane for streaming execution, tool use, orchestration, tracing, and persistence.

## How is Agiwo different from LangGraph?

Agiwo emphasizes explicit runtime boundaries between agent execution, tools, scheduler orchestration, and console projections. It is designed around a canonical agent runtime instead of a graph-centric programming model.

## Does Agiwo support multi-agent systems?

Yes. Agiwo supports both `Agent.as_tool()` composition and scheduler-managed child agent workflows.

## Is the Console required?

No. The SDK can be used on its own. The Console is a separate control plane.
```

- [ ] **Step 5: Add the comparison page**

Write `website/src/content/docs/compare/agiwo-vs-langgraph-openai-agents-autogen.mdx`:

```mdx
---
title: Agiwo vs LangGraph vs OpenAI Agents SDK vs AutoGen
description: Compare Agiwo with popular agent frameworks across runtime boundaries, orchestration, and observability.
seoTitle: Agiwo vs LangGraph vs OpenAI Agents SDK vs AutoGen
seoDescription: Compare Agiwo to LangGraph, OpenAI Agents SDK, and AutoGen for Python AI agent orchestration, tracing, and runtime design.
---

## Who this comparison is for

This page is for developers choosing a Python AI agent framework for production-style systems that need tool use, orchestration, and runtime visibility.

## Agiwo

- Streaming-first runtime
- Explicit tool and scheduler boundaries
- Built-in persistence and trace collection
- Separate control plane instead of mixing UI concerns into the agent core

## LangGraph

- Strong fit for graph-oriented workflow modeling
- More graph-centric mental model than Agiwo

## OpenAI Agents SDK

- Tight alignment with OpenAI-native workflows
- Simpler starting point when OpenAI is the primary platform

## AutoGen

- Known for multi-agent conversation patterns
- Different tradeoffs around orchestration control and runtime boundaries

## When to choose Agiwo

Choose Agiwo when you want a Python-first runtime with explicit separation between agent execution, tool execution, scheduler orchestration, persistence, and control-plane projections.
```

- [ ] **Step 6: Add a public docs pointer in the repo docs index**

Append to `docs/README.md`:

```md
## Public Website

The public documentation site is intended to live at `https://docs.agiwo.o-ai.tech`.

The source for the public site lives under `website/`. The Markdown files in `docs/` remain the repository-native documentation source and design/archive materials.
```

- [ ] **Step 7: Build and inspect docs routes**

Run:

```bash
npm --prefix website run build
find website/dist/docs -maxdepth 3 -type f | sort | sed -n '1,40p'
```

Expected:

```text
website/dist/docs/getting-started/index.html
website/dist/docs/architecture/overview/index.html
website/dist/docs/guides/multi-agent/index.html
website/dist/docs/compare/agiwo-vs-langgraph-openai-agents-autogen/index.html
```

- [ ] **Step 8: Commit the content launch set**

Run:

```bash
git add website/src/content/docs/index.mdx website/src/content/docs/getting-started.mdx website/src/content/docs/architecture/overview.mdx website/src/content/docs/guides/multi-agent.mdx website/src/content/docs/guides/custom-tools.mdx website/src/content/docs/guides/storage.mdx website/src/content/docs/faq.mdx website/src/content/docs/compare/agiwo-vs-langgraph-openai-agents-autogen.mdx docs/README.md
git commit -m "feat: add public docs content set"
```

## Task 4: Add robots, deployment workflow, and operator runbook

**Files:**
- Create: `website/src/pages/robots.txt.ts`
- Create: `.github/workflows/public-docs.yml`
- Create: `docs/public-site-deploy.md`

- [ ] **Step 1: Add a generated robots file**

Write `website/src/pages/robots.txt.ts`:

```ts
import type { APIRoute } from "astro";

export const GET: APIRoute = ({ site }) => {
  const origin = site?.toString().replace(/\/$/, "") ?? "https://docs.agiwo.o-ai.tech";
  const body = `User-agent: *
Allow: /

Sitemap: ${origin}/sitemap-index.xml
`;

  return new Response(body, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
    },
  });
};
```

- [ ] **Step 2: Add the GitHub Pages workflow**

Write `.github/workflows/public-docs.yml`:

```yaml
name: Public Docs

on:
  push:
    branches: [main, docs/public-docs-seo-design]
    paths:
      - "website/**"
      - ".github/workflows/public-docs.yml"
      - "README.md"
      - "docs/README.md"
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "22"
          cache: "npm"
          cache-dependency-path: website/package-lock.json
      - name: Install dependencies
        run: npm --prefix website ci
      - name: Build site
        run: npm --prefix website run build
      - name: Configure Pages
        uses: actions/configure-pages@v5
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: website/dist

  deploy:
    if: github.ref == 'refs/heads/main'
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 3: Generate the site lockfile**

Run:

```bash
npm --prefix website install
```

Expected:

```text
website/package-lock.json created
```

- [ ] **Step 4: Add the deployment and DNS runbook**

Write `docs/public-site-deploy.md`:

```md
# Public Site Deployment

## GitHub Pages

1. Open the repository Settings page.
2. Open **Pages**.
3. Set **Source** to **GitHub Actions**.
4. After the `Public Docs` workflow succeeds on `main`, confirm the site artifact is deployed.

## Custom Domain

Set the custom domain to:

`docs.agiwo.o-ai.tech`

## Cloudflare DNS

Create one of the following:

- `CNAME` record: `docs.agiwo.o-ai.tech` -> `<username>.github.io`
- or the GitHub Pages record pattern required by the Pages settings UI

Keep the record DNS-only during the initial verification pass.

## Post-deploy checks

- Open `https://docs.agiwo.o-ai.tech/`
- Open `https://docs.agiwo.o-ai.tech/robots.txt`
- Open `https://docs.agiwo.o-ai.tech/sitemap-index.xml`
- Verify the domain in Google Search Console
- Submit the sitemap in Search Console
```

- [ ] **Step 5: Build and verify generated SEO artifacts**

Run:

```bash
npm --prefix website run build
sed -n '1,80p' website/dist/robots.txt
find website/dist -maxdepth 1 -type f | sort
```

Expected:

```text
User-agent: *
Sitemap: https://docs.agiwo.o-ai.tech/sitemap-index.xml
website/dist/robots.txt
website/dist/sitemap-index.xml
```

- [ ] **Step 6: Commit the deployment layer**

Run:

```bash
git add website/src/pages/robots.txt.ts .github/workflows/public-docs.yml docs/public-site-deploy.md website/package-lock.json
git commit -m "feat: add public docs deployment workflow"
```

## Task 5: Update repository copy for discoverability

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace the README hero copy with search-oriented messaging**

Update the opening block in `README.md` to:

```md
<h1 align="center">Agiwo</h1>

<p align="center">
  <em>Open-source streaming-first Python AI agent framework and control plane</em>
</p>

<p align="center">
  Build, orchestrate, trace, and operate tool-using LLM agents with streaming execution, scheduler-based orchestration, persistence, and observability.
</p>
```

- [ ] **Step 2: Add the public docs URL near the top of the README**

Insert below the badge block in `README.md`:

```md
## Public Docs

- Website: `https://docs.agiwo.o-ai.tech`
- Getting started: `https://docs.agiwo.o-ai.tech/docs/getting-started/`
- Comparison: `https://docs.agiwo.o-ai.tech/docs/compare/agiwo-vs-langgraph-openai-agents-autogen/`
```

- [ ] **Step 3: Add a discoverability note for GitHub settings**

Append near the end of `README.md`:

```md
## Repository Discoverability Notes

Recommended GitHub repository description:

`Open-source Python AI agent framework and control plane for streaming, tool use, orchestration, tracing, and persistence.`

Recommended GitHub topics:

`ai-agents`, `python`, `llm`, `agent-framework`, `multi-agent`, `tool-calling`, `observability`, `agent-orchestration`, `fastapi`
```

- [ ] **Step 4: Verify the README still renders cleanly on GitHub**

Run:

```bash
sed -n '1,80p' README.md
```

Expected:

```text
<em>Open-source streaming-first Python AI agent framework and control plane</em>
## Public Docs
```

- [ ] **Step 5: Commit the repository copy changes**

Run:

```bash
git add README.md
git commit -m "docs: improve repository discoverability copy"
```

## Task 6: Final validation and launch checklist

**Files:**
- Validate: `website/dist/**`
- Validate: `README.md`
- Validate: `docs/public-site-deploy.md`

- [ ] **Step 1: Run the full site validation commands**

Run:

```bash
npm --prefix website run check
npm --prefix website run build
```

Expected:

```text
Result (0 errors)
build completed
```

- [ ] **Step 2: Inspect the built homepage metadata and structured data**

Run:

```bash
sed -n '1,220p' website/dist/index.html
```

Expected:

```text
<link rel="canonical" href="https://docs.agiwo.o-ai.tech/" />
<meta property="og:title" content="Agiwo: Open-source Python AI Agent Framework and Control Plane" />
<script type="application/ld+json">
```

- [ ] **Step 3: Inspect docs, robots, and sitemap outputs**

Run:

```bash
find website/dist/docs -maxdepth 3 -type f | sort | sed -n '1,40p'
sed -n '1,40p' website/dist/robots.txt
sed -n '1,80p' website/dist/sitemap-index.xml
```

Expected:

```text
website/dist/docs/getting-started/index.html
website/dist/robots.txt
<sitemapindex ...
```

- [ ] **Step 4: Review repository changes before push**

Run:

```bash
git status --short
git log --oneline --decorate -6
```

Expected:

```text
working tree clean
recent commits for scaffold, homepage, content, deployment, and README
```

- [ ] **Step 5: Push the branch and open a PR**

Run:

```bash
git push -u origin docs/public-docs-seo-design
```

Expected:

```text
branch 'docs/public-docs-seo-design' set up to track 'origin/docs/public-docs-seo-design'
```

- [ ] **Step 6: Complete the manual launch checklist**

Run through:

```text
1. Merge the PR after review.
2. Open repository Settings > Pages and confirm GitHub Actions is selected.
3. Set the custom domain to docs.agiwo.o-ai.tech.
4. Add the Cloudflare DNS record.
5. Wait for the Pages certificate to issue.
6. Verify robots.txt and sitemap on the live domain.
7. Verify the domain in Google Search Console and submit the sitemap.
8. Set the GitHub repository website field to https://docs.agiwo.o-ai.tech.
```

## Self-Review

### Spec coverage

- Public standalone site: covered by Tasks 1 and 4.
- Strong SEO homepage: covered by Task 2.
- Curated docs and comparison content: covered by Task 3.
- Technical SEO: covered by Tasks 2 and 4.
- README and repository discoverability updates: covered by Task 5.
- GitHub Pages + custom domain + Search Console flow: covered by Tasks 4 and 6.

### Placeholder scan

- No `TODO`, `TBD`, or deferred implementation markers remain in the plan.
- Commands, file paths, and initial content are explicit.

### Type and naming consistency

- The public site root is consistently `website/`.
- The public domain is consistently `https://docs.agiwo.o-ai.tech`.
- The comparison slug is consistently `agiwo-vs-langgraph-openai-agents-autogen`.
