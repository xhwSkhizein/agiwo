# Agiwo Repository Overview Citations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add source-aware citations to the public repository overview page so layout items, runtime surfaces, and boundary notes all show direct supporting sources.

**Architecture:** Extend the existing manual Python generator to attach deterministic citation metadata to each repo overview entity, then keep Astro presentation thin by rendering that generated JSON through a small pair of citation UI primitives. The page will use compact inline citation chips for each item and a deduplicated `Sources` rollup per section, while retaining a global reference-documents section for broader navigation.

**Tech Stack:** Python 3.11, standard library JSON generation, Astro, Starlight, TypeScript, CSS

---

## File Structure

### New files and directories

- Create: `website/src/components/repo-overview/citation-chips.astro`
- Create: `website/src/components/repo-overview/section-sources.astro`

### Existing files to modify

- Modify: `scripts/generate_repo_overview.py`
- Modify: `website/src/generated/repo-overview.json`
- Modify: `website/src/components/repo-overview/layout-section.astro`
- Modify: `website/src/components/repo-overview/runtime-surface-list.astro`
- Modify: `website/src/components/repo-overview/boundary-list.astro`
- Modify: `website/src/components/repo-overview/repo-overview-page.astro`
- Modify: `website/src/styles/site.css`

### Tests and validation targets

- Validate: `python scripts/generate_repo_overview.py`
- Validate: `python scripts/generate_repo_overview.py --check`
- Validate: `npm --prefix website run check`
- Validate: `npm --prefix website run build`

### Notes on scope

- Do not add line-level GitHub permalinks.
- Do not make Astro inspect the repository directly.
- Do not expand this into symbol-level or API-level explanation.
- Keep the existing page structure recognizable and only add citation-aware rendering around it.

## Task 1: Extend the generator schema with citation-bearing entities

**Files:**
- Modify: `scripts/generate_repo_overview.py`

- [ ] **Step 1: Write a failing check for the new schema**

Run:

```bash
python scripts/generate_repo_overview.py --check
```

Expected:

```text
Exit code 1 because the checked-in JSON still uses the old schema without citations attached to layout items and boundaries.
```

- [ ] **Step 2: Replace the flat boundary strings with structured boundary objects and add a reusable citation helper**

Update the top half of `scripts/generate_repo_overview.py` so the generator has one consistent citation shape:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "website" / "src" / "generated" / "repo-overview.json"
AGENTS_PATH = ROOT / "AGENTS.md"


def citation(
    label: str,
    path: str,
    *,
    kind: str,
    title: str | None = None,
) -> dict[str, str]:
    payload = {
        "label": label,
        "path": path,
        "kind": kind,
    }
    if title:
        payload["title"] = title
    return payload


BOUNDARIES: list[dict[str, object]] = [
    {
        "text": "Scheduler sits on top of the agent runtime instead of the reverse direction.",
        "citations": [
            citation("Scheduler package", "agiwo/scheduler", kind="directory"),
            citation("Repository guide", "AGENTS.md", kind="document", title="AGENTS.md"),
        ],
    },
    {
        "text": "Console code should go through scheduler and agent facades rather than reading scheduler store internals directly.",
        "citations": [
            citation("Console services", "console/server/services", kind="directory"),
            citation("Repository guide", "AGENTS.md", kind="document", title="AGENTS.md"),
        ],
    },
]
```

- [ ] **Step 3: Attach citations to layout items and runtime surfaces with deterministic rules**

Add a helper and update the existing builders:

```python
def build_layout_citations(path: str) -> list[dict[str, str]]:
    citations = [
        citation(path, path, kind="directory"),
        citation("Repository guide", "AGENTS.md", kind="document", title="AGENTS.md"),
    ]
    return citations


def build_runtime_surface_citations(
    import_path: str,
    source_paths: list[str],
) -> list[dict[str, str]]:
    citations: list[dict[str, str]] = []
    if source_paths:
        citations.append(citation(import_path, source_paths[0], kind="directory"))
        for path in source_paths[1:]:
            citations.append(citation(path.split("/")[-1], path, kind="entrypoint"))
    citations.append(citation("Repository guide", "AGENTS.md", kind="document", title="AGENTS.md"))
    return citations
```

Then change the payload builders so they emit citation-bearing entities:

```python
group.append(
    {
        "path": path,
        "label": label,
        "responsibility": responsibility,
        "citations": build_layout_citations(path),
    }
)
```

```python
{
    "import_path": "agiwo.agent",
    "role": "Public entry for the canonical agent runtime, agent configuration, execution handles, and related types.",
    "source_paths": [
        "agiwo/agent",
        "agiwo/agent/__init__.py",
        "agiwo/agent/types.py",
    ],
    "citations": build_runtime_surface_citations(
        "agiwo.agent",
        ["agiwo/agent", "agiwo/agent/__init__.py", "agiwo/agent/types.py"],
    ),
}
```

- [ ] **Step 4: Keep the global document list, but rename it in the payload to match its new role**

Replace the old `source_pointers` field with `reference_documents` in the returned payload:

```python
return {
    "summary": (
        "Agiwo is organized around a canonical agent runtime, a separate scheduler "
        "orchestration layer, a tool abstraction, a model layer, and an internal control plane."
    ),
    "layout_sections": [
        {"key": key, "title": title} for key, title, _entries in LAYOUT_GROUPS
    ],
    **layout_payload,
    "runtime_surfaces": RUNTIME_SURFACES,
    "boundaries": BOUNDARIES,
    "reference_documents": build_source_pointers(),
}
```

- [ ] **Step 5: Regenerate the JSON and inspect the new structure**

Run:

```bash
python scripts/generate_repo_overview.py
python -m json.tool website/src/generated/repo-overview.json | sed -n '1,220p'
```

Expected:

```text
Each layout item contains citations.
Each runtime surface contains citations.
Boundaries are objects with text and citations.
The top-level field is reference_documents rather than source_pointers.
```

- [ ] **Step 6: Commit the generator schema update**

Run:

```bash
git add scripts/generate_repo_overview.py website/src/generated/repo-overview.json
git commit -m "feat: add repo overview citations data"
```

## Task 2: Add small reusable citation components for inline chips and section rollups

**Files:**
- Create: `website/src/components/repo-overview/citation-chips.astro`
- Create: `website/src/components/repo-overview/section-sources.astro`

- [ ] **Step 1: Add an inline citation chip component**

Create `website/src/components/repo-overview/citation-chips.astro`:

```astro
---
interface Citation {
  label: string;
  path: string;
  title?: string;
  kind: string;
}

interface Props {
  items: Citation[];
}

const { items } = Astro.props;
---

{items.length ? (
  <span class="repo-citation-chips" aria-label="Sources">
    {items.map((item) => (
      <span class={`repo-citation-chip repo-citation-chip--${item.kind}`}>
        <span class="repo-citation-chip__label">{item.label}</span>
        <code>{item.path}</code>
      </span>
    ))}
  </span>
) : null}
```

- [ ] **Step 2: Add a deduplicated section-sources component**

Create `website/src/components/repo-overview/section-sources.astro`:

```astro
---
interface Citation {
  label: string;
  path: string;
  title?: string;
  kind: string;
}

interface Props {
  items: Citation[];
}

const { items } = Astro.props;

const deduped = Array.from(
  new Map(items.map((item) => [`${item.kind}:${item.path}`, item])).values(),
);
---

{deduped.length ? (
  <div class="repo-section-sources">
    <p class="repo-section-sources__title">Sources</p>
    <ul class="repo-section-sources__list">
      {deduped.map((item) => (
        <li>
          <strong>{item.label}</strong>
          {item.title ? <> — {item.title}</> : null}
          <div class="repo-overview-subtle">
            <code>{item.path}</code>
          </div>
        </li>
      ))}
    </ul>
  </div>
) : null}
```

- [ ] **Step 3: Add a focused component smoke check**

Run:

```bash
sed -n '1,220p' website/src/components/repo-overview/citation-chips.astro
sed -n '1,240p' website/src/components/repo-overview/section-sources.astro
```

Expected:

```text
One small inline component and one small rollup component, with no page-level business logic embedded inside them.
```

- [ ] **Step 4: Commit the new citation primitives**

Run:

```bash
git add website/src/components/repo-overview/citation-chips.astro website/src/components/repo-overview/section-sources.astro
git commit -m "feat: add repo overview citation components"
```

## Task 3: Refactor the repo overview sections to consume citations

**Files:**
- Modify: `website/src/components/repo-overview/layout-section.astro`
- Modify: `website/src/components/repo-overview/runtime-surface-list.astro`
- Modify: `website/src/components/repo-overview/boundary-list.astro`
- Modify: `website/src/components/repo-overview/repo-overview-page.astro`

- [ ] **Step 1: Update the layout section component to render inline chips and a section source rollup**

Replace `website/src/components/repo-overview/layout-section.astro` with:

```astro
---
import CitationChips from "./citation-chips.astro";
import SectionSources from "./section-sources.astro";

interface Citation {
  label: string;
  path: string;
  title?: string;
  kind: string;
}

interface Item {
  path: string;
  label: string;
  responsibility: string;
  citations: Citation[];
}

interface Props {
  title: string;
  items: Item[];
}

const { title, items } = Astro.props;
const sectionSources = items.flatMap((item) => item.citations);
---

<section class="repo-overview-section">
  <h2>{title}</h2>
  <table class="repo-overview-table">
    <thead>
      <tr>
        <th>Path</th>
        <th>Responsibility</th>
      </tr>
    </thead>
    <tbody>
      {items.map((item) => (
        <tr>
          <td>
            <strong>{item.label}</strong>
            <div><code>{item.path}</code></div>
          </td>
          <td>
            <div>{item.responsibility}</div>
            <div class="repo-overview-inline-meta">
              <CitationChips items={item.citations} />
            </div>
          </td>
        </tr>
      ))}
    </tbody>
  </table>
  <SectionSources items={sectionSources} />
</section>
```

- [ ] **Step 2: Update the runtime surface and boundary components to follow the same pattern**

Replace `website/src/components/repo-overview/runtime-surface-list.astro` with:

```astro
---
import CitationChips from "./citation-chips.astro";
import SectionSources from "./section-sources.astro";

interface Citation {
  label: string;
  path: string;
  title?: string;
  kind: string;
}

interface Surface {
  import_path: string;
  role: string;
  source_paths: string[];
  citations: Citation[];
}

interface Props {
  items: Surface[];
}

const { items } = Astro.props;
const sectionSources = items.flatMap((item) => item.citations);
---

<section class="repo-overview-section">
  <h2>Core Runtime Surfaces</h2>
  <ul class="repo-overview-list">
    {items.map((item) => (
      <li>
        <div>
          <strong><code>{item.import_path}</code></strong> — {item.role}
        </div>
        <div class="repo-overview-inline-meta">
          <CitationChips items={item.citations} />
        </div>
      </li>
    ))}
  </ul>
  <SectionSources items={sectionSources} />
</section>
```

Replace `website/src/components/repo-overview/boundary-list.astro` with:

```astro
---
import CitationChips from "./citation-chips.astro";
import SectionSources from "./section-sources.astro";

interface Citation {
  label: string;
  path: string;
  title?: string;
  kind: string;
}

interface Boundary {
  text: string;
  citations: Citation[];
}

interface Props {
  items: Boundary[];
}

const { items } = Astro.props;
const sectionSources = items.flatMap((item) => item.citations);
---

<section class="repo-overview-section">
  <h2>Architectural Boundaries</h2>
  <ul class="repo-overview-list">
    {items.map((item) => (
      <li>
        <div>{item.text}</div>
        <div class="repo-overview-inline-meta">
          <CitationChips items={item.citations} />
        </div>
      </li>
    ))}
  </ul>
  <SectionSources items={sectionSources} />
</section>
```

- [ ] **Step 3: Reposition the final document list as reference material**

Update `website/src/components/repo-overview/repo-overview-page.astro`:

```astro
---
import overview from "../../generated/repo-overview.json";
import BoundaryList from "./boundary-list.astro";
import LayoutSection from "./layout-section.astro";
import RuntimeSurfaceList from "./runtime-surface-list.astro";
---

<section class="repo-overview-hero">
  <p class="eyebrow">Repository Overview</p>
  <h1>Understand Agiwo from the repository structure.</h1>
  <p>{overview.summary}</p>
</section>

<section class="repo-overview-section">
  <h2>What Agiwo Contains</h2>
  <p>
    The repository is split across the SDK runtime, the internal Console control plane,
    and a small set of supporting directories for tests, scripts, and design docs.
    This page is generated from the code tree and selected supporting documents so it
    stays closer to the current repository state than a fully hand-written overview.
  </p>
</section>

{overview.layout_sections.map((section) => (
  <LayoutSection title={section.title} items={overview[section.key]} />
))}

<RuntimeSurfaceList items={overview.runtime_surfaces} />
<BoundaryList items={overview.boundaries} />

<section class="repo-overview-section">
  <h2>Reference Documents</h2>
  <ul class="repo-overview-list">
    {overview.reference_documents.map((item) => (
      <li>
        <strong>{item.label}</strong>
        {item.title ? <> — {item.title}</> : null}
        <div class="repo-overview-subtle"><code>{item.path}</code></div>
      </li>
    ))}
  </ul>
</section>
```

- [ ] **Step 4: Run a rendering-oriented type and build check**

Run:

```bash
npm --prefix website run check
npm --prefix website run build
```

Expected:

```text
Astro and Starlight complete without prop-shape or template errors.
The built repo overview page includes inline citations and Sources sections.
```

- [ ] **Step 5: Commit the repo overview rendering refactor**

Run:

```bash
git add website/src/components/repo-overview/layout-section.astro website/src/components/repo-overview/runtime-surface-list.astro website/src/components/repo-overview/boundary-list.astro website/src/components/repo-overview/repo-overview-page.astro
git commit -m "feat: render repo overview citations"
```

## Task 4: Style the citation UI so it stays readable instead of noisy

**Files:**
- Modify: `website/src/styles/site.css`

- [ ] **Step 1: Add styles for inline chip clusters and section-level source boxes**

Append these rules to `website/src/styles/site.css` near the existing repo overview styles:

```css
.repo-overview-inline-meta {
  margin-top: 0.6rem;
}

.repo-citation-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
}

.repo-citation-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.24rem 0.55rem;
  border-radius: 999px;
  border: 1px solid var(--sl-color-hairline-light);
  background: rgba(103, 232, 249, 0.08);
  color: var(--sl-color-text);
  font-size: 0.85rem;
}

.repo-citation-chip--document {
  background: rgba(251, 191, 36, 0.1);
}

.repo-citation-chip__label {
  font-weight: 700;
}

.repo-section-sources {
  margin-top: 1rem;
  padding: 1rem 1.1rem;
  border: 1px solid var(--sl-color-hairline-light);
  border-radius: 18px;
  background: rgba(10, 22, 38, 0.42);
}

.repo-section-sources__title {
  margin: 0 0 0.7rem;
  color: var(--sl-color-white);
  font-weight: 700;
}

.repo-section-sources__list {
  display: grid;
  gap: 0.7rem;
  padding-left: 1.2rem;
  margin: 0;
}
```

- [ ] **Step 2: Verify the CSS stays within the existing site language**

Run:

```bash
sed -n '140,280p' website/src/styles/site.css
```

Expected:

```text
The new citation styles reuse the existing dark-panel visual language and do not introduce a separate theme.
```

- [ ] **Step 3: Commit the citation styling**

Run:

```bash
git add website/src/styles/site.css
git commit -m "style: refine repo overview citation presentation"
```

## Task 5: Regenerate, validate, and review the final page output

**Files:**
- Modify: `website/src/generated/repo-overview.json`

- [ ] **Step 1: Regenerate the final artifact from the updated generator**

Run:

```bash
python scripts/generate_repo_overview.py
python scripts/generate_repo_overview.py --check
```

Expected:

```text
The generate command rewrites the JSON once.
The --check command exits 0 immediately after regeneration.
```

- [ ] **Step 2: Run the website validation commands together**

Run:

```bash
npm --prefix website run check
npm --prefix website run build
```

Expected:

```text
No Astro build failures and no generated-data import failures.
```

- [ ] **Step 3: Inspect the built page content for the intended UX**

Run:

```bash
rg -n "Reference Documents|Sources|repo-citation-chip" website/src/components website/src/styles website/src/generated -S
```

Expected:

```text
The final source shows:
- inline citation chips
- per-section Sources rollups
- a final Reference Documents section
```

- [ ] **Step 4: Run repo-wide formatting and guardrails relevant to the touched areas**

Run:

```bash
uv run ruff check --ignore C901 --ignore PLR0911 --ignore PLR0912 scripts/
uv run ruff format --check scripts/
```

Expected:

```text
The generator remains lint-clean and correctly formatted.
```

- [ ] **Step 5: Commit the validated final state**

Run:

```bash
git add scripts/generate_repo_overview.py website/src/generated/repo-overview.json website/src/components/repo-overview website/src/styles/site.css
git commit -m "feat: add source-aware repo overview citations"
```

## Self-Review

- Spec coverage check:
  - hybrid citations are implemented by inline chips plus section rollups in Tasks 2 and 3
  - deterministic generator changes are covered in Task 1
  - global reference-documents repositioning is covered in Task 3
  - readability and low-noise styling are covered in Task 4
  - validation and deterministic regeneration are covered in Task 5

- Placeholder scan:
  - no `TODO`, `TBD`, or “implement later” markers remain
  - every task lists exact files, concrete commands, and expected outcomes

- Type consistency:
  - the plan uses one shared citation shape: `label`, `path`, optional `title`, and `kind`
  - boundaries are consistently modeled as `{ text, citations }`
  - the final top-level documents field is consistently named `reference_documents`
