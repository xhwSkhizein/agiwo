# Agiwo Repository Overview Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a code-tree-driven public repository overview page to the docs site, backed by a manual generator and calibrated supporting documentation.

**Architecture:** The implementation first aligns `README.md` and `AGENTS.md` with the current codebase so they can safely act as supporting sources. A manual Python generator will then scan selected repository structure and supporting docs to emit `website/src/generated/repo-overview.json`, and the Astro/Starlight public site will render that data into `/docs/repo-overview/`.

**Tech Stack:** Python 3.11, standard library file parsing, Astro, Starlight, TypeScript, MDX, JSON

---

## File Structure

### New files and directories

- Create: `scripts/generate_repo_overview.py`
- Create: `website/src/generated/repo-overview.json`
- Create: `website/src/components/repo-overview/repo-overview-page.astro`
- Create: `website/src/components/repo-overview/layout-section.astro`
- Create: `website/src/components/repo-overview/runtime-surface-list.astro`
- Create: `website/src/components/repo-overview/boundary-list.astro`
- Create: `website/src/content/docs/docs/repo-overview.mdx`

### Existing files to modify

- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `website/src/content/docs/docs/index.mdx`
- Modify: `website/astro.config.mjs`
- Modify: `website/src/styles/site.css`

### Tests and validation targets

- Validate: `python scripts/generate_repo_overview.py`
- Validate: `python scripts/generate_repo_overview.py --check`
- Validate: `npm --prefix website run check`
- Validate: `npm --prefix website run build`

### Notes on scope

- Do not build symbol-level API extraction.
- Do not make Astro scan the repository directly.
- Do not add automatic regeneration to CI in this phase.

## Task 1: Calibrate README.md against the current codebase

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Review the current README and compare it to code-facing source of truth**

Run:

```bash
sed -n '1,260p' README.md
sed -n '1,260p' AGENTS.md
```

Expected:

```text
Current README positioning, quick start, capability list, and docs links are visible.
```

- [ ] **Step 2: Replace stale or ambiguous capability bullets with code-aligned wording**

Update the README capability section so it reflects the current boundaries and wording used in the repository. Keep the section concise and public-facing.

Write this replacement block inside `README.md` under `## Current Capabilities`:

```md
## Current Capabilities

- Streaming-first agent execution through one runtime pipeline surfaced as `start()`, `run()`, and `run_stream()`
- Tool calling with builtin tools, custom `BaseTool` implementations, and agent-as-tool composition via `Agent.as_tool()`
- Scheduler orchestration for roots and child agents, including submit, routing, steering, waiting, and cancellation flows
- Run and step persistence plus trace collection with memory, SQLite, and MongoDB-backed storage options
- Global skill discovery with per-agent allowlisting through explicit `allowed_skills`
- Console APIs and web control plane for agent config management, session chat, scheduler views, and trace inspection
```

- [ ] **Step 3: Add a short repository structure pointer for public readers**

Insert this section after the public docs links in `README.md`:

```md
## Repository Structure

Agiwo has three main areas:

- `agiwo/` — the SDK runtime, including agent execution, tools, scheduler orchestration, model abstraction, memory, workspace, and observability
- `console/` — the FastAPI control plane and internal web UI
- `docs/` — design notes, concepts, and repository-native documentation

For a code-aware public architecture overview, see:

- `https://docs.agiwo.o-ai.tech/docs/repo-overview/`
```

- [ ] **Step 4: Verify the README still reads cleanly as a public entry document**

Run:

```bash
sed -n '1,140p' README.md
```

Expected:

```text
Public Docs
Repository Structure
Current Capabilities
```

- [ ] **Step 5: Commit the README calibration**

Run:

```bash
git add README.md
git commit -m "docs: calibrate readme for repo overview generation"
```

## Task 2: Calibrate AGENTS.md against the current codebase

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Verify the top-level repository layout against the actual tree**

Run:

```bash
find agiwo -maxdepth 2 -type d | sort | sed -n '1,120p'
find console -maxdepth 3 -type d | sort | sed -n '1,120p'
```

Expected:

```text
Top-level SDK and Console directories align with the responsibility tables in AGENTS.md.
```

- [ ] **Step 2: Correct any stale responsibility wording discovered during tree review**

Edit `AGENTS.md` conservatively:

- remove directory claims that no longer map to real paths
- tighten wording where the current code has clearly shifted
- preserve the document’s package-level responsibility style

Use the following rule while editing:

```text
If a directory still exists but the wording is slightly off, rewrite the responsibility sentence.
If a path no longer exists, delete the stale claim.
If a new top-level or second-level responsibility is now central and stable, add one concise row or bullet.
```

- [ ] **Step 3: Re-check the most important public boundary notes against code**

Run:

```bash
rg -n "class Agent|class Scheduler|def as_tool|def route_root_input|allowed_skills|allowed_tools" agiwo console/server -S | sed -n '1,120p'
```

Expected:

```text
Current boundary-relevant identifiers are visible for spot-checking AGENTS.md statements.
```

- [ ] **Step 4: Keep AGENTS.md at the right abstraction level**

Before saving, verify these constraints manually:

```text
- No new per-file dump was added.
- The document still focuses on directory/package responsibilities and stable APIs.
- Code-specific implementation details stay out unless they are stable boundaries.
```

- [ ] **Step 5: Commit the AGENTS.md calibration**

Run:

```bash
git add AGENTS.md
git commit -m "docs: align agents guide with current codebase"
```

## Task 3: Implement the repository overview generator

**Files:**
- Create: `scripts/generate_repo_overview.py`
- Create: `website/src/generated/repo-overview.json`

- [ ] **Step 1: Write a failing generation check by asserting the output file does not exist yet or is stale**

Run:

```bash
test -f website/src/generated/repo-overview.json && python -m json.tool website/src/generated/repo-overview.json >/dev/null || false
```

Expected:

```text
The command fails before the generator exists or before valid output is generated.
```

- [ ] **Step 2: Add the generator script skeleton**

Write `scripts/generate_repo_overview.py`:

```python
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "website" / "src" / "generated" / "repo-overview.json"


@dataclass
class ResponsibilityItem:
    path: str
    label: str
    responsibility: str


@dataclass
class RuntimeSurface:
    import_path: str
    role: str


@dataclass
class RepoOverview:
    summary: str
    sdk_layout: list[ResponsibilityItem]
    console_layout: list[ResponsibilityItem]
    supporting_layout: list[ResponsibilityItem]
    runtime_surfaces: list[RuntimeSurface]
    boundaries: list[str]
    source_pointers: list[dict[str, str]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    raise NotImplementedError(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Implement deterministic repository data assembly**

Replace the script with this minimal implementation:

```python
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "website" / "src" / "generated" / "repo-overview.json"


@dataclass
class ResponsibilityItem:
    path: str
    label: str
    responsibility: str


@dataclass
class RuntimeSurface:
    import_path: str
    role: str


@dataclass
class RepoOverview:
    summary: str
    sdk_layout: list[ResponsibilityItem]
    console_layout: list[ResponsibilityItem]
    supporting_layout: list[ResponsibilityItem]
    runtime_surfaces: list[RuntimeSurface]
    boundaries: list[str]
    source_pointers: list[dict[str, str]]


SDK_LAYOUT = [
    ResponsibilityItem("agiwo/agent", "Agent runtime", "Canonical agent runtime, execution loop, runtime state, models, hooks, nested-agent adapters, and persistence hooks."),
    ResponsibilityItem("agiwo/llm", "Model layer", "Model abstraction, provider adapters, factory construction, and message/event normalization."),
    ResponsibilityItem("agiwo/tool", "Tool layer", "Tool contracts, builtin tools, execution context, process registry, and tool-side storage."),
    ResponsibilityItem("agiwo/scheduler", "Scheduler", "Agent-level orchestration, lifecycle control, runtime tools, and scheduler state persistence."),
    ResponsibilityItem("agiwo/observability", "Observability", "Trace/span storage, query interfaces, and runtime trace adaptation."),
    ResponsibilityItem("agiwo/memory", "Memory", "Workspace memory indexing, chunking, and search services."),
]

CONSOLE_LAYOUT = [
    ResponsibilityItem("console/server", "Control plane", "FastAPI runtime integration and application services."),
    ResponsibilityItem("console/server/routers", "API boundary", "HTTP and SSE routing with request/response assembly."),
    ResponsibilityItem("console/server/services", "Application services", "Runtime management, tool catalog, registry, storage wiring, and metrics."),
    ResponsibilityItem("console/server/models", "Shared models", "Console-facing runtime, configuration, and aggregated view models."),
    ResponsibilityItem("console/server/channels", "Channel adapters", "Delivery, parsing, and integration for external channels such as Feishu."),
    ResponsibilityItem("console/web", "Internal web UI", "Internal control-plane frontend for sessions, traces, scheduler, and settings."),
]

SUPPORTING_LAYOUT = [
    ResponsibilityItem("tests", "SDK tests", "Subsystem-level tests for the SDK runtime."),
    ResponsibilityItem("docs", "Repository docs", "Design notes, architecture documents, and repository-native documentation."),
    ResponsibilityItem("scripts", "Repo scripts", "Lint entrypoints, guardrails, and local maintenance scripts."),
    ResponsibilityItem("templates", "Runtime templates", "Template assets consumed by runtime features."),
]

RUNTIME_SURFACES = [
    RuntimeSurface("agiwo.agent", "Public entry for the canonical agent runtime."),
    RuntimeSurface("agiwo.scheduler", "Public entry for agent orchestration and persistent roots."),
    RuntimeSurface("agiwo.tool", "Public entry for tool contracts and builtin tool integration."),
    RuntimeSurface("agiwo.llm", "Public entry for model abstractions and provider construction."),
]

BOUNDARIES = [
    "Scheduler depends on agent runtime instead of the reverse direction.",
    "Console code should use public scheduler and agent facades instead of store internals.",
    "Public agent types should enter through the proper facade rather than internal runtime modules.",
    "The public docs site is static and separate from the internal Console web app.",
]

SOURCE_POINTERS = [
    {"label": "Repository guide", "path": "AGENTS.md"},
    {"label": "Public entry README", "path": "README.md"},
    {"label": "Architecture overview", "path": "docs/architecture/overview.md"},
    {"label": "Public docs site", "path": "website/"},
]


def build_payload() -> RepoOverview:
    return RepoOverview(
        summary=(
            "Agiwo is organized around a canonical agent runtime, a separate scheduler "
            "orchestration layer, a tool abstraction, a model layer, and an internal control plane."
        ),
        sdk_layout=SDK_LAYOUT,
        console_layout=CONSOLE_LAYOUT,
        supporting_layout=SUPPORTING_LAYOUT,
        runtime_surfaces=RUNTIME_SURFACES,
        boundaries=BOUNDARIES,
        source_pointers=SOURCE_POINTERS,
    )


def render_payload() -> str:
    return json.dumps(asdict(build_payload()), indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    expected = render_payload()

    if args.check:
      return 0 if OUTPUT_PATH.exists() and OUTPUT_PATH.read_text() == expected else 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Generate the JSON artifact**

Run:

```bash
python scripts/generate_repo_overview.py
python -m json.tool website/src/generated/repo-overview.json >/dev/null
```

Expected:

```text
The script exits 0 and the generated JSON is valid.
```

- [ ] **Step 5: Verify deterministic check mode**

Run:

```bash
python scripts/generate_repo_overview.py --check
```

Expected:

```text
Exit code 0 because the generated file matches the current repository state.
```

- [ ] **Step 6: Commit the generator and generated artifact**

Run:

```bash
git add scripts/generate_repo_overview.py website/src/generated/repo-overview.json
git commit -m "feat: add repo overview generator"
```

## Task 4: Render the generated overview in the public docs site

**Files:**
- Create: `website/src/components/repo-overview/repo-overview-page.astro`
- Create: `website/src/components/repo-overview/layout-section.astro`
- Create: `website/src/components/repo-overview/runtime-surface-list.astro`
- Create: `website/src/components/repo-overview/boundary-list.astro`
- Create: `website/src/content/docs/docs/repo-overview.mdx`
- Modify: `website/src/content/docs/docs/index.mdx`
- Modify: `website/src/styles/site.css`

- [ ] **Step 1: Add the layout section component**

Write `website/src/components/repo-overview/layout-section.astro`:

```astro
---
interface Item {
  path: string;
  label: string;
  responsibility: string;
}

interface Props {
  title: string;
  items: Item[];
}

const { title, items } = Astro.props;
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
          <td>{item.responsibility}</td>
        </tr>
      ))}
    </tbody>
  </table>
</section>
```

- [ ] **Step 2: Add the runtime surfaces and boundary list components**

Write `website/src/components/repo-overview/runtime-surface-list.astro`:

```astro
---
interface Surface {
  import_path: string;
  role: string;
}

interface Props {
  items: Surface[];
}

const { items } = Astro.props;
---

<section class="repo-overview-section">
  <h2>Core Runtime Surfaces</h2>
  <ul class="repo-overview-list">
    {items.map((item) => (
      <li>
        <code>{item.import_path}</code> — {item.role}
      </li>
    ))}
  </ul>
</section>
```

Write `website/src/components/repo-overview/boundary-list.astro`:

```astro
---
interface Props {
  items: string[];
}

const { items } = Astro.props;
---

<section class="repo-overview-section">
  <h2>Architectural Boundaries</h2>
  <ul class="repo-overview-list">
    {items.map((item) => <li>{item}</li>)}
  </ul>
</section>
```

- [ ] **Step 3: Add the page component that consumes generated JSON**

Write `website/src/components/repo-overview/repo-overview-page.astro`:

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

<LayoutSection title="SDK Layout" items={overview.sdk_layout} />
<LayoutSection title="Console Layout" items={overview.console_layout} />
<LayoutSection title="Supporting Directories" items={overview.supporting_layout} />
<RuntimeSurfaceList items={overview.runtime_surfaces} />
<BoundaryList items={overview.boundaries} />

<section class="repo-overview-section">
  <h2>Source Pointers</h2>
  <ul class="repo-overview-list">
    {overview.source_pointers.map((item) => (
      <li>
        <strong>{item.label}</strong> — <code>{item.path}</code>
      </li>
    ))}
  </ul>
</section>
```

- [ ] **Step 4: Add the docs page and expose it from the docs index**

Write `website/src/content/docs/docs/repo-overview.mdx`:

```mdx
---
title: Repository Overview
description: Understand the Agiwo repository layout, public runtime surfaces, and architecture boundaries.
---

import RepoOverviewPage from "../../../components/repo-overview/repo-overview-page.astro";

<RepoOverviewPage />
```

Update `website/src/content/docs/docs/index.mdx` actions to include the new page:

```md
    - text: Repository Overview
      link: /docs/repo-overview/
      variant: minimal
```

- [ ] **Step 5: Add the minimal styling for overview tables and lists**

Append to `website/src/styles/site.css`:

```css
.repo-overview-hero,
.repo-overview-section {
  margin-top: 2rem;
}

.repo-overview-table {
  width: 100%;
  border-collapse: collapse;
}

.repo-overview-table th,
.repo-overview-table td {
  padding: 0.9rem 1rem;
  border-bottom: 1px solid var(--sl-color-hairline-light);
  vertical-align: top;
}

.repo-overview-list {
  display: grid;
  gap: 0.75rem;
  padding-left: 1.2rem;
}
```

- [ ] **Step 6: Build and inspect the new page**

Run:

```bash
npm --prefix website run check
npm --prefix website run build
find website/dist/docs/repo-overview -maxdepth 2 -type f | sort
```

Expected:

```text
website/dist/docs/repo-overview/index.html
```

- [ ] **Step 7: Commit the public page rendering**

Run:

```bash
git add website/src/components/repo-overview website/src/content/docs/docs/repo-overview.mdx website/src/content/docs/docs/index.mdx website/src/styles/site.css
git commit -m "feat: add public repo overview page"
```

## Task 5: Make the generator part of the documented author workflow

**Files:**
- Modify: `docs/public-site-deploy.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Document the manual generation command in the public site workflow note**

Append to `docs/public-site-deploy.md`:

```md
## Repository Overview Refresh

When repository structure or architecture boundaries change:

1. Update `README.md` and `AGENTS.md` if needed
2. Run `python scripts/generate_repo_overview.py`
3. Review `website/src/generated/repo-overview.json`
4. Rebuild the public site before publishing
```

- [ ] **Step 2: Add a short maintenance note in AGENTS.md**

Append near the `Maintaining AGENTS.md` section:

```md
- Public repository overview generation consumes repository structure first and may also use `AGENTS.md` as a supporting source; keep directory responsibilities and stable boundary notes current.
```

- [ ] **Step 3: Verify the generator workflow is discoverable**

Run:

```bash
rg -n "generate_repo_overview|Repository Overview Refresh|supporting source" docs/public-site-deploy.md AGENTS.md
```

Expected:

```text
The generator command and maintenance note are present in both files.
```

- [ ] **Step 4: Commit the workflow documentation**

Run:

```bash
git add docs/public-site-deploy.md AGENTS.md
git commit -m "docs: document repo overview refresh workflow"
```

## Task 6: Final validation and branch update

**Files:**
- Validate: `README.md`
- Validate: `AGENTS.md`
- Validate: `scripts/generate_repo_overview.py`
- Validate: `website/src/generated/repo-overview.json`
- Validate: `website/dist/docs/repo-overview/index.html`

- [ ] **Step 1: Run the generator and site validation end-to-end**

Run:

```bash
python scripts/generate_repo_overview.py
python scripts/generate_repo_overview.py --check
npm --prefix website run check
npm --prefix website run build
```

Expected:

```text
Generator exits 0 in both normal and check modes.
Astro check and build both succeed.
```

- [ ] **Step 2: Inspect the generated JSON and rendered page output**

Run:

```bash
sed -n '1,220p' website/src/generated/repo-overview.json
sed -n '1,220p' website/dist/docs/repo-overview/index.html
```

Expected:

```text
The JSON includes summary, layout groups, runtime surfaces, boundaries, and source pointers.
The rendered page includes repository overview sections with generated content.
```

- [ ] **Step 3: Review repository state before push**

Run:

```bash
git status --short
git log --oneline --decorate -8
```

Expected:

```text
Working tree clean except for unrelated untracked files such as seo_review.md.
Recent commits cover README calibration, AGENTS calibration, generator, public page, and workflow docs.
```

- [ ] **Step 4: Push the branch update**

Run:

```bash
git push origin docs/public-docs-seo-design
```

Expected:

```text
Remote branch is updated with the repo overview generation work.
```

## Self-Review

### Spec coverage

- README and AGENTS calibration: covered by Tasks 1 and 2.
- Manual generator and structured JSON output: covered by Task 3.
- Public `/docs/repo-overview/` page: covered by Task 4.
- Manual refresh workflow: covered by Task 5.
- End-to-end validation: covered by Task 6.

### Placeholder scan

- No placeholder language remains in tasks or steps.
- All new files, commands, and validation points are explicit.

### Type and naming consistency

- Generator output path is consistently `website/src/generated/repo-overview.json`.
- Public route is consistently `/docs/repo-overview/`.
- The generator entrypoint is consistently `scripts/generate_repo_overview.py`.
