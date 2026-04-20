# Documentation Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align repository-facing documentation with the current SDK, Console, and scheduler/session implementation.

**Architecture:** Audit the user-facing entry docs first, then fix API/reference pages against the current public Python and FastAPI surfaces, and finally refresh internal package READMEs so directory responsibilities match the real code layout. Keep examples small and bias toward stable contracts instead of implementation trivia.

**Tech Stack:** Markdown, Python SDK (`agiwo/`), FastAPI Console (`console/server/`)

---

## Task 1: Refresh top-level documentation entrypoints

**Files:**
- Modify: `README.md`
- Modify: `docs/README.md`
- Modify: `docs/getting-started.md`
- Create or modify: `docs/concepts/agent.md`

- [ ] Verify current public SDK surface and broken links from the top-level docs.
- [ ] Rewrite stale quick-start and navigation sections to match the current `Agent`, `Model`, and built-in tool APIs.
- [ ] Add or restore the missing Agent concept page so doc links resolve.

### Task 2: Realign SDK concept/reference guides

**Files:**
- Modify: `docs/concepts/model.md`
- Modify: `docs/concepts/tool.md`
- Modify: `docs/concepts/scheduler.md`
- Modify: `docs/api/model.md`
- Modify: `docs/api/tool.md`
- Modify: `docs/api/scheduler.md`
- Modify: `docs/guides/storage.md`

- [ ] Update examples and field descriptions against the current `agiwo.agent`, `agiwo.llm`, `agiwo.tool`, and `agiwo.scheduler` public exports.
- [ ] Remove claims that no longer match runtime behavior, especially old session/chat routes and outdated storage wiring.
- [ ] Keep descriptions focused on stable contracts, not internal implementation details that churn quickly.

### Task 3: Realign Console operator docs

**Files:**
- Modify: `docs/console/overview.md`
- Modify: `docs/console/api.md`
- Modify: `docs/console/feishu.md`
- Modify: `docs/console/docker.md`

- [ ] Update CLI, route, and deployment docs to reflect the current `agiwo-console` commands and FastAPI routes.
- [ ] Replace deprecated `/api/chat/...` material with the current session-first SSE flow.
- [ ] Fix Feishu integration docs so service names and directory references match the current runtime/services split.

### Task 4: Refresh internal Console package READMEs

**Files:**
- Modify: `console/server/README.md`
- Modify: `console/server/channels/README.md`
- Modify: `console/server/services/README.md`
- Modify: `console/server/channels/feishu/README.md`

- [ ] Update directory maps and control-flow notes to match the actual package structure.
- [ ] Remove references to deleted modules and rename runtime/service owners accurately.

### Task 5: Verify documentation consistency

**Files:**
- Verify: `README.md`
- Verify: `docs/**/*.md`
- Verify: `console/server/**/*.md`

- [ ] Run targeted searches for removed routes, missing modules, and broken doc references.
- [ ] Run repository lint/checks appropriate for Markdown-only changes if available.
- [ ] Review the touched docs for consistency with current code terminology.
