# Agiwo Repository Overview Refresh Guardrails Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep repo overview generation manual, but make CI fail whenever the checked-in generated artifact is stale.

**Architecture:** Extend the existing `Public Docs` GitHub Actions workflow instead of creating a second workflow. The workflow will watch the repo-overview source inputs, run `python scripts/generate_repo_overview.py --check` before the Astro build, and the maintenance doc will explain how to fix stale-artifact failures locally.

**Tech Stack:** GitHub Actions, Python 3, Astro/Starlight docs build, Markdown docs

---

## File Structure

### Existing files to modify

- Modify: `.github/workflows/public-docs.yml`
- Modify: `docs/public-site-deploy.md`

### Tests and validation targets

- Validate: `python scripts/generate_repo_overview.py --check`
- Validate: `npm --prefix website run build`

### Notes on scope

- Do not add a second workflow.
- Do not auto-regenerate or auto-commit `website/src/generated/repo-overview.json`.
- Do not change Pages deployment behavior from `main` only.

## Task 1: Expand the Public Docs workflow trigger scope

**Files:**
- Modify: `.github/workflows/public-docs.yml`

- [ ] **Step 1: Inspect the current workflow trigger paths**

Run:

```bash
sed -n '1,120p' .github/workflows/public-docs.yml
```

Expected:

```text
The workflow currently watches website changes and only a small subset of repo-overview source files.
```

- [ ] **Step 2: Add repo-overview source inputs to the workflow paths**

Update the `paths:` block in `.github/workflows/public-docs.yml` to include the generator and docs inputs:

```yaml
    paths:
      - "website/**"
      - ".github/workflows/public-docs.yml"
      - "scripts/generate_repo_overview.py"
      - "README.md"
      - "AGENTS.md"
      - "docs/architecture/**"
      - "docs/guides/**"
      - "docs/getting-started.md"
      - "docs/public-site-deploy.md"
```

- [ ] **Step 3: Verify the trigger scope is explicit and minimal**

Run:

```bash
sed -n '1,80p' .github/workflows/public-docs.yml
```

Expected:

```text
The workflow now watches repo-overview source inputs without adding unrelated top-level repository paths.
```

- [ ] **Step 4: Commit the trigger-scope update**

Run:

```bash
git add .github/workflows/public-docs.yml
git commit -m "ci: expand public docs trigger scope"
```

## Task 2: Add a repo-overview freshness check before the docs build

**Files:**
- Modify: `.github/workflows/public-docs.yml`

- [ ] **Step 1: Add Python setup for the generator check**

Insert this step in the `build` job after checkout and before Node setup:

```yaml
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
```

- [ ] **Step 2: Add the deterministic freshness check before the Astro build**

Insert this step after dependency installation and before `Build site`:

```yaml
      - name: Check repo overview freshness
        run: python scripts/generate_repo_overview.py --check
```

- [ ] **Step 3: Verify the final build-job order is coherent**

Run:

```bash
sed -n '1,160p' .github/workflows/public-docs.yml
```

Expected:

```text
The build job now performs:
checkout
setup-python
setup-node
install dependencies
check repo overview freshness
build site
```

- [ ] **Step 4: Commit the freshness check**

Run:

```bash
git add .github/workflows/public-docs.yml
git commit -m "ci: check repo overview freshness in public docs workflow"
```

## Task 3: Document the failure mode and local fix workflow

**Files:**
- Modify: `docs/public-site-deploy.md`

- [ ] **Step 1: Update the maintenance section with the enforced local refresh workflow**

Replace the `## Repository Overview Refresh` section in `docs/public-site-deploy.md` with:

```md
## Repository Overview Refresh

When repository structure, architecture boundaries, or the generator rules change:

1. Update `README.md`, `AGENTS.md`, or the relevant public docs if needed
2. Run `python scripts/generate_repo_overview.py`
3. Review `website/src/generated/repo-overview.json`
4. Run `python scripts/generate_repo_overview.py --check`
5. Rebuild the public site before publishing

The `Public Docs` workflow now enforces this. If CI fails on the repo-overview freshness check, regenerate the JSON locally and commit the updated artifact in the same branch.
```

- [ ] **Step 2: Verify the doc now explains both the refresh command and the CI failure meaning**

Run:

```bash
sed -n '1,220p' docs/public-site-deploy.md
```

Expected:

```text
The maintenance section explicitly names the refresh command, the --check command, and the fix for CI freshness failures.
```

- [ ] **Step 3: Commit the maintenance-doc update**

Run:

```bash
git add docs/public-site-deploy.md
git commit -m "docs: document repo overview freshness guardrail"
```

## Task 4: Validate the local workflow still passes

**Files:**
- Modify: `.github/workflows/public-docs.yml`
- Modify: `docs/public-site-deploy.md`

- [ ] **Step 1: Run the generator freshness check locally**

Run:

```bash
python scripts/generate_repo_overview.py --check
```

Expected:

```text
Exit code 0 because the checked-in generated artifact is current.
```

- [ ] **Step 2: Run the public docs build locally**

Run:

```bash
npm --prefix website run build
```

Expected:

```text
The docs site still builds successfully after the workflow and documentation changes.
```

- [ ] **Step 3: Inspect the diff for only the intended workflow and doc updates**

Run:

```bash
git diff --stat
git status --short
```

Expected:

```text
Only .github/workflows/public-docs.yml and docs/public-site-deploy.md are modified for this task, ignoring unrelated local files such as seo_review.md.
```

- [ ] **Step 4: Commit the validated final state**

Run:

```bash
git add .github/workflows/public-docs.yml docs/public-site-deploy.md
git commit -m "ci: guard repo overview freshness"
```

## Self-Review

- Spec coverage:
  - trigger-scope expansion is covered in Task 1
  - CI freshness enforcement is covered in Task 2
  - maintenance documentation is covered in Task 3
  - local validation is covered in Task 4

- Placeholder scan:
  - no `TODO`, `TBD`, or vague “handle this later” steps remain
  - each task includes exact files, commands, and expected outcomes

- Type and naming consistency:
  - the plan consistently refers to the existing `Public Docs` workflow
  - the guarded command is consistently `python scripts/generate_repo_overview.py --check`
  - the generated artifact is consistently `website/src/generated/repo-overview.json`
