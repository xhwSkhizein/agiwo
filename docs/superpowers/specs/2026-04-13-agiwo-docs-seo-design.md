# Agiwo Public Docs And SEO Site Design

## Summary

Agiwo needs a public, indexable website that can rank for category terms such as "Python AI agent framework" and "multi-agent orchestration", instead of relying on the GitHub repository page alone.

This design introduces a separate static documentation site for `docs.agiwo.o-ai.tech` using `Astro + Starlight`. The site will have a search-oriented homepage at `/`, product documentation under `/docs/`, and dedicated comparison/content pages that target problem-oriented search queries. The internal Console web app remains private and out of scope for public SEO.

## Goals

- Create a public site that Google can crawl, index, and understand as the canonical homepage for Agiwo.
- Add a strong SEO landing page that clearly explains what Agiwo is, who it is for, and which problems it solves.
- Reuse the existing Markdown documentation in `docs/` instead of rewriting the entire documentation set.
- Establish basic technical SEO hygiene from day one: metadata, sitemap, robots, canonical URLs, and social cards.
- Make the GitHub repository point to the public site so brand and category signals converge on one canonical web property.

## Non-Goals

- Exposing `console/web` publicly or turning it into a marketing/documentation site.
- Building a full blog platform in the first phase.
- Adding versioned docs in the first phase.
- Rewriting every existing document for launch.
- Automating Cloudflare account setup from inside the repository.

## Constraints And Assumptions

- `console/web` is an internal control plane and must remain isolated from the public website.
- The public site should live at `docs.agiwo.o-ai.tech`.
- The repository already contains useful documentation in `docs/`, but no existing static site framework or deployment pipeline for public docs.
- Initial deployment should be simple, static, and easy to operate through GitHub Pages.
- DNS and final domain binding may require manual steps in Cloudflare and repository settings.

## Recommended Approach

Use `Astro + Starlight` as a separate public docs site inside the repository.

Why this approach:

- It supports a custom homepage better than doc-first tools such as MkDocs.
- It keeps Markdown-based docs easy to migrate and maintain.
- It produces a static site that works well with GitHub Pages.
- It provides a clean path to future additions such as comparison pages, guides, and blog-like content without replacing the foundation.

Rejected alternatives:

- `MkDocs`: faster for pure docs, but weak for a strong SEO landing page and comparison-style content.
- `Docusaurus`: feature-rich, but heavier than needed for the current scope.

## Site Architecture

The public site will be a new standalone frontend project dedicated to external discovery and documentation.

Proposed structure:

- `/`: SEO-oriented homepage
- `/docs/`: documentation pages migrated from the existing `docs/` tree
- `/compare/`: comparison pages targeting high-intent search traffic
- `/faq/`: short, indexable answers to common product questions

The site should be built and deployed independently from `console/web`.

Suggested repository placement:

- `website/` or `docs-site/` as the new public site project root

The implementation phase should choose one name and keep it explicit. `website/` is the preferred directory name because it is broader than pure documentation and better matches the homepage-first strategy.

## Information Architecture

### Homepage

The homepage is the primary entry point for Google and first-time visitors. It should behave like a product landing page, not like a README mirror or doc index.

Homepage sections:

1. Hero
   - Primary headline built around category language, not just the brand name
   - Supporting copy explaining SDK + orchestration + tracing + control plane
   - Primary CTA to Getting Started
   - Secondary CTA to GitHub
2. Why Agiwo
   - Streaming-first execution
   - Tool calling
   - Scheduler-based orchestration
   - Persistence and traceability
   - SDK + control plane separation
3. Code Example
   - Minimal Python agent example
   - Optional link to full Getting Started guide
4. Architecture Snapshot
   - High-level explanation of Agent, Tool, Scheduler, and Console roles
5. Comparison Entry
   - Clear link to comparison content
6. FAQ
   - Short answers to high-intent questions

### Docs

Initial docs surfaced under `/docs/`:

- Getting Started
- Architecture Overview
- Multi-Agent
- Custom Tools
- Storage & Observability

Other existing docs can be migrated incrementally after launch.

### Comparison Content

Initial comparison page:

- `Agiwo vs LangGraph vs OpenAI Agents SDK vs AutoGen`

This page should target real search intent and explain design tradeoffs, not just feature marketing.

### FAQ

Initial FAQ questions:

- What is Agiwo?
- Who is Agiwo for?
- How is Agiwo different from LangGraph?
- Does Agiwo support multi-agent orchestration?
- Does Agiwo include tracing and persistence?

## Content Strategy

The public site should target two search classes:

1. Brand search
   - `Agiwo`
   - `Agiwo docs`
   - `Agiwo GitHub`
2. Problem/category search
   - `python ai agent framework`
   - `open source ai agent sdk python`
   - `multi-agent orchestration python`
   - `tool-using llm agents`
   - `agent tracing and control plane`

Copy rules:

- Homepage H1 and description must use category terms explicitly.
- Brand terms should always be paired with descriptive category language.
- Comparison pages should focus on decision-making clarity and tradeoffs.
- Docs page titles should stay literal and searchable.

## Technical SEO Requirements

The first public release should include:

- Unique `title` and `description` per page
- Canonical URL support
- `sitemap.xml`
- `robots.txt`
- Open Graph metadata
- Twitter/X card metadata
- Clean static URLs
- Reasonable internal linking from homepage to docs and comparison pages
- Basic structured data for the homepage and article-like pages where useful

Out of scope for first release:

- Advanced analytics-driven SEO experiments
- Programmatic large-scale keyword landing pages
- Multilingual SEO

## Repository Changes

### New Public Site

Add a new project directory for the public site, preferably `website/`, containing:

- Astro/Starlight configuration
- Content collections or markdown ingestion setup
- Homepage implementation
- SEO helpers
- GitHub Pages deployment support

### Documentation Migration

The implementation should reuse the existing `docs/` content as source material. Depending on framework ergonomics, this can be done by either:

- copying curated docs into the site content tree, or
- syncing/importing curated docs from `docs/`

The first phase should optimize for simplicity and launch speed. Avoid overengineering a docs ingestion pipeline unless it clearly reduces maintenance cost.

### Repository Metadata Updates

Update repository-facing content to reinforce discoverability:

- Improve `README.md` first-screen copy around category keywords
- Point the GitHub repository website field to `https://docs.agiwo.o-ai.tech`
- Prepare a recommended repository description and topic set for manual GitHub settings updates

Suggested repository description:

`Open-source Python AI agent framework and control plane for streaming, tool use, orchestration, tracing, and persistence.`

Suggested repository topics:

- `ai-agents`
- `python`
- `llm`
- `agent-framework`
- `multi-agent`
- `tool-calling`
- `observability`
- `agent-orchestration`
- `fastapi`

## Deployment Design

Deployment target:

- GitHub Pages serving `docs.agiwo.o-ai.tech`

Deployment flow:

1. Build the static site in GitHub Actions
2. Publish the generated output to GitHub Pages
3. Bind the custom domain in repository Pages settings
4. Create the required DNS records in Cloudflare
5. Verify the final public URL, sitemap, and robots behavior

Operational boundary:

- GitHub Actions and Pages configuration live in the repository
- Cloudflare DNS changes are manual unless explicitly automated later

## Search Console And Post-Launch Setup

After launch:

1. Verify `docs.agiwo.o-ai.tech` in Google Search Console
2. Submit `https://docs.agiwo.o-ai.tech/sitemap-index.xml` or the actual generated sitemap path
3. Request indexing for the homepage and the initial comparison page
4. Update the GitHub repository website field
5. Publish at least one external announcement linking to the new site

## Implementation Phases

### Phase 1: Foundation

- Create Astro/Starlight public site project
- Configure base metadata, sitemap, robots, and social metadata
- Implement homepage shell and navigation
- Add GitHub Pages workflow

### Phase 2: Content Launch

- Migrate the initial documentation set
- Add the first comparison page
- Add homepage FAQ
- Improve `README.md` first-screen copy

### Phase 3: Search Readiness

- Bind custom domain
- Verify Search Console
- Submit sitemap
- Update repository metadata and external links

## Risks

- The site may ship with good technical SEO but weak topical authority if comparison and explanatory content remain too thin.
- A heavy migration pipeline from `docs/` into the site could slow down launch without meaningful SEO benefit.
- If repository metadata is not updated in GitHub settings, brand/entity signals remain weaker than they should be.
- If the custom domain is not verified and submitted in Search Console, indexing feedback loops will stay slow.

## Testing And Validation

Implementation should validate:

- Static site builds successfully in CI
- Key pages render with correct title, description, canonical, and Open Graph tags
- `robots.txt` and sitemap are generated
- Internal links work between homepage, docs, FAQ, and comparison pages
- Deployed Pages site resolves under the custom domain after DNS setup

Manual validation checklist after deploy:

- Inspect homepage HTML and confirm important copy is server-rendered
- Confirm the sitemap is reachable
- Confirm `robots.txt` is reachable
- Confirm social previews resolve
- Confirm Google Search Console can verify the property

## Open Decisions Resolved In This Design

- Use a separate public site instead of reusing `console/web`
- Use `Astro + Starlight`
- Use a strong SEO homepage at `/`
- Use `docs.agiwo.o-ai.tech` as the canonical public domain
- Keep Cloudflare configuration manual in the first phase

## Success Criteria

- Agiwo has a public canonical site at `docs.agiwo.o-ai.tech`
- The homepage clearly states what Agiwo is using searchable language
- Existing high-value docs are available under `/docs/`
- At least one comparison page exists and is internally linked
- Sitemap and robots are live
- The GitHub repository points to the public site
