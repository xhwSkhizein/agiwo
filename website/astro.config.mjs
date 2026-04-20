import { defineConfig } from "astro/config";
import sitemap from "@astrojs/sitemap";
import starlight from "@astrojs/starlight";

export default defineConfig({
  site: "https://docs.agiwo.o-ai.tech",
  integrations: [
    sitemap(),
    starlight({
      title: "Agiwo",
      description:
        "Agiwo is a runtime harness for orchestrated, self-improving agents in Python, with tools, scheduler orchestration, persistence, tracing, and a control plane.",
      disable404Route: true,
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
            {
              label: "Context Optimization",
              link: "/docs/guides/context-optimization/",
            },
          ],
        },
        {
          label: "Concepts",
          items: [
            { label: "Agent", link: "/docs/concepts/agent/" },
            { label: "Model", link: "/docs/concepts/model/" },
            { label: "Tool", link: "/docs/concepts/tool/" },
            { label: "Scheduler", link: "/docs/concepts/scheduler/" },
            { label: "Memory", link: "/docs/concepts/memory/" },
            {
              label: "Runtime Harness",
              link: "/docs/concepts/runtime-harness/",
            },
          ],
        },
        {
          label: "Reference",
          items: [
            { label: "Model API", link: "/docs/reference/api/model/" },
            { label: "Tool API", link: "/docs/reference/api/tool/" },
            {
              label: "Scheduler API",
              link: "/docs/reference/api/scheduler/",
            },
            {
              label: "Console Overview",
              link: "/docs/reference/console/overview/",
            },
            { label: "Console API", link: "/docs/reference/console/api/" },
            {
              label: "Console Docker",
              link: "/docs/reference/console/docker/",
            },
            {
              label: "Configuration",
              link: "/docs/reference/configuration/",
            },
          ],
        },
        {
          label: "Architecture",
          items: [
            {
              label: "Architecture Overview",
              link: "/docs/architecture/overview/",
            },
            { label: "Memory System", link: "/docs/architecture/memory/" },
            { label: "Repository Overview", link: "/docs/repo-overview/" },
          ],
        },
        {
          label: "Compare",
          items: [
            {
              label: "Agiwo vs LangGraph / OpenAI Agents / AutoGen",
              link: "/docs/compare/agiwo-vs-langgraph-openai-agents-autogen/",
            },
          ],
        },
        {
          label: "FAQ",
          items: [{ label: "FAQ", link: "/docs/faq/" }],
        },
      ],
    }),
  ],
});
