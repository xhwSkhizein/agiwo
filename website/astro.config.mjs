import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

export default defineConfig({
  site: "https://docs.agiwo.o-ai.tech",
  integrations: [
    starlight({
      title: "Agiwo Docs",
      description:
        "Open-source Python AI agent framework and control plane docs for streaming, tool use, orchestration, tracing, and persistence.",
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
          label: "Documentation",
          autogenerate: { directory: "docs" },
        },
      ],
    }),
  ],
});
