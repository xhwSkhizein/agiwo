import { defineConfig } from "astro/config";
import sitemap from "@astrojs/sitemap";
import starlight from "@astrojs/starlight";
import { docsSidebar } from "./src/config/docsSidebar.mjs";

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
      sidebar: docsSidebar,
    }),
  ],
});
