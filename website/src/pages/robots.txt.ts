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
