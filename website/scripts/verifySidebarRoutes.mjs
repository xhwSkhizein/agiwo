import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { docsSidebar } from "../src/config/docsSidebar.mjs";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const websiteRoot = path.resolve(scriptDir, "..");
const docsRoot = path.join(websiteRoot, "src", "content", "docs", "docs");

function collectLinks(items) {
  const links = [];
  for (const item of items) {
    if (typeof item?.link === "string") {
      links.push(item.link);
    }
    if (Array.isArray(item?.items)) {
      links.push(...collectLinks(item.items));
    }
  }
  return links;
}

function candidatesForLink(link) {
  if (!link.startsWith("/docs/")) {
    return [];
  }
  const relative = link.replace(/^\/docs\/?/, "").replace(/\/$/, "");
  if (relative === "") {
    return [path.join(docsRoot, "index.mdx"), path.join(docsRoot, "index.md")];
  }
  const directBase = path.join(docsRoot, relative);
  return [
    `${directBase}.mdx`,
    `${directBase}.md`,
    path.join(directBase, "index.mdx"),
    path.join(directBase, "index.md"),
  ];
}

const missing = [];
for (const link of collectLinks(docsSidebar)) {
  const candidates = candidatesForLink(link);
  if (candidates.length === 0) {
    continue;
  }
  if (!candidates.some((candidate) => fs.existsSync(candidate))) {
    missing.push({ link, candidates });
  }
}

if (missing.length > 0) {
  console.error("Sidebar route verification failed.");
  for (const item of missing) {
    console.error(`- Missing route target for ${item.link}`);
    for (const candidate of item.candidates) {
      console.error(`  checked: ${path.relative(websiteRoot, candidate)}`);
    }
  }
  process.exit(1);
}

console.log(`Verified ${collectLinks(docsSidebar).length} sidebar doc links.`);
