"use client";

import { useMemo, useState } from "react";
import { Check, ChevronDown, ChevronRight, Copy } from "lucide-react";

import { cn } from "@/lib/utils";

type JsonDisclosureProps = {
  label: string;
  value: unknown;
  className?: string;
  contentClassName?: string;
};

/**
 * Produce a concise, human-readable summary of a value's shape or type.
 *
 * @param value - The value to summarize
 * @returns A string describing the value: "`<n> item(s)`" for arrays, "`<n> field(s)`" for non-null objects, "`<n> chars`" for strings, or the result of `typeof` for other values
 */
function summarizeValue(value: unknown): string {
  if (Array.isArray(value)) {
    return `${value.length} item${value.length === 1 ? "" : "s"}`;
  }

  if (value && typeof value === "object") {
    return `${Object.keys(value).length} field${Object.keys(value).length === 1 ? "" : "s"}`;
  }

  if (typeof value === "string") {
    return `${value.length} chars`;
  }

  return typeof value;
}

/**
 * Renders a labeled, collapsible JSON viewer that shows a short summary when closed and pretty-printed JSON when opened.
 *
 * @param label - Text shown on the toggle button
 * @param value - Data to summarize and (optionally) display as JSON when expanded
 * @param className - Optional CSS classes applied to the outer container
 * @param contentClassName - Optional CSS classes applied to the expandable `<pre>` content
 * @returns A React element containing the toggle button and, when expanded, the formatted JSON content
 */
export function JsonDisclosure({
  label,
  value,
  className,
  contentClassName,
}: JsonDisclosureProps) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  const serialized = useMemo(() => JSON.stringify(value, null, 2), [value]);
  const copyLabel = label.toLowerCase().includes("json")
    ? `Copy ${label}`
    : `Copy ${label} JSON`;

  const copyJson = async () => {
    if (!serialized) {
      return;
    }
    try {
      await navigator.clipboard.writeText(serialized);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className={cn("rounded-xl border border-line bg-panel-muted", className)}>
      <div className="flex items-center gap-1 px-3 py-2 text-xs text-ink-muted">
        <button
          type="button"
          aria-expanded={expanded}
          onClick={() => {
            setExpanded((current) => !current);
          }}
          className="flex min-w-0 flex-1 items-center gap-2 text-left transition-colors hover:text-foreground"
        >
          {expanded ? (
            <ChevronDown className="h-3.5 w-3.5 shrink-0" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5 shrink-0" />
          )}
          <span className="font-medium">{label}</span>
          <span className="ml-auto text-[11px] text-ink-faint">{summarizeValue(value)}</span>
        </button>
        <button
          type="button"
          onClick={copyJson}
          className="inline-flex h-6 items-center gap-1 rounded border border-line px-1.5 text-[11px] text-ink-faint transition-colors hover:border-line-strong hover:text-foreground"
          aria-label={copyLabel}
          title={copyLabel}
        >
          {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
          <span>{copied ? "Copied" : "Copy"}</span>
        </button>
      </div>
      {expanded && (
        <pre
          className={cn(
            "max-h-64 overflow-auto border-t border-line px-3 py-3 text-xs text-ink-soft whitespace-pre-wrap break-words",
            contentClassName,
          )}
        >
          {serialized}
        </pre>
      )}
    </div>
  );
}

export default JsonDisclosure;
