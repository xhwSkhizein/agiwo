"use client";

import { useMemo, useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

import { cn } from "@/lib/utils";

type JsonDisclosureProps = {
  label: string;
  value: unknown;
  className?: string;
  contentClassName?: string;
};

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

export function JsonDisclosure({
  label,
  value,
  className,
  contentClassName,
}: JsonDisclosureProps) {
  const [expanded, setExpanded] = useState(false);
  const serialized = useMemo(
    () => (expanded ? JSON.stringify(value, null, 2) : null),
    [expanded, value],
  );

  return (
    <div className={cn("rounded-xl border border-line bg-panel-muted", className)}>
      <button
        type="button"
        aria-expanded={expanded}
        onClick={() => {
          setExpanded((current) => !current);
        }}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs text-ink-muted transition-colors hover:text-foreground"
      >
        {expanded ? (
          <ChevronDown className="h-3.5 w-3.5 shrink-0" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 shrink-0" />
        )}
        <span className="font-medium">{label}</span>
        <span className="ml-auto text-[11px] text-ink-faint">{summarizeValue(value)}</span>
      </button>
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
