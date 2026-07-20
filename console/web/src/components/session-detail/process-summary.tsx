"use client";

import type { ObjectiveView } from "@/lib/api";
import { compactTimelineNodes } from "@/lib/objective-status";
import { cn } from "@/lib/utils";

type ProcessSummaryProps = {
  objective: ObjectiveView | null;
};

function nodeTone(kind: string): string {
  if (kind.includes("Delivered") || kind.includes("Completed")) {
    return "bg-emerald-400";
  }
  if (kind.includes("Waiting") || kind.includes("Pause") || kind.includes("Budget")) {
    return "bg-amber-400";
  }
  if (kind.includes("Fault") || kind.includes("Failed")) {
    return "bg-red-400";
  }
  if (kind.includes("Decision") || kind.includes("Outcome")) {
    return "bg-blue-400";
  }
  return "bg-zinc-500";
}

export function ProcessSummary({ objective }: ProcessSummaryProps) {
  if (!objective) {
    return (
      <div className="rounded-xl border border-line bg-panel">
        <div className="border-b border-line px-3 py-2 text-sm font-medium">
          Process summary
        </div>
        <p className="px-3 py-3 text-xs text-ink-muted">No Objective facts yet.</p>
      </div>
    );
  }

  const nodes = compactTimelineNodes(objective, 4);

  return (
    <div className="rounded-xl border border-line bg-panel">
      <div className="border-b border-line px-3 py-2 text-sm font-medium">
        Process summary
      </div>
      <div className="space-y-2.5 px-3 py-3">
        {nodes.map((node) => (
          <div key={node.fact_id} className="flex gap-2 text-xs leading-5">
            <span
              className={cn(
                "mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full",
                nodeTone(node.kind),
              )}
            />
            <div className="min-w-0">
              <div className="font-mono text-[10px] text-ink-faint">{node.kind}</div>
              <p className="text-ink-muted">{node.summary}</p>
            </div>
          </div>
        ))}
      </div>
      {objective.timeline.length > nodes.length ? (
        <details className="border-t border-line px-3 py-2">
          <summary className="cursor-pointer text-xs text-ink-muted">
            More Objective facts ({objective.timeline.length})
          </summary>
          <ol className="mt-2 max-h-48 space-y-2 overflow-auto pb-1">
            {objective.timeline.map((node) => (
              <li key={node.fact_id} className="text-xs">
                <span className="font-mono text-[10px] text-ink-faint">
                  #{node.sequence} {node.kind}
                </span>
                <p className="text-ink-muted">{node.summary}</p>
              </li>
            ))}
          </ol>
        </details>
      ) : null}
    </div>
  );
}
