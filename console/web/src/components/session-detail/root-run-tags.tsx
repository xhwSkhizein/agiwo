"use client";

import type { ObjectiveView, RootRunView } from "@/lib/api";
import { activeRootRunId } from "@/lib/objective-status";
import { cn } from "@/lib/utils";

type RootRunTagsProps = {
  objective: ObjectiveView | null;
  onSelectRunId?: (runId: string) => void;
};

function tagClass(rootRun: RootRunView, currentId: string | null): string {
  const status = rootRun.status.toUpperCase();
  if (status === "SKIPPED" || status === "INTERRUPTED") {
    return "border-line text-ink-faint line-through opacity-60";
  }
  if (status === "RUNNING") {
    return "border-cyan-500/40 bg-cyan-500/10 text-cyan-300";
  }
  if (rootRun.run_id === currentId) {
    return "border-accent/50 bg-accent/10 text-foreground";
  }
  if (status === "COMPLETED") {
    return "border-emerald-500/25 bg-emerald-500/5 text-emerald-300/90";
  }
  if (status === "FAILED") {
    return "border-red-500/30 bg-red-500/10 text-red-300";
  }
  return "border-line bg-panel text-ink-muted";
}

export function RootRunTags({ objective, onSelectRunId }: RootRunTagsProps) {
  const rootRuns = objective?.root_runs ?? [];
  if (!objective || rootRuns.length === 0) {
    return null;
  }
  const currentId = activeRootRunId(objective);

  return (
    <div className="flex flex-wrap items-center gap-2 rounded-lg border border-line bg-panel/50 px-3 py-2 text-xs">
      <span className="text-[10px] font-semibold uppercase tracking-[0.05em] text-ink-faint">
        Root run
      </span>
      {rootRuns.map((rootRun, index) => (
        <span key={rootRun.run_id} className="inline-flex items-center gap-1.5">
          <button
            type="button"
            title={`${rootRun.run_id} · ${rootRun.status}`}
            onClick={() => {
              if (rootRun.run_id && onSelectRunId) {
                onSelectRunId(rootRun.run_id);
              }
            }}
            className={cn(
              "rounded-full border px-2.5 py-0.5 font-medium capitalize transition-colors",
              tagClass(rootRun, currentId),
              rootRun.run_id && onSelectRunId
                ? "cursor-pointer hover:brightness-110"
                : "cursor-default",
            )}
          >
            {rootRun.role}
          </button>
          {index < rootRuns.length - 1 ? (
            <span className="text-ink-faint" aria-hidden>
              →
            </span>
          ) : null}
        </span>
      ))}
      <span className="ml-auto text-[11px] text-ink-faint">
        tags are context — open a run below for steps
      </span>
    </div>
  );
}
