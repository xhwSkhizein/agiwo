"use client";

import Link from "next/link";
import { useMemo, useState } from "react";

import { MonoText } from "@/components/mono-text";
import { PaginationControls } from "@/components/pagination-controls";
import { SessionStepCard } from "@/components/session-detail/session-step-card";
import { EmptyStateMessage } from "@/components/state-message";
import { contentText } from "@/components/step-content-preview";
import type { RunResponse, StepResponse } from "@/lib/api";
import {
  formatDurationMs,
  formatUsd,
  parseGenericMetrics,
} from "@/lib/metrics";
import { cn } from "@/lib/utils";

type StepRoleFilter = "all" | "user" | "assistant" | "tool";

type RunsStepsPanelProps = {
  sessionId: string;
  runs: RunResponse[];
  selectedRunId: string | null;
  onSelectRunId: (runId: string) => void;
  steps: StepResponse[];
  stepsLoading: boolean;
  stepsError: string | null;
  hasMoreSteps: boolean;
  loadingMoreSteps: boolean;
  onLoadEarlierSteps: () => void;
  runsOffset: number;
  runsPageSize: number;
  runsTotal: number | null;
  runsHasMore: boolean;
  onRunsPageSizeChange: (size: number) => void;
  onRunsPrevious: () => void;
  onRunsNext: () => void;
  runsLoading: boolean;
};

function stepMatchesSearch(step: StepResponse, query: string): boolean {
  if (!query) {
    return true;
  }
  const haystack = [
    step.role,
    String(step.sequence),
    step.name ?? "",
    step.content_for_user ?? "",
    contentText(step.content) ?? "",
  ]
    .join("\n")
    .toLowerCase();
  return haystack.includes(query.toLowerCase());
}

function runTitle(run: RunResponse): string {
  if (run.parent_run_id) {
    return `child · ${run.agent_id || "agent"}`;
  }
  return run.agent_id || "root run";
}

export function RunsStepsPanel({
  sessionId,
  runs,
  selectedRunId,
  onSelectRunId,
  steps,
  stepsLoading,
  stepsError,
  hasMoreSteps,
  loadingMoreSteps,
  onLoadEarlierSteps,
  runsOffset,
  runsPageSize,
  runsTotal,
  runsHasMore,
  onRunsPageSizeChange,
  onRunsPrevious,
  onRunsNext,
  runsLoading,
}: RunsStepsPanelProps) {
  const [roleFilter, setRoleFilter] = useState<StepRoleFilter>("all");
  const [search, setSearch] = useState("");

  const selectedRun = runs.find((run) => run.id === selectedRunId) ?? null;
  const selectedMetrics = selectedRun
    ? parseGenericMetrics(selectedRun.metrics ?? undefined)
    : null;

  const filteredSteps = useMemo(
    () =>
      steps.filter((step) => {
        if (roleFilter !== "all" && step.role !== roleFilter) {
          return false;
        }
        return stepMatchesSearch(step, search.trim());
      }),
    [roleFilter, search, steps],
  );

  const relatedTraceHref = selectedRunId
    ? `/traces?session_id=${encodeURIComponent(sessionId)}`
    : `/traces?session_id=${encodeURIComponent(sessionId)}`;

  return (
    <section className="overflow-hidden rounded-xl border border-line bg-panel">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line bg-panel-muted px-4 py-3">
        <div>
          <h2 className="text-sm font-semibold text-foreground">
            Agent Runs &amp; Steps
          </h2>
          <p className="text-xs text-ink-muted">
            Select a run, then read its steps and tool calls
          </p>
        </div>
        <Link
          href={relatedTraceHref}
          className="rounded-md border border-line px-2.5 py-1 text-xs text-ink-muted hover:border-accent/40 hover:text-foreground"
        >
          Related traces
        </Link>
      </div>

      <div className="grid gap-0 lg:grid-cols-[15rem_minmax(0,1fr)]">
        <div className="border-b border-line lg:border-b-0 lg:border-r lg:border-line">
          <div className="px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.05em] text-ink-faint">
            Runs
          </div>
          <div className="max-h-[28rem] space-y-1 overflow-auto px-2 pb-3">
            {runs.length === 0 ? (
              <p className="px-2 py-4 text-xs text-ink-muted">No runs yet.</p>
            ) : (
              runs.map((run) => {
                const metrics = parseGenericMetrics(run.metrics ?? undefined);
                const selected = run.id === selectedRunId;
                return (
                  <button
                    key={run.id}
                    type="button"
                    onClick={() => onSelectRunId(run.id)}
                    className={cn(
                      "w-full rounded-lg border px-2.5 py-2 text-left transition-colors",
                      run.parent_run_id ? "ml-2 border-l-2 border-l-accent/30" : "",
                      selected
                        ? "border-accent/40 bg-accent/10"
                        : "border-transparent hover:border-line hover:bg-panel-muted",
                    )}
                  >
                    <div className="truncate font-mono text-[10px] text-ink-faint">
                      {run.id}
                    </div>
                    <div className="mt-0.5 truncate text-xs font-medium text-foreground">
                      {runTitle(run)}
                    </div>
                    <div className="mt-0.5 text-[11px] text-ink-muted">
                      {run.status} · {metrics.stepCount} steps ·{" "}
                      {formatUsd(metrics.tokenCost)}
                    </div>
                  </button>
                );
              })
            )}
          </div>
          <div className="border-t border-line px-2 py-2">
            <PaginationControls
              offset={runsOffset}
              pageSize={runsPageSize}
              itemCount={runs.length}
              totalCount={runsTotal}
              hasMore={runsHasMore}
              itemLabel="runs"
              disabled={runsLoading}
              onPageSizeChange={onRunsPageSizeChange}
              onPrevious={onRunsPrevious}
              onNext={onRunsNext}
            />
          </div>
        </div>

        <div className="min-w-0 space-y-3 p-3">
          {selectedRun ? (
            <div className="flex flex-wrap items-center gap-2 text-xs text-ink-muted">
              <span className="rounded-full border border-line px-2 py-0.5 capitalize">
                {selectedRun.status}
              </span>
              <span>
                {selectedMetrics?.stepCount ?? 0} steps ·{" "}
                {selectedMetrics?.toolCallsCount ?? 0} tools ·{" "}
                {formatDurationMs(selectedMetrics?.durationMs ?? 0)} ·{" "}
                {formatUsd(selectedMetrics?.tokenCost ?? 0)}
              </span>
              <MonoText className="ml-auto text-[11px]">{selectedRun.id}</MonoText>
            </div>
          ) : (
            <p className="text-xs text-ink-muted">Select a run to inspect steps.</p>
          )}

          <div className="flex flex-wrap items-end gap-2">
            <label className="min-w-[12rem] flex-1 space-y-1">
              <span className="text-[10px] uppercase tracking-wide text-ink-faint">
                Search
              </span>
              <input
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="tool, content, sequence"
                className="w-full rounded-md border border-line bg-background px-2.5 py-1.5 text-sm text-foreground placeholder:text-ink-faint"
              />
            </label>
            <div className="flex gap-1">
              {(["all", "user", "assistant", "tool"] as StepRoleFilter[]).map(
                (role) => (
                  <button
                    key={role}
                    type="button"
                    aria-pressed={roleFilter === role}
                    onClick={() => setRoleFilter(role)}
                    className={cn(
                      "rounded-full border px-2.5 py-1 text-xs capitalize",
                      roleFilter === role
                        ? "border-accent bg-panel-strong text-foreground"
                        : "border-line text-ink-muted hover:border-line-strong",
                    )}
                  >
                    {role}
                  </button>
                ),
              )}
            </div>
            {hasMoreSteps ? (
              <button
                type="button"
                onClick={onLoadEarlierSteps}
                disabled={loadingMoreSteps}
                className="rounded-md border border-line px-2.5 py-1 text-xs text-ink-muted disabled:opacity-40"
              >
                {loadingMoreSteps ? "Loading…" : "Load earlier"}
              </button>
            ) : null}
          </div>

          {stepsError ? (
            <p className="text-sm text-red-300">{stepsError}</p>
          ) : null}

          {stepsLoading && steps.length === 0 ? (
            <p className="py-8 text-center text-sm text-ink-muted">Loading steps…</p>
          ) : filteredSteps.length === 0 ? (
            <EmptyStateMessage className="py-8 text-center text-ink-muted">
              {selectedRunId
                ? "No steps match the current filters"
                : "Select a run to load steps"}
            </EmptyStateMessage>
          ) : (
            <div className="space-y-2">
              {filteredSteps.map((step) => (
                <SessionStepCard key={step.id} step={step} />
              ))}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
