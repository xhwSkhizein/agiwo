"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { Workflow } from "lucide-react";
import { BackHeader } from "@/components/back-header";
import { MetricCard } from "@/components/metric-card";
import { PaginationControls } from "@/components/pagination-controls";
import { MonoText } from "@/components/mono-text";
import { ConversationEventList } from "@/components/session-detail/conversation-event-list";
import { MilestoneBoard } from "@/components/session-detail/milestone-board";
import { SessionObservabilityPanel } from "@/components/session-detail/session-observability-panel";
import { SessionStepCard } from "@/components/session-detail/session-step-card";
import { contentText } from "@/components/step-content-preview";
import {
  useSessionDetailResource,
  useSessionRunsPage,
  useSessionStepsFeed,
} from "@/components/session-detail/use-session-detail-data";
import {
  EmptyStateMessage,
  ErrorStateMessage,
  TextStateMessage,
} from "@/components/state-message";
import { TokenSummaryCards } from "@/components/token-summary-cards";
import {
  formatDurationMs,
  formatTokenCount,
  formatUsd,
  normalizeRunMetricsSummary,
  parseGenericMetrics,
} from "@/lib/metrics";
import { getSchedulerRunResultView } from "@/lib/scheduler-run-result";
import { formatLocalDateTime } from "@/lib/time";

type StepRoleFilter = "all" | "user" | "assistant" | "tool";

function stepMatchesSearch(step: { content: unknown; content_for_user: string | null; name: string | null; role: string; sequence: number }, query: string): boolean {
  if (!query) {
    return true;
  }
  const haystack = [
    step.role,
    String(step.sequence),
    step.name ?? "",
    step.content_for_user ?? "",
    contentText(step.content) ?? "",
    JSON.stringify(step.content ?? ""),
  ]
    .join("\n")
    .toLowerCase();
  return haystack.includes(query.toLowerCase());
}

export default function SessionDetailPage() {
  const params = useParams();
  const router = useRouter();
  const searchParams = useSearchParams();
  const sessionId = params.id as string;
  const [viewMode, setViewMode] = useState<"mainline" | "debug">(
    searchParams.get("view") === "debug" ? "debug" : "mainline",
  );
  const [runsOffset, setRunsOffset] = useState(0);
  const [runsPageSize, setRunsPageSize] = useState(50);
  const [stepRoleFilter, setStepRoleFilter] = useState<StepRoleFilter>("all");
  const [stepSearch, setStepSearch] = useState("");
  const debugActive = viewMode === "debug";
  const detailState = useSessionDetailResource(sessionId);
  const runsState = useSessionRunsPage(
    sessionId,
    runsPageSize,
    runsOffset,
    debugActive,
  );
  const stepsState = useSessionStepsFeed(sessionId, debugActive);
  const detail = detailState.detail;
  const steps = stepsState.steps;
  const runs = runsState.runs;
  const loading =
    detailState.loading ||
    (debugActive &&
      (stepsState.loading || (runsState.loading && runs.length === 0)));
  const error =
    detailState.error ||
    (debugActive ? runsState.error || stepsState.error : null);

  const runTotals = normalizeRunMetricsSummary(detail?.summary.metrics);
  const schedulerResult = getSchedulerRunResultView(
    detail?.scheduler_state?.last_run_result,
    detail?.scheduler_state?.result_summary,
  );
  const runTableRows = useMemo(
    () =>
      runs.map((run) => ({
        run,
        metrics: parseGenericMetrics(run.metrics ?? undefined),
      })),
    [runs],
  );
  const filteredSteps = useMemo(
    () =>
      steps.filter((step) => {
        if (stepRoleFilter !== "all" && step.role !== stepRoleFilter) {
          return false;
        }
        return stepMatchesSearch(step, stepSearch.trim());
      }),
    [stepRoleFilter, stepSearch, steps],
  );
  const updateViewMode = (nextViewMode: "mainline" | "debug") => {
    setViewMode(nextViewMode);
    const next = new URLSearchParams(searchParams.toString());
    if (nextViewMode === "debug") {
      next.set("view", "debug");
    } else {
      next.delete("view");
    }
    router.replace(`/sessions/${sessionId}${next.toString() ? `?${next}` : ""}`);
  };

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <BackHeader
        href="/sessions"
        title="Session Detail"
        subtitle={sessionId}
      />

      {loading ? (
        <TextStateMessage>Loading session metrics...</TextStateMessage>
      ) : error ? (
        <ErrorStateMessage>{error}</ErrorStateMessage>
      ) : !detail ? (
        <EmptyStateMessage>Session not found</EmptyStateMessage>
      ) : (
        <div className="space-y-6">
          <div className="flex flex-wrap items-center gap-3 text-xs text-zinc-500">
            {detail.scheduler_state && (
              <>
                <Link
                  href={`/scheduler/${detail.scheduler_state.id}`}
                  className="inline-flex items-center gap-1 rounded border border-zinc-800 px-2 py-1 hover:border-zinc-700 hover:text-zinc-300"
                >
                  <Workflow className="w-3 h-3" />
                  Scheduler State
                </Link>
                <Link
                  href={`/scheduler/${
                    detail.scheduler_state.root_state_id ??
                    detail.scheduler_state.id
                  }/tree`}
                  className="inline-flex items-center gap-1 rounded border border-zinc-800 px-2 py-1 hover:border-zinc-700 hover:text-zinc-300"
                >
                  <Workflow className="w-3 h-3" />
                  Scheduler Tree
                </Link>
              </>
            )}
            <Link
              href={`/traces?session_id=${sessionId}`}
              className="rounded border border-zinc-800 px-2 py-1 hover:border-zinc-700 hover:text-zinc-300"
            >
              Related Traces
            </Link>
          </div>

          <TokenSummaryCards
            cost={runTotals.token_cost}
            inputTokens={runTotals.input_tokens}
            outputTokens={runTotals.output_tokens}
            totalTokens={runTotals.total_tokens}
            cacheReadTokens={runTotals.cache_read_tokens}
            cacheCreationTokens={runTotals.cache_creation_tokens}
            className="grid grid-cols-2 md:grid-cols-5 gap-3"
            cardClassName="p-3"
            labelClassName="text-[11px]"
            valueClassName="text-sm font-medium"
            extraCardsPosition="after"
            extraCards={
              <MetricCard
                label="Runs / Steps / Tools"
                className="p-3"
                labelClassName="text-[11px]"
                valueClassName="text-sm font-medium"
                value={`${detail.summary.run_count} / ${runTotals.step_count} / ${runTotals.tool_calls_count}`}
              />
            }
          />

          <div className="grid gap-3 md:grid-cols-4">
            <MetricCard
              label="Base Agent"
              value={<MonoText>{detail.summary.base_agent_id || "-"}</MonoText>}
            />
            <MetricCard
              label="Root State"
              value={<MonoText>{detail.scheduler_state?.id || "-"}</MonoText>}
            />
            <MetricCard
              label="Scheduler Status"
              value={detail.summary.root_state_status || "Not started"}
            />
            <MetricCard
              label="Last Run"
              value={schedulerResult?.reasonLabel || (schedulerResult ? "Summary" : "-")}
            />
          </div>

          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              aria-pressed={viewMode === "mainline"}
              onClick={() => updateViewMode("mainline")}
              className={`rounded-full border px-3 py-1.5 text-sm transition-colors ${
                viewMode === "mainline"
                  ? "border-accent bg-panel-strong text-foreground"
                  : "border-line text-ink-muted hover:border-line-strong hover:text-foreground"
              }`}
            >
              Mainline
            </button>
            <button
              type="button"
              aria-pressed={viewMode === "debug"}
              onClick={() => updateViewMode("debug")}
              className={`rounded-full border px-3 py-1.5 text-sm transition-colors ${
                viewMode === "debug"
                  ? "border-accent bg-panel-strong text-foreground"
                  : "border-line text-ink-muted hover:border-line-strong hover:text-foreground"
              }`}
            >
              Debug
            </button>
          </div>

          {schedulerResult && (
            <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4 space-y-3">
              <div className="flex flex-wrap items-center gap-2 text-xs uppercase tracking-wide text-zinc-500">
                <Workflow className="h-3.5 w-3.5" />
                Scheduler Run Result
              </div>
              <div className="flex flex-wrap gap-3 text-xs text-zinc-500">
                {schedulerResult.reasonLabel && (
                  <span className="rounded-full border border-zinc-700 px-2 py-1 text-zinc-200">
                    {schedulerResult.reasonLabel}
                  </span>
                )}
                {schedulerResult.completedAt && (
                  <span>Completed {formatLocalDateTime(schedulerResult.completedAt)}</span>
                )}
                {schedulerResult.runId && (
                  <span>
                    Run <MonoText>{schedulerResult.runId}</MonoText>
                  </span>
                )}
              </div>
              {schedulerResult.error && (
                <div className="rounded border border-red-900/50 bg-red-950/30 px-3 py-2 text-sm whitespace-pre-wrap text-red-300">
                  {schedulerResult.error}
                </div>
              )}
              {schedulerResult.summary && schedulerResult.summary !== schedulerResult.error && (
                <p className="text-sm whitespace-pre-wrap text-zinc-300">
                  {schedulerResult.summary}
                </p>
              )}
            </div>
          )}

          {viewMode === "mainline" ? (
            <div className="space-y-6">
              <MilestoneBoard
                board={detail.milestone_board}
                reviewCycles={detail.review_cycles}
              />
              <ConversationEventList events={detail.conversation_events} />
            </div>
          ) : (
            <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_22rem]">
              <div className="min-w-0 space-y-3">
                {steps.length === 0 ? (
                  <EmptyStateMessage className="text-zinc-500 text-center py-8">
                    No steps found
                  </EmptyStateMessage>
                ) : (
                  <>
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <h2 className="text-sm font-medium text-zinc-300">Steps</h2>
                      <p className="text-xs text-zinc-500">
                        Showing {filteredSteps.length} of {steps.length}
                      </p>
                    </div>
                    {stepsState.hasMore && (
                      <button
                        type="button"
                        onClick={stepsState.loadEarlier}
                        disabled={stepsState.loadingMore}
                        className="rounded-md border border-zinc-800 px-3 py-1.5 text-xs text-zinc-400 hover:border-zinc-700 hover:text-zinc-200 disabled:opacity-40"
                      >
                        {stepsState.loadingMore ? "Loading..." : "Load earlier"}
                      </button>
                    )}
                  </div>
                  <div className="grid gap-3 rounded-lg border border-zinc-800 bg-zinc-900 p-3 md:grid-cols-[1fr_auto]">
                    <label className="space-y-1">
                      <span className="text-xs uppercase tracking-wide text-zinc-500">Search steps</span>
                      <input
                        value={stepSearch}
                        onChange={(event) => setStepSearch(event.target.value)}
                        placeholder="tool name, content, sequence"
                        className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-200 placeholder:text-zinc-600"
                      />
                    </label>
                    <div className="flex items-end gap-1">
                      {(["all", "user", "assistant", "tool"] as StepRoleFilter[]).map((role) => (
                        <button
                          key={role}
                          type="button"
                          aria-pressed={stepRoleFilter === role}
                          onClick={() => setStepRoleFilter(role)}
                          className={`rounded-full border px-3 py-1.5 text-xs capitalize transition-colors ${
                            stepRoleFilter === role
                              ? "border-accent bg-panel-strong text-foreground"
                              : "border-line text-ink-muted hover:border-line-strong hover:text-foreground"
                          }`}
                        >
                          {role}
                        </button>
                      ))}
                    </div>
                  </div>
                  {filteredSteps.length === 0 ? (
                    <EmptyStateMessage className="text-zinc-500 text-center py-8">
                      No steps match the current filters
                    </EmptyStateMessage>
                  ) : null}
                  {filteredSteps.map((step) => (
                    <SessionStepCard key={step.id} step={step} />
                  ))}
                  </>
                )}
              </div>

              <aside className="space-y-4 lg:sticky lg:top-4 lg:self-start">
                <SessionObservabilityPanel
                  sessionId={sessionId}
                  observability={detail.observability}
                  compact
                />

                <div className="rounded-2xl border border-line bg-panel">
                  <div className="border-b border-line bg-panel-muted px-4 py-3 text-sm font-medium text-foreground">
                    Runs
                  </div>
                  <div className="space-y-2 px-4 py-4">
                    {runTableRows.length === 0 ? (
                      <div className="rounded-xl border border-dashed border-line px-3 py-4 text-sm text-ink-muted">
                        No runs loaded.
                      </div>
                    ) : (
                      runTableRows.map(({ run, metrics }) => (
                        <div
                          key={run.id}
                          className="rounded-xl border border-line bg-panel-muted px-3 py-3"
                        >
                          <div className="flex items-center justify-between gap-3">
                            <MonoText className="truncate text-xs">{run.id.slice(0, 12)}</MonoText>
                            <span className="rounded-full border border-line px-2 py-0.5 text-[11px] text-ink-muted">
                              {run.status}
                            </span>
                          </div>
                          <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-ink-muted">
                            <span>{metrics.stepCount} steps</span>
                            <span>{metrics.toolCallsCount} tools</span>
                            <span>{formatDurationMs(metrics.durationMs)}</span>
                            <span>{formatUsd(metrics.tokenCost)}</span>
                          </div>
                          <details className="mt-2">
                            <summary className="cursor-pointer list-none text-xs text-ink-faint">
                              Token details
                            </summary>
                            <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-ink-muted">
                              <span>in {formatTokenCount(metrics.inputTokens)}</span>
                              <span>out {formatTokenCount(metrics.outputTokens)}</span>
                              <span>total {formatTokenCount(metrics.totalTokens)}</span>
                              <span>
                                cache {formatTokenCount(metrics.cacheReadTokens)} /{" "}
                                {formatTokenCount(metrics.cacheCreationTokens)}
                              </span>
                            </div>
                          </details>
                        </div>
                      ))
                    )}
                  </div>
                  <div className="border-t border-line px-3 py-3">
                    <PaginationControls
                      offset={runsOffset}
                      pageSize={runsPageSize}
                      itemCount={runs.length}
                      totalCount={runsState.total}
                      hasMore={runsState.hasMore}
                      itemLabel="runs"
                      disabled={loading}
                      onPageSizeChange={(nextPageSize) => {
                        setRunsPageSize(nextPageSize);
                        setRunsOffset(0);
                      }}
                      onPrevious={() => {
                        setRunsOffset((current) => Math.max(0, current - runsPageSize));
                      }}
                      onNext={() => {
                        setRunsOffset((current) => current + runsPageSize);
                      }}
                    />
                  </div>
                </div>
              </aside>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
