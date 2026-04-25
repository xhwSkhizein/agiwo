"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { useParams } from "next/navigation";
import { Workflow } from "lucide-react";
import { BackHeader } from "@/components/back-header";
import { MetricCard } from "@/components/metric-card";
import { PaginationControls } from "@/components/pagination-controls";
import { MonoText } from "@/components/mono-text";
import { ConversationEventList } from "@/components/session-detail/conversation-event-list";
import { MilestoneBoard } from "@/components/session-detail/milestone-board";
import { SessionObservabilityPanel } from "@/components/session-detail/session-observability-panel";
import { SessionStepCard } from "@/components/session-detail/session-step-card";
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

export default function SessionDetailPage() {
  const params = useParams();
  const sessionId = params.id as string;
  const [viewMode, setViewMode] = useState<"mainline" | "debug">("mainline");
  const [runsOffset, setRunsOffset] = useState(0);
  const [runsPageSize, setRunsPageSize] = useState(50);
  const detailState = useSessionDetailResource(sessionId);
  const runsState = useSessionRunsPage(sessionId, runsPageSize, runsOffset);
  const stepsState = useSessionStepsFeed(sessionId);
  const detail = detailState.detail;
  const steps = stepsState.steps;
  const runs = runsState.runs;
  const loading =
    detailState.loading ||
    stepsState.loading ||
    (runsState.loading && runs.length === 0);
  const error = detailState.error || runsState.error || stepsState.error;

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
              onClick={() => setViewMode("mainline")}
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
              onClick={() => setViewMode("debug")}
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
            <div className="space-y-6">
              <SessionObservabilityPanel
                sessionId={sessionId}
                observability={detail.observability}
              />

              <div className="rounded-lg border border-zinc-800 overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-zinc-900 text-zinc-500 text-xs uppercase tracking-wide">
                    <tr>
                      <th className="text-left px-4 py-2.5">Run</th>
                      <th className="text-right px-4 py-2.5">Cost</th>
                      <th className="text-right px-4 py-2.5">Input</th>
                      <th className="text-right px-4 py-2.5">Output</th>
                      <th className="text-right px-4 py-2.5">Total</th>
                      <th className="text-right px-4 py-2.5">Cache R/C</th>
                      <th className="text-right px-4 py-2.5">Steps/Tools</th>
                      <th className="text-right px-4 py-2.5">Duration</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-800">
                    {runTableRows.map(({ run, metrics }) => {
                      return (
                        <tr key={run.id} className="hover:bg-zinc-900/50">
                          <td className="px-4 py-2.5 text-zinc-300">
                            <MonoText className="text-xs font-mono text-zinc-300">
                              {run.id}
                            </MonoText>
                          </td>
                          <td className="px-4 py-2.5 text-right text-zinc-200">
                            {formatUsd(metrics.tokenCost)}
                          </td>
                          <td className="px-4 py-2.5 text-right text-zinc-400">
                            {formatTokenCount(metrics.inputTokens)}
                          </td>
                          <td className="px-4 py-2.5 text-right text-zinc-400">
                            {formatTokenCount(metrics.outputTokens)}
                          </td>
                          <td className="px-4 py-2.5 text-right text-zinc-400">
                            {formatTokenCount(metrics.totalTokens)}
                          </td>
                          <td className="px-4 py-2.5 text-right text-zinc-500">
                            {formatTokenCount(metrics.cacheReadTokens)} / {formatTokenCount(metrics.cacheCreationTokens)}
                          </td>
                          <td className="px-4 py-2.5 text-right text-zinc-500">
                            {metrics.stepCount} / {metrics.toolCallsCount}
                          </td>
                          <td className="px-4 py-2.5 text-right text-zinc-500">
                            {formatDurationMs(metrics.durationMs)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

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

              {steps.length === 0 ? (
                <EmptyStateMessage className="text-zinc-500 text-center py-8">
                  No steps found
                </EmptyStateMessage>
              ) : (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <h2 className="text-sm font-medium text-zinc-300">Steps</h2>
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
                  {steps.map((step) => (
                    <SessionStepCard key={step.id} step={step} />
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
