"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { Workflow } from "lucide-react";
import { BackHeader } from "@/components/back-header";
import { MetricCard } from "@/components/metric-card";
import { MonoText } from "@/components/mono-text";
import { ConversationEventList } from "@/components/session-detail/conversation-event-list";
import { MilestoneBoard } from "@/components/session-detail/milestone-board";
import { RunsStepsPanel } from "@/components/session-detail/runs-steps-panel";
import { SessionObservabilityPanel } from "@/components/session-detail/session-observability-panel";
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
  formatTokenCount,
  formatUsd,
  normalizeRunMetricsSummary,
} from "@/lib/metrics";
import { getSchedulerRunResultView } from "@/lib/scheduler-run-result";
import { formatLocalDateTime } from "@/lib/time";

export default function SessionDetailPage() {
  const params = useParams();
  const router = useRouter();
  const searchParams = useSearchParams();
  const sessionId = params.id as string;
  const [showExtras, setShowExtras] = useState(
    searchParams.get("view") === "debug",
  );
  const [runsOffset, setRunsOffset] = useState(0);
  const [runsPageSize, setRunsPageSize] = useState(50);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

  const detailState = useSessionDetailResource(sessionId);
  const runsState = useSessionRunsPage(
    sessionId,
    runsPageSize,
    runsOffset,
    true,
  );
  const detail = detailState.detail;
  const runs = runsState.runs;

  const preferredRunId = useMemo(() => {
    if (runs.length === 0) {
      return null;
    }
    return (
      runs.find((run) => !run.parent_run_id)?.id ||
      runs[0]?.id ||
      null
    );
  }, [runs]);

  const effectiveRunId =
    selectedRunId && runs.some((run) => run.id === selectedRunId)
      ? selectedRunId
      : preferredRunId;

  const stepsState = useSessionStepsFeed(sessionId, true, effectiveRunId);

  const loading =
    detailState.loading || (runsState.loading && runs.length === 0);
  const error = detailState.error || runsState.error;

  const runTotals = normalizeRunMetricsSummary(detail?.summary.metrics);
  const schedulerResult = getSchedulerRunResultView(
    detail?.scheduler_state?.last_run_result,
    detail?.scheduler_state?.result_summary,
  );

  const updateExtras = (next: boolean) => {
    setShowExtras(next);
    const paramsNext = new URLSearchParams(searchParams.toString());
    if (next) {
      paramsNext.set("view", "debug");
    } else {
      paramsNext.delete("view");
    }
    router.replace(
      `/sessions/${sessionId}${paramsNext.toString() ? `?${paramsNext}` : ""}`,
    );
  };

  return (
    <div className="mx-auto max-w-7xl space-y-5 p-6">
      <BackHeader
        href="/sessions"
        title="Session Detail"
        subtitle={sessionId}
      />

      {loading ? (
        <TextStateMessage>Loading session…</TextStateMessage>
      ) : error ? (
        <ErrorStateMessage>{error}</ErrorStateMessage>
      ) : !detail ? (
        <EmptyStateMessage>Session not found</EmptyStateMessage>
      ) : (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-2 text-xs text-ink-muted">
            {detail.scheduler_state && (
              <>
                <Link
                  href={`/scheduler/${detail.scheduler_state.id}`}
                  className="inline-flex items-center gap-1 rounded border border-line px-2 py-1 hover:border-line-strong hover:text-foreground"
                >
                  <Workflow className="h-3 w-3" />
                  Scheduler State
                </Link>
                <Link
                  href={`/scheduler/${
                    detail.scheduler_state.root_state_id ??
                    detail.scheduler_state.id
                  }/tree`}
                  className="inline-flex items-center gap-1 rounded border border-line px-2 py-1 hover:border-line-strong hover:text-foreground"
                >
                  <Workflow className="h-3 w-3" />
                  Scheduler Tree
                </Link>
              </>
            )}
            <Link
              href={`/traces?session_id=${sessionId}`}
              className="rounded border border-line px-2 py-1 hover:border-line-strong hover:text-foreground"
            >
              Related Traces
            </Link>
            <button
              type="button"
              aria-pressed={showExtras}
              onClick={() => updateExtras(!showExtras)}
              className={`ml-auto rounded border px-2 py-1 ${
                showExtras
                  ? "border-accent bg-panel-strong text-foreground"
                  : "border-line text-ink-muted hover:text-foreground"
              }`}
            >
              {showExtras ? "Hide debug extras" : "Show debug extras"}
            </button>
          </div>

          <div className="rounded-xl border border-line bg-panel px-4 py-3">
            <div className="grid grid-cols-2 gap-3 text-xs text-ink-muted sm:grid-cols-4">
              <span>Cost {formatUsd(runTotals.token_cost)}</span>
              <span>Tokens {formatTokenCount(runTotals.total_tokens)}</span>
              <span>
                Runs {detail.summary.run_count} / Steps {runTotals.step_count}
              </span>
              <span>Status {detail.summary.root_state_status || "—"}</span>
            </div>
            <div className="mt-2 text-xs text-ink-faint">
              Agent{" "}
              <MonoText className="text-[11px]">
                {detail.summary.base_agent_id || "-"}
              </MonoText>
            </div>
          </div>

          <RunsStepsPanel
            sessionId={sessionId}
            runs={runs}
            selectedRunId={effectiveRunId}
            onSelectRunId={setSelectedRunId}
            steps={stepsState.steps}
            stepsLoading={stepsState.loading}
            stepsError={stepsState.error}
            hasMoreSteps={stepsState.hasMore}
            loadingMoreSteps={stepsState.loadingMore}
            onLoadEarlierSteps={() => void stepsState.loadEarlier()}
            runsOffset={runsOffset}
            runsPageSize={runsPageSize}
            runsTotal={runsState.total}
            runsHasMore={runsState.hasMore}
            runsLoading={runsState.loading}
            onRunsPageSizeChange={(size) => {
              setRunsPageSize(size);
              setRunsOffset(0);
            }}
            onRunsPrevious={() =>
              setRunsOffset((current) => Math.max(0, current - runsPageSize))
            }
            onRunsNext={() => setRunsOffset((current) => current + runsPageSize)}
          />

          {schedulerResult && (
            <div className="space-y-2 rounded-lg border border-line bg-panel p-4">
              <div className="flex flex-wrap items-center gap-2 text-xs uppercase tracking-wide text-ink-faint">
                <Workflow className="h-3.5 w-3.5" />
                Scheduler Run Result
              </div>
              <div className="flex flex-wrap gap-3 text-xs text-ink-muted">
                {schedulerResult.reasonLabel && (
                  <span className="rounded-full border border-line px-2 py-1 text-foreground">
                    {schedulerResult.reasonLabel}
                  </span>
                )}
                {schedulerResult.completedAt && (
                  <span>
                    Completed {formatLocalDateTime(schedulerResult.completedAt)}
                  </span>
                )}
                {schedulerResult.runId && (
                  <span>
                    Run <MonoText>{schedulerResult.runId}</MonoText>
                  </span>
                )}
              </div>
              {schedulerResult.error && (
                <div className="whitespace-pre-wrap rounded border border-red-900/50 bg-red-950/30 px-3 py-2 text-sm text-red-300">
                  {schedulerResult.error}
                </div>
              )}
              {schedulerResult.summary &&
                schedulerResult.summary !== schedulerResult.error && (
                  <p className="whitespace-pre-wrap text-sm text-ink-muted">
                    {schedulerResult.summary}
                  </p>
                )}
            </div>
          )}

          {showExtras ? (
            <div className="space-y-4 rounded-xl border border-dashed border-line p-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 className="text-sm font-medium text-foreground">
                  Debug extras
                </h3>
                <TokenSummaryCards
                  cost={runTotals.token_cost}
                  inputTokens={runTotals.input_tokens}
                  outputTokens={runTotals.output_tokens}
                  totalTokens={runTotals.total_tokens}
                  cacheReadTokens={runTotals.cache_read_tokens}
                  cacheCreationTokens={runTotals.cache_creation_tokens}
                  className="grid grid-cols-2 gap-2 md:grid-cols-5"
                  cardClassName="p-2"
                  labelClassName="text-[10px]"
                  valueClassName="text-xs font-medium"
                  extraCardsPosition="after"
                  extraCards={
                    <MetricCard
                      label="Tools"
                      className="p-2"
                      labelClassName="text-[10px]"
                      valueClassName="text-xs font-medium"
                      value={String(runTotals.tool_calls_count)}
                    />
                  }
                />
              </div>
              <MilestoneBoard
                board={detail.milestone_board}
                reviewCycles={detail.review_cycles}
              />
              <ConversationEventList events={detail.conversation_events} />
              <SessionObservabilityPanel
                sessionId={sessionId}
                observability={detail.observability}
                compact
              />
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
}
