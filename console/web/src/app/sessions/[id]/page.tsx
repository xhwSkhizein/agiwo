"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { Bot, ChevronDown, ChevronRight, User, Wrench, Workflow } from "lucide-react";
import { BackHeader } from "@/components/back-header";
import { MetricCard } from "@/components/metric-card";
import { PaginationControls } from "@/components/pagination-controls";
import { MonoText } from "@/components/mono-text";
import {
  EmptyStateMessage,
  ErrorStateMessage,
  TextStateMessage,
} from "@/components/state-message";
import { TokenSummaryCards } from "@/components/token-summary-cards";
import { TokenMetricsBadges } from "@/components/token-metrics-badges";
import { UserInputDetail } from "@/components/user-input-detail";
import {
  getSessionDetail,
  getSessionSteps,
  listRuns,
} from "@/lib/api";
import type {
  RunResponse,
  SessionDetail,
  StepResponse,
  ToolCallPayload,
} from "@/lib/api";
import {
  formatDurationMs,
  formatTokenCount,
  formatUsd,
  normalizeRunMetricsSummary,
  parseGenericMetrics,
} from "@/lib/metrics";

function StepCardOriginalToggle({ content }: { content: string }) {
  const [expanded, setExpanded] = useState(false);
  const toggle = useCallback(() => setExpanded((prev) => !prev), []);
  return (
    <div className="mt-1">
      <button
        type="button"
        onClick={toggle}
        className="inline-flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
      >
        {expanded ? (
          <ChevronDown className="w-3 h-3" />
        ) : (
          <ChevronRight className="w-3 h-3" />
        )}
        {expanded ? "Hide original result" : "View original result"}
      </button>
      {expanded && (
        <div className="mt-1 px-3 py-2 rounded bg-zinc-800/50 text-xs text-zinc-400 whitespace-pre-wrap max-h-64 overflow-auto">
          {content}
        </div>
      )}
    </div>
  );
}

function StepCard({ step }: { step: StepResponse }) {
  const isUser = step.role === "user";
  const isAssistant = step.role === "assistant";
  const isTool = step.role === "tool";
  const metrics = parseGenericMetrics(step.metrics ?? undefined);

  const getToolLabel = (toolCall: ToolCallPayload): string =>
    toolCall.function?.name || "tool_call";
  const getToolArgs = (toolCall: ToolCallPayload): string =>
    toolCall.function?.arguments || JSON.stringify(toolCall);

  const hasStructuredUserInput =
    isUser && step.user_input !== null && step.user_input !== undefined;
  const hasCondensed = isTool && typeof step.condensed_content === "string";
  const displayContent = hasCondensed ? step.condensed_content : step.content;
  const originalContent =
    hasCondensed && typeof step.content === "string" ? step.content : null;

  return (
    <div
      className={`rounded-lg border p-4 ${
        isUser
          ? "border-blue-800/50 bg-blue-950/20"
          : isTool
            ? "border-amber-800/50 bg-amber-950/20"
            : "border-zinc-800 bg-zinc-900"
      }`}
    >
      <div className="flex items-center gap-2 mb-2">
        {isUser && <User className="w-4 h-4 text-blue-400" />}
        {isAssistant && <Bot className="w-4 h-4 text-green-400" />}
        {isTool && <Wrench className="w-4 h-4 text-amber-400" />}
        <span className="text-xs font-medium uppercase tracking-wide text-zinc-400">
          {step.role}
          {isTool && step.name && ` — ${step.name}`}
          {step.agent_id && ` — ${step.agent_id}`}
        </span>
        <span className="text-xs text-zinc-600 ml-auto">#{step.sequence}</span>
      </div>

      {step.reasoning_content && (
        <div className="mb-2 px-3 py-2 rounded bg-zinc-800/50 text-xs text-zinc-400 whitespace-pre-wrap max-h-48 overflow-auto">
          <span className="text-zinc-500 font-medium">Thinking: </span>
          {step.reasoning_content}
        </div>
      )}

      {hasStructuredUserInput && (
        <div className="max-h-96 overflow-auto">
          <UserInputDetail input={step.user_input} maxTextLength={2000} />
        </div>
      )}

      {!hasStructuredUserInput && Boolean(displayContent) && (
        <div className="max-h-96 overflow-auto">
          <div className="text-sm whitespace-pre-wrap break-words">
            {typeof displayContent === "string"
              ? displayContent
              : JSON.stringify(displayContent, null, 2)}
          </div>
        </div>
      )}

      {originalContent && (
        <StepCardOriginalToggle content={originalContent} />
      )}

      {step.tool_calls && step.tool_calls.length > 0 && (
        <div className="mt-2 space-y-1">
          {step.tool_calls.map((tc, i) => (
            <div
              key={i}
              className="text-xs bg-zinc-800/50 rounded px-3 py-2 font-mono overflow-auto max-h-48"
            >
              <span className="text-amber-400">{getToolLabel(tc)}</span>
              <span className="text-zinc-500 ml-2">{getToolArgs(tc)}</span>
            </div>
          ))}
        </div>
      )}

      {step.metrics && (
        <div className="mt-3">
          <TokenMetricsBadges
            metrics={metrics}
            showDuration={true}
            showModelName={true}
            modelName={step.metrics?.model_name ?? null}
          />
        </div>
      )}
    </div>
  );
}

export default function SessionDetailPage() {
  const params = useParams();
  const sessionId = params.id as string;
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [steps, setSteps] = useState<StepResponse[]>([]);
  const [runs, setRuns] = useState<RunResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [runsOffset, setRunsOffset] = useState(0);
  const [runsPageSize, setRunsPageSize] = useState(50);
  const [runsHasMore, setRunsHasMore] = useState(false);
  const [runsTotal, setRunsTotal] = useState<number | null>(null);
  const [stepsHasMore, setStepsHasMore] = useState(false);
  const [loadingMoreSteps, setLoadingMoreSteps] = useState(false);

  useEffect(() => {
    setLoading(true);
    setError(null);
    Promise.all([
      getSessionDetail(sessionId),
      listRuns({ session_id: sessionId, limit: runsPageSize, offset: runsOffset }),
      getSessionSteps(sessionId, { limit: 100, order: "desc" }),
    ])
      .then(([nextDetail, sessionRuns, sessionSteps]) => {
        setDetail(nextDetail);
        setRuns(sessionRuns.items);
        setRunsHasMore(sessionRuns.has_more);
        setRunsTotal(sessionRuns.total);
        setSteps([...sessionSteps.items].reverse());
        setStepsHasMore(sessionSteps.has_more);
      })
      .catch((err) => {
        setDetail(null);
        setRuns([]);
        setSteps([]);
        setRunsHasMore(false);
        setStepsHasMore(false);
        setError(err instanceof Error ? err.message : "Failed to load session");
      })
      .finally(() => setLoading(false));
  }, [runsOffset, runsPageSize, sessionId]);

  async function loadEarlierSteps() {
    if (steps.length === 0 || loadingMoreSteps) {
      return;
    }
    const oldestSequence = steps[0]?.sequence;
    if (!oldestSequence || oldestSequence <= 1) {
      setStepsHasMore(false);
      return;
    }
    setLoadingMoreSteps(true);
    try {
      const nextPage = await getSessionSteps(sessionId, {
        limit: 100,
        order: "desc",
        end_seq: oldestSequence - 1,
      });
      setSteps((current) => [...nextPage.items.reverse(), ...current]);
      setStepsHasMore(nextPage.has_more);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load older steps");
    } finally {
      setLoadingMoreSteps(false);
    }
  }

  const runTotals = normalizeRunMetricsSummary(detail?.summary.metrics);

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

          <div className="grid gap-3 md:grid-cols-3">
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
          </div>

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
                {runs.map((run) => {
                  const metrics = parseGenericMetrics(run.metrics ?? undefined);
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
            totalCount={runsTotal}
            hasMore={runsHasMore}
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
                {stepsHasMore && (
                  <button
                    type="button"
                    onClick={loadEarlierSteps}
                    disabled={loadingMoreSteps}
                    className="rounded-md border border-zinc-800 px-3 py-1.5 text-xs text-zinc-400 hover:border-zinc-700 hover:text-zinc-200 disabled:opacity-40"
                  >
                    {loadingMoreSteps ? "Loading..." : "Load earlier"}
                  </button>
                )}
              </div>
              {steps.map((step) => (
                <StepCard key={step.id} step={step} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
