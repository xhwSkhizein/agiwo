"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { User, Bot, Wrench } from "lucide-react";
import { BackHeader } from "@/components/back-header";
import { getSessionSteps, getSessionSummary, listRuns } from "@/lib/api";
import type { RunResponse, SessionSummary, StepResponse, ToolCallPayload } from "@/lib/api";
import { MetricCard } from "@/components/metric-card";
import { MonoText } from "@/components/mono-text";
import { EmptyStateMessage, TextStateMessage } from "@/components/state-message";
import { TokenSummaryCards } from "@/components/token-summary-cards";
import { TokenMetricsBadges } from "@/components/token-metrics-badges";
import { UserInputDetail } from "@/components/user-input-detail";
import {
  formatDurationMs,
  formatTokenCount,
  formatUsd,
  normalizeRunMetricsSummary,
  parseGenericMetrics,
} from "@/lib/metrics";

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
  const content = step.content;

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

      {!hasStructuredUserInput && Boolean(content) && (
        <div className="max-h-96 overflow-auto">
          {(
            <div className="text-sm whitespace-pre-wrap break-words">
              {typeof content === "string"
                ? content
                : JSON.stringify(content, null, 2)}
            </div>
          )}
        </div>
      )}

      {step.tool_calls && step.tool_calls.length > 0 && (
        <div className="mt-2 space-y-1">
          {step.tool_calls.map((tc, i) => (
            <div
              key={i}
              className="text-xs bg-zinc-800/50 rounded px-3 py-2 font-mono overflow-auto max-h-48"
            >
              <span className="text-amber-400">{getToolLabel(tc)}</span>
              <span className="text-zinc-500 ml-2">
                {getToolArgs(tc)}
              </span>
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
  const [steps, setSteps] = useState<StepResponse[]>([]);
  const [runs, setRuns] = useState<RunResponse[]>([]);
  const [summary, setSummary] = useState<SessionSummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      getSessionSteps(sessionId).catch(() => []),
      getSessionSummary(sessionId).catch(() => null),
      listRuns({ session_id: sessionId, limit: 200 }).catch(() => []),
    ])
      .then(([sessionSteps, sessionSummary, sessionRuns]) => {
        setSteps(sessionSteps);
        setSummary(sessionSummary);
        setRuns(sessionRuns);
      })
      .finally(() => setLoading(false));
  }, [sessionId]);

  const runTotals = normalizeRunMetricsSummary(summary?.metrics);

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <BackHeader
        href="/sessions"
        title="Session Detail"
        subtitle={sessionId}
      />

      {loading ? (
        <TextStateMessage>Loading session metrics...</TextStateMessage>
      ) : (
        <div className="space-y-6">
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
                value={`${summary?.run_count ?? 0} / ${runTotals.step_count} / ${runTotals.tool_calls_count}`}
              />
            }
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
                {runs.map((run) => {
                  const metrics = parseGenericMetrics(
                    run.metrics ?? undefined
                  );
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

          {steps.length === 0 ? (
            <EmptyStateMessage className="text-zinc-500 text-center py-8">
              No steps found
            </EmptyStateMessage>
          ) : (
            <div className="space-y-3">
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
