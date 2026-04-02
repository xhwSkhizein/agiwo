"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { Clock, Zap, Cpu, Wrench } from "lucide-react";
import { BackHeader } from "@/components/back-header";
import { MetricCard } from "@/components/metric-card";
import { MonoText } from "@/components/mono-text";
import { SectionCard } from "@/components/section-card";
import { ErrorStateMessage, FullPageMessage } from "@/components/state-message";
import { TokenMetricsBadges } from "@/components/token-metrics-badges";
import { TokenSummaryCards } from "@/components/token-summary-cards";
import { TraceStatusBadge } from "@/components/trace-status-badge";
import { getTrace } from "@/lib/api";
import type { TraceDetail, SpanResponse } from "@/lib/api";
import {
  formatDurationMs,
  parseGenericMetrics,
} from "@/lib/metrics";
import { formatRoundedMs } from "@/lib/time";

function KindIcon({ kind }: { kind: string }) {
  if (kind === "agent") return <Cpu className="w-3.5 h-3.5 text-purple-400" />;
  if (kind === "llm_call") return <Zap className="w-3.5 h-3.5 text-blue-400" />;
  if (kind === "tool_call") return <Wrench className="w-3.5 h-3.5 text-amber-400" />;
  return <Clock className="w-3.5 h-3.5 text-zinc-400" />;
}

function SpanRow({
  span,
  traceStartMs,
  traceDurationMs,
  expanded,
  onToggle,
}: {
  span: SpanResponse;
  traceStartMs: number;
  traceDurationMs: number;
  expanded: boolean;
  onToggle: () => void;
}) {
  const spanMetrics = parseGenericMetrics(span.metrics);
  const spanStartMs = span.start_time
    ? new Date(span.start_time).getTime() - traceStartMs
    : 0;
  const spanDuration = span.duration_ms || 0;
  const leftPct = traceDurationMs > 0 ? (spanStartMs / traceDurationMs) * 100 : 0;
  const widthPct =
    traceDurationMs > 0
      ? Math.max((spanDuration / traceDurationMs) * 100, 0.5)
      : 100;

  const barColor =
    span.kind === "agent"
      ? "bg-purple-500"
      : span.kind === "llm_call"
      ? "bg-blue-500"
      : span.kind === "tool_call"
      ? "bg-amber-500"
      : "bg-zinc-500";

  return (
    <>
      <div
        className="flex items-center gap-2 px-4 py-2.5 hover:bg-zinc-800/50 cursor-pointer transition-colors border-b border-zinc-800/50"
        onClick={onToggle}
      >
        <div
          className="flex items-center gap-2 shrink-0"
          style={{ paddingLeft: `${span.depth * 20}px`, width: "280px" }}
        >
          <KindIcon kind={span.kind} />
          <span className="text-sm truncate">{span.name}</span>
          <TraceStatusBadge status={span.status} />
        </div>

        <div className="flex-1 h-5 relative bg-zinc-800/30 rounded overflow-hidden">
          <div
            className={`absolute h-full rounded ${barColor} opacity-80`}
            style={{
              left: `${Math.min(leftPct, 99)}%`,
              width: `${Math.min(widthPct, 100 - leftPct)}%`,
            }}
          />
        </div>

        <div className="w-20 text-right text-xs text-zinc-400 shrink-0">
          {spanDuration ? `${Math.round(spanDuration)}ms` : "-"}
        </div>
      </div>

      {expanded && (
        <div className="px-4 py-3 bg-zinc-900/50 border-b border-zinc-800 text-xs space-y-2">
          <div className="grid grid-cols-2 gap-x-6 gap-y-1">
            <div>
              <span className="text-zinc-500">Span ID: </span>
              <MonoText className="font-mono">
                {span.span_id.slice(0, 12)}
              </MonoText>
            </div>
            <div>
              <span className="text-zinc-500">Kind: </span>
              <span>{span.kind}</span>
            </div>
            {span.duration_ms != null && (
              <div>
                <span className="text-zinc-500">Duration: </span>
                <span>{formatRoundedMs(span.duration_ms)}</span>
              </div>
            )}
            {span.error_message && (
              <div className="col-span-2">
                <span className="text-red-400">Error: </span>
                <span className="text-red-300">{span.error_message}</span>
              </div>
            )}
          </div>

          {Object.keys(span.metrics).length > 0 && (
            <div>
              <TokenMetricsBadges
                metrics={spanMetrics}
                showCacheRead={false}
                showCacheCreation={false}
                chipClassName="bg-zinc-800"
                className="mb-2 grid grid-cols-2 sm:grid-cols-4 gap-2"
              />
              <p className="text-zinc-500 mb-1">Metrics:</p>
              <div className="bg-zinc-800/50 rounded px-3 py-2 font-mono overflow-auto max-h-32">
                {JSON.stringify(span.metrics, null, 2)}
              </div>
            </div>
          )}

          {span.llm_details && (
            <div>
              <p className="text-zinc-500 mb-1">LLM Details:</p>
              <div className="bg-zinc-800/50 rounded px-3 py-2 font-mono overflow-auto max-h-64 whitespace-pre-wrap">
                {JSON.stringify(span.llm_details, null, 2)}
              </div>
            </div>
          )}

          {span.tool_details && (
            <div>
              <p className="text-zinc-500 mb-1">Tool Details:</p>
              <div className="bg-zinc-800/50 rounded px-3 py-2 font-mono overflow-auto max-h-64 whitespace-pre-wrap">
                {JSON.stringify(span.tool_details, null, 2)}
              </div>
            </div>
          )}

          {span.output_preview && (
            <div>
              <p className="text-zinc-500 mb-1">Output Preview:</p>
              <div className="bg-zinc-800/50 rounded px-3 py-2 whitespace-pre-wrap max-h-48 overflow-auto">
                {span.output_preview}
              </div>
            </div>
          )}
        </div>
      )}
    </>
  );
}

export default function TraceDetailPage() {
  const params = useParams();
  const traceId = params.id as string;
  const [trace, setTrace] = useState<TraceDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedSpans, setExpandedSpans] = useState<Set<string>>(new Set());

  useEffect(() => {
    getTrace(traceId)
      .then((value) => {
        setTrace(value);
        setError(null);
      })
      .catch((err) => {
        setTrace(null);
        setError(err instanceof Error ? err.message : "Failed to load trace");
      })
      .finally(() => setLoading(false));
  }, [traceId]);

  const toggleSpan = (spanId: string) => {
    setExpandedSpans((prev) => {
      const next = new Set(prev);
      if (next.has(spanId)) next.delete(spanId);
      else next.add(spanId);
      return next;
    });
  };

  if (loading) {
    return <FullPageMessage>Loading trace...</FullPageMessage>;
  }

  if (!trace) {
    return <FullPageMessage>Trace not found</FullPageMessage>;
  }

  const traceStartMs = trace.start_time
    ? new Date(trace.start_time).getTime()
    : 0;
  const traceDurationMs = trace.duration_ms || 1;

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <BackHeader
        href="/traces"
        title="Trace Detail"
        subtitle={trace.trace_id}
      />

      {error && <ErrorStateMessage>{error}</ErrorStateMessage>}

      <div className="flex flex-wrap gap-2 text-xs text-zinc-500">
        {trace.session_id && (
          <Link
            href={`/sessions/${trace.session_id}`}
            className="rounded border border-zinc-800 px-2 py-1 hover:border-zinc-700 hover:text-zinc-300"
          >
            Open session
          </Link>
        )}
        {trace.agent_id && (
          <Link
            href={`/agents/${trace.agent_id}`}
            className="rounded border border-zinc-800 px-2 py-1 hover:border-zinc-700 hover:text-zinc-300"
          >
            Open agent
          </Link>
        )}
      </div>

      <TokenSummaryCards
        cost={trace.total_token_cost}
        costLabel="Total Cost"
        inputTokens={trace.total_input_tokens}
        outputTokens={trace.total_output_tokens}
        totalTokens={trace.total_tokens}
        cacheReadTokens={trace.total_cache_read_tokens}
        cacheCreationTokens={trace.total_cache_creation_tokens}
        extraCards={
          <>
            <MetricCard
              label="Status"
              valueClassName="text-lg font-medium"
              value={<TraceStatusBadge status={trace.status} />}
            />
            <MetricCard
              label="Duration"
              valueClassName="text-lg font-medium"
              value={formatDurationMs(trace.duration_ms || 0)}
            />
            <MetricCard
              label="LLM / Tool"
              valueClassName="text-lg font-medium"
              value={`${trace.total_llm_calls} / ${trace.total_tool_calls}`}
            />
          </>
        }
      />

      {trace.input_query && (
        <SectionCard className="p-4">
          <p className="text-xs text-zinc-500 mb-1">Input</p>
          <p className="text-sm">{trace.input_query}</p>
        </SectionCard>
      )}

      <SectionCard
        className="overflow-hidden"
        title={`Span Waterfall (${trace.spans.length} spans)`}
        headerClassName="px-4 py-3 border-b border-zinc-800"
      >
        <div>
          {trace.spans.map((span) => (
            <SpanRow
              key={span.span_id}
              span={span}
              traceStartMs={traceStartMs}
              traceDurationMs={traceDurationMs}
              expanded={expandedSpans.has(span.span_id)}
              onToggle={() => toggleSpan(span.span_id)}
            />
          ))}
        </div>
      </SectionCard>

      {trace.final_output && (
        <SectionCard className="p-4">
          <p className="text-xs text-zinc-500 mb-1">Final Output</p>
          <p className="text-sm whitespace-pre-wrap">{trace.final_output}</p>
        </SectionCard>
      )}
    </div>
  );
}
