"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { ArrowLeft, Clock, Zap, Cpu, Wrench } from "lucide-react";
import { getTrace } from "@/lib/api";
import type { TraceDetail, SpanResponse } from "@/lib/api";

function StatusBadge({ status }: { status: string }) {
  const cls =
    status === "ok"
      ? "bg-green-900/50 text-green-400"
      : status === "error"
      ? "bg-red-900/50 text-red-400"
      : status === "running"
      ? "bg-blue-900/50 text-blue-400"
      : "bg-zinc-800 text-zinc-400";
  return (
    <span className={`text-xs px-1.5 py-0.5 rounded ${cls}`}>{status}</span>
  );
}

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
          <StatusBadge status={span.status} />
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
              <span className="font-mono">{span.span_id.slice(0, 12)}</span>
            </div>
            <div>
              <span className="text-zinc-500">Kind: </span>
              <span>{span.kind}</span>
            </div>
            {span.duration_ms != null && (
              <div>
                <span className="text-zinc-500">Duration: </span>
                <span>{Math.round(span.duration_ms)}ms</span>
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
  const [expandedSpans, setExpandedSpans] = useState<Set<string>>(new Set());

  useEffect(() => {
    getTrace(traceId)
      .then(setTrace)
      .catch(() => setTrace(null))
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
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-zinc-500">Loading trace...</div>
      </div>
    );
  }

  if (!trace) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-zinc-500">Trace not found</div>
      </div>
    );
  }

  const traceStartMs = trace.start_time
    ? new Date(trace.start_time).getTime()
    : 0;
  const traceDurationMs = trace.duration_ms || 1;

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div className="flex items-center gap-3">
        <Link
          href="/traces"
          className="p-1.5 rounded hover:bg-zinc-800 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
        </Link>
        <div>
          <h1 className="text-xl font-semibold">Trace Detail</h1>
          <p className="text-xs text-zinc-500 font-mono mt-0.5">
            {trace.trace_id}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <p className="text-xs text-zinc-500">Status</p>
          <div className="mt-1">
            <StatusBadge status={trace.status} />
          </div>
        </div>
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <p className="text-xs text-zinc-500">Duration</p>
          <p className="text-lg font-medium mt-1">
            {trace.duration_ms ? `${Math.round(trace.duration_ms)}ms` : "-"}
          </p>
        </div>
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <p className="text-xs text-zinc-500">Total Tokens</p>
          <p className="text-lg font-medium mt-1">
            {trace.total_tokens.toLocaleString()}
          </p>
        </div>
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <p className="text-xs text-zinc-500">LLM / Tool Calls</p>
          <p className="text-lg font-medium mt-1">
            {trace.total_llm_calls} / {trace.total_tool_calls}
          </p>
        </div>
      </div>

      {trace.input_query && (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <p className="text-xs text-zinc-500 mb-1">Input</p>
          <p className="text-sm">{trace.input_query}</p>
        </div>
      )}

      <div className="rounded-lg border border-zinc-800 overflow-hidden">
        <div className="px-4 py-3 bg-zinc-900 border-b border-zinc-800">
          <h2 className="text-sm font-medium">
            Span Waterfall ({trace.spans.length} spans)
          </h2>
        </div>
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
      </div>

      {trace.final_output && (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <p className="text-xs text-zinc-500 mb-1">Final Output</p>
          <p className="text-sm whitespace-pre-wrap">{trace.final_output}</p>
        </div>
      )}
    </div>
  );
}
