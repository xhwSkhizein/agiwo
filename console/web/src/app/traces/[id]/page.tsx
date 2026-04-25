"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { Clock, Zap, Cpu, Wrench } from "lucide-react";
import { BackHeader } from "@/components/back-header";
import { JsonDisclosure } from "@/components/json-disclosure";
import { MetricCard } from "@/components/metric-card";
import { MonoText } from "@/components/mono-text";
import { SectionCard } from "@/components/section-card";
import { ErrorStateMessage, FullPageMessage } from "@/components/state-message";
import { TokenMetricsBadges } from "@/components/token-metrics-badges";
import { TokenSummaryCards } from "@/components/token-summary-cards";
import { TraceLlmCalls } from "@/components/trace-detail/trace-llm-calls";
import { TraceLoopTimeline } from "@/components/trace-detail/trace-loop-timeline";
import { TraceMainlineEvents } from "@/components/trace-detail/trace-mainline-events";
import { TraceReviewCycles } from "@/components/trace-detail/trace-review-cycles";
import { TraceRuntimeDecisions } from "@/components/trace-detail/trace-runtime-decisions";
import { TraceStatusBadge } from "@/components/trace-status-badge";
import { getTrace } from "@/lib/api";
import type { TraceDetail, SpanResponse } from "@/lib/api";
import {
  formatDurationMs,
  parseGenericMetrics,
} from "@/lib/metrics";
import { formatRoundedMs } from "@/lib/time";

/**
 * Render a small Lucide icon representing the span `kind`.
 *
 * @param kind - The span kind; expected values: `"agent"`, `"llm_call"`, `"tool_call"`. Any other value renders the default icon.
 * @returns A JSX element for the corresponding icon: `Cpu` for `"agent"`, `Zap` for `"llm_call"`, `Wrench` for `"tool_call"`, and `Clock` for other kinds. Icons are rendered with the component's standard sizing and semantic color classes.
 */
function KindIcon({ kind }: { kind: string }) {
  if (kind === "agent") return <Cpu className="h-3.5 w-3.5 text-accent" />;
  if (kind === "llm_call") return <Zap className="h-3.5 w-3.5 text-success" />;
  if (kind === "tool_call") return <Wrench className="h-3.5 w-3.5 text-warning" />;
  return <Clock className="h-3.5 w-3.5 text-ink-faint" />;
}

/**
 * Render a clickable span row in the trace waterfall with an inline duration bar and optional expandable details.
 *
 * The row is indented by `span.depth`, positions and sizes the bar using `traceStartMs` and `traceDurationMs`,
 * and when expanded displays span metadata, error message, metrics/LLM/tool JSON disclosures, and an output preview.
 *
 * @param span - The span object to display
 * @param traceStartMs - Trace start time in milliseconds since epoch; used to compute the span's offset within the trace
 * @param traceDurationMs - Total trace duration in milliseconds; used to compute the span bar's width and left offset
 * @param expanded - Whether the span's details panel is currently expanded
 * @param onToggle - Callback invoked when the row's expand/collapse button is clicked
 * @returns The rendered span row element
 */
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
      ? "bg-accent"
      : span.kind === "llm_call"
      ? "bg-success"
      : span.kind === "tool_call"
      ? "bg-warning"
      : "bg-line-strong";

  return (
    <>
      <button
        type="button"
        aria-expanded={expanded}
        aria-controls={`${span.span_id}-details`}
        className="flex w-full items-center gap-2 border-b border-line px-4 py-2.5 text-left transition-colors hover:bg-panel-muted"
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

        <div className="relative h-5 flex-1 overflow-hidden rounded bg-panel-muted">
          <div
            className={`absolute h-full rounded ${barColor} opacity-80`}
            style={{
              left: `${Math.min(leftPct, 99)}%`,
              width: `${Math.min(widthPct, 100 - leftPct)}%`,
            }}
          />
        </div>

        <div className="w-20 shrink-0 text-right text-xs text-ink-muted">
          {spanDuration ? `${Math.round(spanDuration)}ms` : "-"}
        </div>
      </button>

      {expanded && (
        <div
          id={`${span.span_id}-details`}
          className="space-y-2 border-b border-line bg-panel-muted px-4 py-3 text-xs"
        >
          <div className="grid grid-cols-2 gap-x-6 gap-y-1">
            <div>
              <span className="text-ink-faint">Span ID: </span>
              <MonoText className="font-mono">
                {span.span_id.slice(0, 12)}
              </MonoText>
            </div>
            <div>
              <span className="text-ink-faint">Kind: </span>
              <span>{span.kind}</span>
            </div>
            {span.duration_ms != null && (
              <div>
                <span className="text-ink-faint">Duration: </span>
                <span>{formatRoundedMs(span.duration_ms)}</span>
              </div>
            )}
            {span.error_message && (
              <div className="col-span-2">
                <span className="text-danger">Error: </span>
                <span className="text-danger">{span.error_message}</span>
              </div>
            )}
          </div>

          {Object.keys(span.metrics).length > 0 && (
            <div>
              <TokenMetricsBadges
                metrics={spanMetrics}
                showCacheRead={false}
                showCacheCreation={false}
                chipClassName="bg-panel-strong"
                className="mb-2 grid grid-cols-2 sm:grid-cols-4 gap-2"
              />
              <JsonDisclosure label="Metrics JSON" value={span.metrics} />
            </div>
          )}

          {span.llm_details && (
            <JsonDisclosure label="LLM Details" value={span.llm_details} />
          )}

          {span.tool_details && (
            <JsonDisclosure label="Tool Details" value={span.tool_details} />
          )}

          {span.output_preview && (
            <div>
              <p className="mb-1 text-ink-faint">Output Preview:</p>
              <div className="max-h-48 overflow-auto rounded bg-panel px-3 py-2 whitespace-pre-wrap">
                {span.output_preview}
              </div>
            </div>
          )}
        </div>
      )}
    </>
  );
}

/**
 * Renders the Trace Detail page for a single trace, showing summary metrics, links to related session/agent,
 * the span waterfall with expandable span rows, and optional input/final output panels.
 *
 * The component reads `id` from route params, fetches the trace detail, and manages loading, error, and expanded-span state.
 * While loading it shows a full-page loading message; if the trace is not found it shows a not-found message.
 *
 * @returns The React element for the trace detail page.
 */
export default function TraceDetailPage() {
  const params = useParams();
  const traceId = params.id as string;
  const [trace, setTrace] = useState<TraceDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedSpans, setExpandedSpans] = useState<Set<string>>(new Set());
  const [viewMode, setViewMode] = useState<"mainline" | "debug">("mainline");

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
  const reviewEventCount = trace.timeline_events.filter((event) =>
    ["review_checkpoint", "review_result", "milestone_update"].includes(event.kind),
  ).length;

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <BackHeader
        href="/traces"
        title="Trace Detail"
        subtitle={trace.trace_id}
      />

      {error && <ErrorStateMessage>{error}</ErrorStateMessage>}

      <div className="flex flex-wrap gap-2 text-xs text-ink-muted">
        {trace.session_id && (
          <Link
            href={`/sessions/${trace.session_id}`}
            className="ui-button ui-button-secondary min-h-9 px-3 py-1.5 text-xs"
          >
            Open session
          </Link>
        )}
        {trace.agent_id && (
          <Link
            href={`/agents/${trace.agent_id}`}
            className="ui-button ui-button-secondary min-h-9 px-3 py-1.5 text-xs"
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
            <MetricCard
              label="Runtime Decisions"
              valueClassName="text-lg font-medium"
              value={String(trace.runtime_decisions.length)}
            />
            <MetricCard
              label="Review Events"
              valueClassName="text-lg font-medium"
              value={String(trace.review_cycles.length || reviewEventCount)}
            />
          </>
        }
      />

      <SectionCard
        title="View Mode"
        bodyClassName="flex flex-wrap gap-2 px-4 py-4"
      >
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
      </SectionCard>

      {trace.input_query && (
        <SectionCard className="p-4">
          <p className="mb-1 text-xs text-ink-faint">Input</p>
          <p className="text-sm">{trace.input_query}</p>
        </SectionCard>
      )}

      {viewMode === "mainline" ? (
        <>
          <TraceMainlineEvents events={trace.mainline_events} />
          <TraceReviewCycles cycles={trace.review_cycles} />
          <TraceRuntimeDecisions decisions={trace.runtime_decisions} />
        </>
      ) : (
        <>
          <TraceLlmCalls llmCalls={trace.llm_calls} />
          <TraceRuntimeDecisions decisions={trace.runtime_decisions} />
          <TraceLoopTimeline events={trace.timeline_events} />
          <SectionCard
            className="overflow-hidden"
            title={`Span Waterfall (${trace.spans.length} spans)`}
            headerClassName="border-b border-line px-4 py-3"
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
        </>
      )}

      {trace.final_output && (
        <SectionCard className="p-4">
          <p className="mb-1 text-xs text-ink-faint">Final Output</p>
          <p className="text-sm whitespace-pre-wrap">{trace.final_output}</p>
        </SectionCard>
      )}
    </div>
  );
}
