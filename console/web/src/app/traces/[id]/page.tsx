"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { Activity, Flag, GitBranch } from "lucide-react";
import { BackHeader } from "@/components/back-header";
import { MetricCard } from "@/components/metric-card";
import { SectionCard } from "@/components/section-card";
import { ErrorStateMessage, FullPageMessage } from "@/components/state-message";
import { TokenSummaryCards } from "@/components/token-summary-cards";
import { TraceDiagnostics } from "@/components/trace-detail/trace-diagnostics";
import { TraceStatusBadge } from "@/components/trace-status-badge";
import { getTrace } from "@/lib/api";
import type { TraceDetail } from "@/lib/api";
import { latestAlignment, latestObjective } from "@/lib/insights";
import { formatDurationMs } from "@/lib/metrics";

interface TraceInsightRailProps {
  trace: TraceDetail;
}

function TraceInsightRail({ trace }: TraceInsightRailProps) {
  const slowestSpan = [...trace.spans]
    .filter((span) => span.duration_ms !== null)
    .sort((a, b) => (b.duration_ms ?? 0) - (a.duration_ms ?? 0))[0];
  const latestDecision = trace.runtime_decisions[trace.runtime_decisions.length - 1];

  return (
    <div className="grid gap-3 lg:grid-cols-4">
      <div className="rounded-xl border border-line bg-panel px-4 py-3 lg:col-span-2">
        <div className="mb-2 flex items-center gap-2 text-xs uppercase tracking-wide text-ink-faint">
          <Flag className="h-3.5 w-3.5" />
          Objective
        </div>
        <p className="text-sm leading-6 text-foreground">{latestObjective(trace)}</p>
      </div>
      <div className="rounded-xl border border-line bg-panel px-4 py-3">
        <div className="mb-2 flex items-center gap-2 text-xs uppercase tracking-wide text-ink-faint">
          <Activity className="h-3.5 w-3.5" />
          Alignment
        </div>
        <p className="text-sm text-foreground">{latestAlignment(trace)}</p>
      </div>
      <div className="rounded-xl border border-line bg-panel px-4 py-3">
        <div className="mb-2 flex items-center gap-2 text-xs uppercase tracking-wide text-ink-faint">
          <GitBranch className="h-3.5 w-3.5" />
          Latest Decision
        </div>
        <p className="truncate text-sm text-foreground">
          {latestDecision?.summary || "No runtime decision"}
        </p>
      </div>
      <div className="rounded-xl border border-line bg-panel px-4 py-3 lg:col-span-4">
        <div className="flex flex-wrap gap-2 text-xs text-ink-muted">
          <span className="rounded-full border border-line bg-panel-muted px-2 py-1">
            {trace.mainline_events.length} narrative events
          </span>
          <span className="rounded-full border border-line bg-panel-muted px-2 py-1">
            {trace.review_cycles.length} review cycles
          </span>
          <span className="rounded-full border border-line bg-panel-muted px-2 py-1">
            {trace.runtime_decisions.length} runtime decisions
          </span>
          <span className="rounded-full border border-line bg-panel-muted px-2 py-1">
            slowest {slowestSpan?.name || "-"}{" "}
            {slowestSpan?.duration_ms ? formatDurationMs(slowestSpan.duration_ms) : ""}
          </span>
        </div>
      </div>
      {trace.final_output ? (
        <div className="rounded-xl border border-line bg-panel px-4 py-3 lg:col-span-4">
          <div className="mb-2 text-xs uppercase tracking-wide text-ink-faint">
            Final Output Preview
          </div>
          <p className="line-clamp-3 text-sm leading-6 text-foreground">
            {trace.final_output}
          </p>
        </div>
      ) : null}
    </div>
  );
}

/**
 * Renders the Trace Detail page for a single trace, showing summary metrics,
 * related links, and a unified execution diagnostics chain.
 *
 * The component reads `id` from route params, fetches the trace detail, and
 * manages `trace`, `loading`, and `error` state.
 *
 * @returns The React element for the trace detail page.
 */
export default function TraceDetailPage() {
  const params = useParams();
  const traceId = params.id as string;
  const [trace, setTrace] = useState<TraceDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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

  if (loading) {
    return <FullPageMessage>Loading trace...</FullPageMessage>;
  }

  if (!trace) {
    return <FullPageMessage>Trace not found</FullPageMessage>;
  }

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
            href={`/scheduler/${trace.agent_id}`}
            className="ui-button ui-button-secondary min-h-9 px-3 py-1.5 text-xs"
          >
            Open scheduler state
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

      <TraceInsightRail trace={trace} />

      {trace.input_query && (
        <SectionCard className="p-4">
          <p className="mb-1 text-xs text-ink-faint">Input</p>
          <p className="text-sm">{trace.input_query}</p>
        </SectionCard>
      )}

      <TraceDiagnostics trace={trace} />

      {trace.final_output && (
        <SectionCard className="p-4">
          <p className="mb-1 text-xs text-ink-faint">Final Output</p>
          <p className="text-sm whitespace-pre-wrap">{trace.final_output}</p>
        </SectionCard>
      )}
    </div>
  );
}
