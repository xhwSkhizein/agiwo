"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { BackHeader } from "@/components/back-header";
import { MetricCard } from "@/components/metric-card";
import { SectionCard } from "@/components/section-card";
import { ErrorStateMessage, FullPageMessage } from "@/components/state-message";
import { TokenSummaryCards } from "@/components/token-summary-cards";
import { TraceFlameExplorer } from "@/components/trace-detail/trace-flame-explorer";
import { TraceStatusBadge } from "@/components/trace-status-badge";
import { getTrace } from "@/lib/api";
import type { TraceDetail } from "@/lib/api";
import { formatDurationMs } from "@/lib/metrics";

/**
 * Trace detail: clickable execution timeline + right-hand span detail rail.
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

  return (
    <div className="mx-auto max-w-7xl space-y-5 p-6">
      <BackHeader href="/traces" title="Trace Detail" subtitle={trace.trace_id} />

      {error && <ErrorStateMessage>{error}</ErrorStateMessage>}

      <div className="flex flex-wrap items-center gap-2 text-xs text-ink-muted">
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
        <span className="ml-auto text-ink-faint">
          Click a span on the timeline → details on the right
        </span>
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
              label="Max depth"
              valueClassName="text-lg font-medium"
              value={String(trace.max_depth)}
            />
          </>
        }
      />

      {trace.input_query && (
        <SectionCard className="p-4">
          <p className="mb-1 text-xs text-ink-faint">Input</p>
          <p className="text-sm">{trace.input_query}</p>
        </SectionCard>
      )}

      <TraceFlameExplorer trace={trace} />

      {trace.final_output && (
        <SectionCard className="p-4">
          <p className="mb-1 text-xs text-ink-faint">Final Output</p>
          <p className="whitespace-pre-wrap text-sm">{trace.final_output}</p>
        </SectionCard>
      )}
    </div>
  );
}
