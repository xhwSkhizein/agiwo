"use client";

import Link from "next/link";
import { Activity, ArrowRight, ChevronRight, Clock3, GitBranch, Scissors, ShieldCheck } from "lucide-react";

import { JsonDisclosure } from "@/components/json-disclosure";
import { MonoText } from "@/components/mono-text";
import { SectionCard } from "@/components/section-card";
import { TraceStatusBadge } from "@/components/trace-status-badge";
import type { RuntimeDecisionEvent, SessionObservability } from "@/lib/api";
import { formatDurationMs } from "@/lib/metrics";
import { formatLocalDateTime } from "@/lib/time";

function DecisionIcon({ kind }: { kind: RuntimeDecisionEvent["kind"] }) {
  if (kind === "termination") return <ShieldCheck className="h-4 w-4 text-red-300" />;
  if (kind === "compaction") return <Scissors className="h-4 w-4 text-cyan-300" />;
  if (kind === "compaction_failed") return <Scissors className="h-4 w-4 text-red-300" />;
  if (kind === "step_back") return <Activity className="h-4 w-4 text-amber-300" />;
  return <GitBranch className="h-4 w-4 text-zinc-300" />;
}

const DETAIL_PREVIEW_MAX_CHARS = 72;

function formatPreviewDetailValue(value: unknown): string | null {
  if (
    value === null ||
    value === undefined ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return String(value);
  }
  try {
    const json = JSON.stringify(value);
    if (!json) return null;
    if (json.length <= DETAIL_PREVIEW_MAX_CHARS) return json;
    return `${json.slice(0, DETAIL_PREVIEW_MAX_CHARS - 1)}…`;
  } catch {
    return null;
  }
}

function decisionPreviewItems(event: RuntimeDecisionEvent): string[] {
  if (event.kind === "step_back") {
    return [
      `affected_count ${String(event.details.affected_count ?? "-")}`,
      `checkpoint_seq ${String(event.details.checkpoint_seq ?? "-")}`,
      typeof event.details.experience === "string" ? event.details.experience : "",
    ].filter(Boolean);
  }
  if (event.kind === "compaction") {
    return [
      `seq ${String(event.details.start_sequence ?? "-")}-${String(event.details.end_sequence ?? "-")}`,
      `${String(event.details.before_token_estimate ?? "-")} -> ${String(event.details.after_token_estimate ?? "-")} tokens`,
    ];
  }
  if (event.kind === "compaction_failed") {
    return [
      `attempt ${String(event.details.attempt ?? "-")}/${String(event.details.max_attempts ?? "-")}`,
      typeof event.details.error === "string" ? event.details.error : "",
    ].filter(Boolean);
  }
  if (event.kind === "rollback") {
    return [
      `seq ${String(event.details.start_sequence ?? "-")}-${String(event.details.end_sequence ?? "-")}`,
      typeof event.details.reason === "string" ? event.details.reason : "",
    ].filter(Boolean);
  }
  if (event.kind === "termination") {
    return [
      typeof event.details.reason === "string" ? event.details.reason : "",
      typeof event.details.phase === "string" ? event.details.phase : "",
      typeof event.details.source === "string" ? event.details.source : "",
    ].filter(Boolean);
  }
  return Object.entries(event.details)
    .slice(0, 3)
    .flatMap(([key, value]) => {
      const preview = formatPreviewDetailValue(value);
      return preview ? [`${key} ${preview}`] : [];
    });
}

function DecisionChevron() {
  return (
    <span className="inline-flex items-center text-ink-muted">
      <span className="sr-only">Toggle decision details</span>
      <ChevronRight
        aria-hidden="true"
        className="h-4 w-4 transition-transform duration-150 group-open:rotate-90"
      />
    </span>
  );
}

function DecisionCard({ event }: { event: RuntimeDecisionEvent }) {
  const previewItems = decisionPreviewItems(event);
  return (
    <details className="group rounded-xl border border-line bg-panel px-3 py-3">
      <summary className="list-none cursor-pointer">
        <div className="flex items-start gap-3">
          <DecisionIcon kind={event.kind} />
          <div className="min-w-0 flex-1 space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm font-medium capitalize text-foreground">{event.kind}</span>
              <span className="rounded-full border border-line px-2 py-0.5 text-[11px] uppercase tracking-wide text-ink-muted">
                seq {event.sequence}
              </span>
            </div>
            <p className="text-sm text-foreground">{event.summary}</p>
            {previewItems.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {previewItems.map((item, index) => (
                  <span
                    key={`${item}-${index}`}
                    className="rounded-full border border-line bg-panel-muted px-2 py-1 text-[11px] text-ink-muted"
                  >
                    {item}
                  </span>
                ))}
              </div>
            )}
            <div className="flex flex-wrap gap-3 text-xs text-ink-muted">
              <span>
                Run <MonoText>{event.run_id}</MonoText>
              </span>
              <span>
                Agent <MonoText>{event.agent_id}</MonoText>
              </span>
              <span>{formatLocalDateTime(event.created_at)}</span>
            </div>
          </div>
          <DecisionChevron />
        </div>
      </summary>
      <div className="mt-3">
        <JsonDisclosure
          label="Details"
          value={event.details}
          className="bg-panel"
          contentClassName="bg-panel"
        />
      </div>
    </details>
  );
}

interface SessionObservabilityPanelProps {
  sessionId: string;
  observability: SessionObservability | null;
  compact?: boolean;
  className?: string;
}

export function SessionObservabilityPanel({
  sessionId,
  observability,
  compact = false,
  className,
}: SessionObservabilityPanelProps) {
  const recentTraces = observability?.recent_traces ?? [];
  const decisionEvents = observability?.decision_events ?? [];
  const latestTrace = recentTraces[0] ?? null;

  return (
    <SectionCard
      title="Observability"
      className={className}
      action={
        <Link
          href={`/traces?session_id=${sessionId}`}
          className="inline-flex items-center gap-1 text-xs text-ink-muted transition-colors hover:text-foreground"
        >
          All traces
          <ArrowRight className="h-3 w-3" />
        </Link>
      }
      bodyClassName={
        compact
          ? "space-y-4 px-4 py-4"
          : "grid gap-4 px-4 py-4 lg:grid-cols-[1.15fr_0.85fr]"
      }
    >
      <div className="space-y-4">
        <div className={compact ? "grid gap-2" : "grid gap-3 md:grid-cols-3"}>
          <div className="rounded-xl border border-line bg-panel-muted px-3 py-3">
            <div className="text-[11px] uppercase tracking-wide text-ink-faint">Recent Traces</div>
            <div className="mt-2 text-lg font-semibold text-foreground">{recentTraces.length}</div>
          </div>
          <div className="rounded-xl border border-line bg-panel-muted px-3 py-3">
            <div className="text-[11px] uppercase tracking-wide text-ink-faint">Latest Trace</div>
            <div className="mt-2 text-sm text-foreground">
              {latestTrace ? <TraceStatusBadge status={latestTrace.status} /> : "No traces"}
            </div>
          </div>
          <div className="rounded-xl border border-line bg-panel-muted px-3 py-3">
            <div className="text-[11px] uppercase tracking-wide text-ink-faint">Latest Duration</div>
            <div className="mt-2 text-lg font-semibold text-foreground">
              {latestTrace ? formatDurationMs(latestTrace.duration_ms || 0) : "-"}
            </div>
          </div>
        </div>

        <div className="space-y-2">
          <div className="text-xs uppercase tracking-wide text-ink-faint">Trace Context</div>
          {recentTraces.length === 0 ? (
            <div className="rounded-xl border border-dashed border-line px-4 py-6 text-sm text-ink-muted">
              No traces captured for this session yet.
            </div>
          ) : (
            <div className="space-y-2">
              {recentTraces.map((trace) => (
                <Link
                  key={trace.trace_id}
                  href={`/traces/${trace.trace_id}`}
                  className="flex items-center justify-between gap-3 rounded-xl border border-line bg-panel px-3 py-3 transition-colors hover:border-line-strong hover:bg-panel-strong"
                >
                  <div className="min-w-0 space-y-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <TraceStatusBadge status={trace.status} />
                      <MonoText className="truncate text-xs">{trace.trace_id.slice(0, 12)}</MonoText>
                    </div>
                    <div className="text-sm text-foreground">
                      {trace.input_query || trace.final_output || "Trace detail"}
                    </div>
                  </div>
                  <div className="shrink-0 text-right text-xs text-ink-muted">
                    <div>{formatDurationMs(trace.duration_ms || 0)}</div>
                    <div>{formatLocalDateTime(trace.start_time)}</div>
                  </div>
                </Link>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="space-y-2">
        <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-ink-faint">
          <Clock3 className="h-3.5 w-3.5" />
          Runtime Decisions
        </div>
        {decisionEvents.length === 0 ? (
          <div className="rounded-xl border border-dashed border-line px-4 py-6 text-sm text-ink-muted">
            No runtime decisions replayed for this session.
          </div>
        ) : (
          <div className="space-y-2">
            {decisionEvents.map((event) => (
              <DecisionCard
                key={`${event.kind}-${event.sequence}-${event.run_id}`}
                event={event}
              />
            ))}
          </div>
        )}
      </div>
    </SectionCard>
  );
}

export default SessionObservabilityPanel;
