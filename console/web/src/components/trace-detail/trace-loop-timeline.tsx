"use client";

import { JsonDisclosure } from "@/components/json-disclosure";
import { MonoText } from "@/components/mono-text";
import { SectionCard } from "@/components/section-card";
import type { TraceTimelineEvent } from "@/lib/api";
import { formatLocalDateTime } from "@/lib/time";

function timelineStatusBadgeClass(status: string): string {
  if (status === "error") {
    return "border-red-500/40 bg-red-500/10 text-red-200";
  }
  if (status === "ok") {
    return "border-emerald-500/40 bg-emerald-500/10 text-emerald-200";
  }
  return "border-line bg-panel-muted text-ink-muted";
}

export function TraceLoopTimeline({
  events,
}: {
  events: TraceTimelineEvent[];
}) {
  return (
    <SectionCard title="Loop Timeline" bodyClassName="space-y-3 px-4 py-4">
      {events.length === 0 ? (
        <div className="rounded-xl border border-dashed border-line px-4 py-6 text-sm text-ink-muted">
          No loop events replayed for this trace.
        </div>
      ) : (
        events.map((event, index) => (
          <details
            key={`${event.kind}-${event.sequence ?? "na"}-${event.span_id ?? event.title}-${event.timestamp ?? "na"}-${index}`}
            className="rounded-xl border border-line bg-panel px-3 py-3"
          >
            <summary className="list-none cursor-pointer">
              <div className="space-y-1.5">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm font-medium text-foreground">{event.title}</span>
                  <span className="rounded-full border border-line px-2 py-0.5 text-[11px] uppercase tracking-wide text-ink-muted">
                    {event.kind}
                  </span>
                  {event.status ? (
                    <span
                      className={`rounded-full border px-2 py-0.5 text-[11px] uppercase tracking-wide ${timelineStatusBadgeClass(event.status)}`}
                    >
                      {event.status}
                    </span>
                  ) : null}
                </div>
                <p className="text-sm text-foreground">{event.summary}</p>
                <div className="flex flex-wrap gap-3 text-xs text-ink-muted">
                  {event.run_id && (
                    <span>
                      Run <MonoText>{event.run_id}</MonoText>
                    </span>
                  )}
                  {event.agent_id && (
                    <span>
                      Agent <MonoText>{event.agent_id}</MonoText>
                    </span>
                  )}
                  <span>seq {event.sequence ?? "-"}</span>
                  <span>{formatLocalDateTime(event.timestamp)}</span>
                </div>
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
        ))
      )}
    </SectionCard>
  );
}

export default TraceLoopTimeline;
