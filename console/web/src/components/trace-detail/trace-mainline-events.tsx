"use client";

import { JsonDisclosure } from "@/components/json-disclosure";
import { MonoText } from "@/components/mono-text";
import { SectionCard } from "@/components/section-card";
import type { TraceMainlineEvent } from "@/lib/api";
import { formatLocalDateTime } from "@/lib/time";

function eventPreviewItems(event: TraceMainlineEvent): string[] {
  if (event.kind === "review_checkpoint") {
    return [
      typeof event.details.trigger_reason === "string"
        ? `trigger ${event.details.trigger_reason}`
        : "",
      typeof event.details.steps_since_last_review === "number"
        ? `${event.details.steps_since_last_review} steps`
        : "",
      typeof event.details.active_milestone === "string"
        ? event.details.active_milestone
        : "",
    ].filter(Boolean);
  }
  if (event.kind === "runtime_decision") {
    return [
      typeof event.details.kind === "string" ? event.details.kind : "",
      typeof event.details.experience === "string" ? event.details.experience : "",
      typeof event.details.affected_count === "number"
        ? `${event.details.affected_count} affected`
        : "",
    ].filter(Boolean);
  }
  if (event.kind === "milestone_update") {
    const milestones = Array.isArray(event.details.milestones)
      ? event.details.milestones
      : [];
    return milestones
      .slice(0, 3)
      .flatMap((item) =>
        item && typeof item === "object" && "description" in item
          ? [String(item.description)]
          : [],
      );
  }
  return [];
}

export function TraceMainlineEvents({
  events,
}: {
  events: TraceMainlineEvent[];
}) {
  return (
    <SectionCard title="Run Narrative" bodyClassName="space-y-3 px-4 py-4">
      {events.length === 0 ? (
        <div className="rounded-xl border border-dashed border-line px-4 py-6 text-sm text-ink-muted">
          No mainline narrative available for this trace.
        </div>
      ) : (
        <div className="relative space-y-3 before:absolute before:left-[1.05rem] before:top-3 before:h-[calc(100%-1.5rem)] before:w-px before:bg-line">
          {events.map((event) => {
          const previewItems = eventPreviewItems(event);
          return (
            <details
              key={event.id}
              className="relative ml-10 rounded-xl border border-line bg-panel px-3 py-3"
            >
              <summary className="cursor-pointer list-none">
                <div className="absolute -left-10 top-3 flex h-8 w-8 items-center justify-center rounded-full border border-line bg-panel font-mono text-[11px] text-ink-muted">
                  {event.sequence ?? "-"}
                </div>
                <div className="space-y-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-sm font-medium text-foreground">
                      {event.title}
                    </span>
                    <span className="rounded-full border border-line px-2 py-0.5 text-[11px] uppercase tracking-wide text-ink-muted">
                      {event.kind}
                    </span>
                    {event.status ? (
                      <span className="rounded-full border border-line px-2 py-0.5 text-[11px] uppercase tracking-wide text-ink-muted">
                        {event.status}
                      </span>
                    ) : null}
                  </div>
                  <p className="text-sm text-foreground">{event.summary}</p>
                  {previewItems.length > 0 ? (
                    <div className="flex flex-wrap gap-2">
                      {previewItems.map((item, index) => (
                        <span
                          key={`${event.id}-${index}`}
                          className="rounded-full border border-line bg-panel-muted px-2 py-1 text-[11px] text-ink-muted"
                        >
                          {item}
                        </span>
                      ))}
                    </div>
                  ) : null}
                  <div className="flex flex-wrap gap-3 text-xs text-ink-muted">
                    {event.run_id ? (
                      <span>
                        Run <MonoText>{event.run_id}</MonoText>
                      </span>
                    ) : null}
                    {event.agent_id ? (
                      <span>
                        Agent <MonoText>{event.agent_id}</MonoText>
                      </span>
                    ) : null}
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
          );
        })}
        </div>
      )}
    </SectionCard>
  );
}

export default TraceMainlineEvents;
