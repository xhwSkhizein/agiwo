"use client";

import { JsonDisclosure } from "@/components/json-disclosure";
import { SectionCard } from "@/components/section-card";
import type { ReviewCycle } from "@/lib/api";
import { formatLocalDateTime } from "@/lib/time";

function reviewSummary(cycle: ReviewCycle): string {
  if (cycle.aligned === true) {
    return "Aligned with the current milestone";
  }
  if (cycle.aligned === false) {
    const scored = cycle.tool_usefulness.filter((item) => item.score !== null);
    if (scored.length > 0) {
      return `Misaligned; ${scored.length} usefulness scores recorded`;
    }
    return "Misaligned with the current milestone";
  }
  return "Checkpoint recorded";
}

export function TraceReviewCycles({
  cycles,
}: {
  cycles: ReviewCycle[];
}) {
  return (
    <SectionCard title="Review Cycles" bodyClassName="space-y-3 px-4 py-4">
      {cycles.length === 0 ? (
        <div className="rounded-xl border border-dashed border-line px-4 py-6 text-sm text-ink-muted">
          No review cycles detected in this trace.
        </div>
      ) : (
        cycles.map((cycle) => (
          <details
            key={cycle.cycle_id}
            className="rounded-xl border border-line bg-panel px-3 py-3"
          >
            <summary className="cursor-pointer list-none">
              <div className="space-y-2">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm font-medium text-foreground">
                    {reviewSummary(cycle)}
                  </span>
                  <span className="rounded-full border border-line px-2 py-0.5 text-[11px] uppercase tracking-wide text-ink-muted">
                    {cycle.trigger_reason}
                  </span>
                  {cycle.active_milestone ? (
                    <span className="rounded-full border border-line px-2 py-0.5 text-[11px] text-ink-muted">
                      {cycle.active_milestone}
                    </span>
                  ) : null}
                </div>
                <div className="flex flex-wrap gap-2 text-xs text-ink-muted">
                  {cycle.steps_since_last_review !== null ? (
                    <span>{cycle.steps_since_last_review} steps since last review</span>
                  ) : null}
                  {cycle.review_latency_ms !== null ? (
                    <span>{cycle.review_latency_ms.toFixed(0)} ms review latency</span>
                  ) : null}
                  {cycle.resolved_at ? (
                    <span>{formatLocalDateTime(cycle.resolved_at)}</span>
                  ) : cycle.started_at ? (
                    <span>{formatLocalDateTime(cycle.started_at)}</span>
                  ) : null}
                </div>
                {cycle.tool_usefulness.length > 0 ? (
                  <div className="flex flex-wrap gap-2 text-xs text-ink-muted">
                    {cycle.tool_usefulness.map((item) => (
                      <span
                        key={item.tool_call_id}
                        className="rounded-full border border-line px-2 py-0.5"
                      >
                        {item.tool_call_id}
                        {item.score === null ? ": unknown" : `: ${item.score}`}
                      </span>
                    ))}
                  </div>
                ) : null}
                {cycle.experience ? (
                  <p className="text-sm text-foreground">{cycle.experience}</p>
                ) : null}
              </div>
            </summary>
            <div className="mt-3 space-y-3">
              {cycle.raw_notice ? (
                <div className="rounded-xl border border-line bg-panel-muted px-3 py-3 text-xs whitespace-pre-wrap text-ink-muted">
                  {cycle.raw_notice}
                </div>
              ) : null}
              <JsonDisclosure
                label="Structured Details"
                value={cycle}
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

export default TraceReviewCycles;
