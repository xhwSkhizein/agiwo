"use client";

import { JsonDisclosure } from "@/components/json-disclosure";
import { MonoText } from "@/components/mono-text";
import { SectionCard } from "@/components/section-card";
import type { RuntimeDecisionEvent } from "@/lib/api";
import { formatLocalDateTime } from "@/lib/time";

function decisionPreviewItems(decision: RuntimeDecisionEvent): string[] {
  if (decision.kind === "step_back") {
    return [
      `affected_count ${String(decision.details.affected_count ?? "-")}`,
      `checkpoint_seq ${String(decision.details.checkpoint_seq ?? "-")}`,
      typeof decision.details.experience === "string" ? decision.details.experience : "",
    ].filter(Boolean);
  }
  if (decision.kind === "compaction_failed") {
    return [
      `attempt ${String(decision.details.attempt ?? "-")}/${String(decision.details.max_attempts ?? "-")}`,
      typeof decision.details.error === "string" ? decision.details.error : "",
    ].filter(Boolean);
  }
  return Object.entries(decision.details)
    .slice(0, 3)
    .map(([key, value]) => `${key} ${String(value)}`);
}

export function TraceRuntimeDecisions({
  decisions,
}: {
  decisions: RuntimeDecisionEvent[];
}) {
  return (
    <SectionCard title="Runtime Decisions" bodyClassName="space-y-3 px-4 py-4">
      {decisions.length === 0 ? (
        <div className="rounded-xl border border-dashed border-line px-4 py-6 text-sm text-ink-muted">
          No runtime decisions recorded for this trace.
        </div>
      ) : (
        decisions.map((decision) => (
          <details
            key={`${decision.kind}-${decision.sequence}-${decision.run_id}`}
            className="rounded-xl border border-line bg-panel px-3 py-3"
          >
            <summary className="list-none cursor-pointer">
              <div className="space-y-2">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm font-medium text-foreground">{decision.summary}</span>
                  <span className="rounded-full border border-line px-2 py-0.5 text-[11px] uppercase tracking-wide text-ink-muted">
                    {decision.kind}
                  </span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {decisionPreviewItems(decision).map((item) => (
                    <span
                      key={item}
                      className="rounded-full border border-line bg-panel-muted px-2 py-1 text-[11px] text-ink-muted"
                    >
                      {item}
                    </span>
                  ))}
                </div>
                <div className="flex flex-wrap gap-3 text-xs text-ink-muted">
                  <span>
                    Run <MonoText>{decision.run_id}</MonoText>
                  </span>
                  <span>
                    Agent <MonoText>{decision.agent_id}</MonoText>
                  </span>
                  <span>seq {decision.sequence}</span>
                  <span>{formatLocalDateTime(decision.created_at)}</span>
                </div>
              </div>
            </summary>
            <div className="mt-3">
              <JsonDisclosure
                label="Details"
                value={decision.details}
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

export default TraceRuntimeDecisions;
