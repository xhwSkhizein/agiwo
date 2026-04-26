"use client";

import { JsonDisclosure } from "@/components/json-disclosure";
import { MonoText } from "@/components/mono-text";
import { SectionCard } from "@/components/section-card";
import {
  StepContentPreview,
  StructuredValuePreview,
  ToolCallPreviewList,
  contentText,
} from "@/components/step-content-preview";
import type { ToolCallPayload, TraceTimelineEvent } from "@/lib/api";
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

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function detailChips(event: TraceTimelineEvent): string[] {
  const details = event.details;
  if (event.kind === "llm_call") {
    return [
      typeof details.finish_reason === "string" ? details.finish_reason : "",
      Array.isArray(details.messages) ? `${details.messages.length} messages` : "",
      Array.isArray(details.response_tool_calls)
        ? `${details.response_tool_calls.length} tool calls`
        : "",
    ].filter(Boolean);
  }
  if (event.kind === "tool_call") {
    return [
      typeof details.tool_name === "string" ? details.tool_name : "",
      typeof details.tool_call_id === "string" ? details.tool_call_id : "",
      typeof details.status === "string" ? details.status : "",
    ].filter(Boolean);
  }
  if (event.kind === "review_checkpoint") {
    return [
      typeof details.trigger_reason === "string" ? details.trigger_reason : "",
      typeof details.steps_since_last_review === "number"
        ? `${details.steps_since_last_review} steps`
        : "",
      typeof details.active_milestone === "string" ? details.active_milestone : "",
    ].filter(Boolean);
  }
  if (event.kind === "runtime_decision" || event.kind === "review_result") {
    return [
      typeof details.kind === "string" ? details.kind : "",
      typeof details.experience === "string" ? details.experience : "",
      typeof details.affected_count === "number" ? `${details.affected_count} affected` : "",
    ].filter(Boolean);
  }
  return [];
}

function TimelineDetailPreview({ event }: { event: TraceTimelineEvent }) {
  const details = event.details;

  if (event.kind === "llm_call") {
    const response = details.response_content;
    const toolCalls: ToolCallPayload[] = Array.isArray(details.response_tool_calls)
      ? (details.response_tool_calls as ToolCallPayload[])
      : [];
    return (
      <div className="space-y-3">
        <div>
          <div className="mb-1 text-xs font-medium text-ink-faint">Response</div>
          <StepContentPreview value={response} emptyLabel="No assistant response content" />
        </div>
        {toolCalls.length > 0 ? <ToolCallPreviewList toolCalls={toolCalls} /> : null}
      </div>
    );
  }

  if (event.kind === "tool_call") {
    const preferredOutput = details.content_for_user ?? details.output ?? details.error;
    return (
      <div className="space-y-3">
        <div>
          <div className="mb-1 text-xs font-medium text-ink-faint">Arguments</div>
          <StructuredValuePreview value={details.input_args ?? {}} />
        </div>
        <div>
          <div className="mb-1 text-xs font-medium text-ink-faint">Result</div>
          <StepContentPreview value={preferredOutput} emptyLabel="No tool result content" />
        </div>
      </div>
    );
  }

  if (event.kind === "milestone_update" && Array.isArray(details.milestones)) {
    return (
      <div className="space-y-2">
        {details.milestones.slice(0, 6).map((milestone, index) => (
          <div key={index} className="rounded-md bg-panel-muted px-3 py-2 text-sm text-ink-soft">
            {isRecord(milestone) && typeof milestone.description === "string"
              ? milestone.description
              : contentText(milestone) ?? `Milestone ${index + 1}`}
          </div>
        ))}
      </div>
    );
  }

  return <StructuredValuePreview value={details} />;
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
                {detailChips(event).length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {detailChips(event).map((item, chipIndex) => (
                      <span
                        key={`${event.kind}-${event.sequence ?? "na"}-${chipIndex}`}
                        className="rounded-full border border-line bg-panel-muted px-2 py-1 text-[11px] text-ink-muted"
                      >
                        {item}
                      </span>
                    ))}
                  </div>
                ) : null}
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
            <div className="mt-3 space-y-3">
              <TimelineDetailPreview event={event} />
              <JsonDisclosure
                label="Raw details JSON"
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
