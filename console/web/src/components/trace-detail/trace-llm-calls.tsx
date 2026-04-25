"use client";

import { SectionCard } from "@/components/section-card";
import type { TraceLlmCall } from "@/lib/api";
import { formatDurationMs, formatTokenCount } from "@/lib/metrics";

function metricChip(label: string, value: string) {
  return (
    <span className="rounded-full border border-line bg-panel-muted px-2 py-1 text-[11px] text-ink-muted">
      {label} {value}
    </span>
  );
}

export function TraceLlmCalls({
  llmCalls,
}: {
  llmCalls: TraceLlmCall[];
}) {
  const tokenValue = (value: number | null) =>
    value !== null ? formatTokenCount(value) : "-";

  return (
    <SectionCard title="LLM Calls" bodyClassName="space-y-3 px-4 py-4">
      {llmCalls.length === 0 ? (
        <div className="rounded-xl border border-dashed border-line px-4 py-6 text-sm text-ink-muted">
          No LLM calls recorded for this trace.
        </div>
      ) : (
        llmCalls.map((call) => (
          <div
            key={call.span_id}
            className="rounded-xl border border-line bg-panel px-3 py-3"
          >
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm font-medium text-foreground">
                {call.model || "LLM"}
              </span>
              {call.provider ? (
                <span className="rounded-full border border-line px-2 py-0.5 text-[11px] text-ink-muted">
                  {call.provider}
                </span>
              ) : null}
              {call.finish_reason ? (
                <span className="rounded-full border border-line px-2 py-0.5 text-[11px] uppercase tracking-wide text-ink-muted">
                  {call.finish_reason}
                </span>
              ) : null}
            </div>
            <div className="mt-2 flex flex-wrap gap-2">
              {metricChip(
                "duration",
                call.duration_ms != null
                  ? formatDurationMs(call.duration_ms)
                  : "-",
              )}
              {metricChip(
                "first token",
                call.first_token_latency_ms !== null
                  ? formatDurationMs(call.first_token_latency_ms)
                  : "-",
              )}
              {metricChip("input", tokenValue(call.input_tokens))}
              {metricChip("output", tokenValue(call.output_tokens))}
              {metricChip("total", tokenValue(call.total_tokens))}
              {metricChip("messages", String(call.message_count))}
              {metricChip("tool schemas", String(call.tool_schema_count))}
              {metricChip("tool calls", String(call.response_tool_call_count))}
            </div>
            {call.output_preview ? (
              <div className="mt-3 rounded-xl border border-line bg-panel-muted px-3 py-3 text-sm whitespace-pre-wrap text-foreground">
                {call.output_preview}
              </div>
            ) : null}
          </div>
        ))
      )}
    </SectionCard>
  );
}

export default TraceLlmCalls;
