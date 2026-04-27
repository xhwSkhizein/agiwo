"use client";

import { AlertTriangle, Bot, Clock, GitBranch, Wrench, Zap } from "lucide-react";

import { JsonDisclosure } from "@/components/json-disclosure";
import { MonoText } from "@/components/mono-text";
import { SectionCard } from "@/components/section-card";
import {
  StepContentPreview,
  StructuredValuePreview,
  ToolCallPreviewList,
  contentText,
  stringifyPretty,
} from "@/components/step-content-preview";
import type { SpanResponse, ToolCallPayload, TraceDetail } from "@/lib/api";
import { formatDurationMs, formatTokenCount } from "@/lib/metrics";
import { formatLocalDateTime } from "@/lib/time";

type RecordValue = Record<string, unknown>;

function isRecord(value: unknown): value is RecordValue {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function stringValue(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function numberValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function compact(value: unknown, maxLength = 180): string {
  const text =
    typeof value === "string" ? value : contentText(value) ?? stringifyPretty(value);
  const normalized = text.replace(/\s+/g, " ").trim();
  if (normalized.length <= maxLength) {
    return normalized;
  }
  return `${normalized.slice(0, maxLength - 3)}...`;
}

function spanSequence(span: SpanResponse): number | null {
  return numberValue(span.attributes.sequence);
}

function llmSpans(trace: TraceDetail): SpanResponse[] {
  return trace.spans.filter((span) => span.kind === "llm_call" && span.llm_details);
}

function toolSpans(trace: TraceDetail): SpanResponse[] {
  return trace.spans.filter((span) => span.kind === "tool_call" && span.tool_details);
}

function toolCallId(span: SpanResponse): string | null {
  return stringValue(span.tool_details?.tool_call_id) ?? stringValue(span.attributes.tool_call_id);
}

function toolName(span: SpanResponse): string {
  return stringValue(span.tool_details?.tool_name) ?? span.name;
}

function responseToolCalls(span: SpanResponse): ToolCallPayload[] {
  const calls = span.llm_details?.response_tool_calls;
  return Array.isArray(calls) ? (calls as ToolCallPayload[]) : [];
}

function traceRiskSignals(trace: TraceDetail): string[] {
  const signals: string[] = [];
  const spans = trace.spans;
  const errors = spans.filter((span) => span.status === "error" || span.error_message);
  if (errors.length > 0) {
    signals.push(`${errors.length} errored span${errors.length === 1 ? "" : "s"}`);
  }

  const slowest = [...spans]
    .filter((span) => span.duration_ms !== null)
    .sort((a, b) => (b.duration_ms ?? 0) - (a.duration_ms ?? 0))[0];
  if (slowest && (slowest.duration_ms ?? 0) >= 10_000) {
    signals.push(`slow span ${slowest.name} took ${formatDurationMs(slowest.duration_ms ?? 0)}`);
  }

  const stepBacks = trace.runtime_decisions.filter((decision) => decision.kind === "step_back");
  if (stepBacks.length > 0) {
    signals.push(`${stepBacks.length} step-back decision${stepBacks.length === 1 ? "" : "s"}`);
  }

  const compactionFailures = trace.runtime_decisions.filter(
    (decision) => decision.kind === "compaction_failed",
  );
  if (compactionFailures.length > 0) {
    signals.push(`${compactionFailures.length} compaction failure${compactionFailures.length === 1 ? "" : "s"}`);
  }

  const abnormalFinishes = llmSpans(trace).filter((span) => {
    const finishReason = stringValue(span.llm_details?.finish_reason);
    return finishReason && !["stop", "tool_calls", "end_turn", "complete"].includes(finishReason);
  });
  if (abnormalFinishes.length > 0) {
    signals.push(`${abnormalFinishes.length} abnormal LLM finish reason${abnormalFinishes.length === 1 ? "" : "s"}`);
  }

  const toolFingerprints = new Map<string, number>();
  for (const span of toolSpans(trace)) {
    const key = `${toolName(span)}:${compact(span.tool_details?.input_args ?? {}, 120)}`;
    toolFingerprints.set(key, (toolFingerprints.get(key) ?? 0) + 1);
  }
  const repeatedTools = [...toolFingerprints.values()].filter((count) => count > 1).length;
  if (repeatedTools > 0) {
    signals.push(`${repeatedTools} repeated tool argument pattern${repeatedTools === 1 ? "" : "s"}`);
  }

  if (trace.total_tokens > 80_000) {
    signals.push(`high token usage ${formatTokenCount(trace.total_tokens)}`);
  }

  return signals;
}

function DiagnosticsOverview({ trace }: { trace: TraceDetail }) {
  const risks = traceRiskSignals(trace);
  const slowest = [...trace.spans]
    .filter((span) => span.duration_ms !== null)
    .sort((a, b) => (b.duration_ms ?? 0) - (a.duration_ms ?? 0))[0];
  const mostExpensive = [...llmSpans(trace)]
    .sort(
      (a, b) =>
        (numberValue(b.metrics["tokens.total"]) ?? 0) -
        (numberValue(a.metrics["tokens.total"]) ?? 0),
    )[0];

  return (
    <SectionCard title="Diagnostic Summary" bodyClassName="grid gap-3 px-4 py-4 lg:grid-cols-3">
      <div className="rounded-xl border border-line bg-panel-muted px-3 py-3">
        <div className="mb-2 flex items-center gap-2 text-xs uppercase tracking-wide text-ink-faint">
          <AlertTriangle className="h-3.5 w-3.5" />
          Risk Signals
        </div>
        {risks.length === 0 ? (
          <p className="text-sm text-ink-muted">No obvious rule-based risks detected.</p>
        ) : (
          <div className="flex flex-wrap gap-2">
            {risks.map((risk) => (
              <span
                key={risk}
                className="rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-1 text-[11px] text-amber-200"
              >
                {risk}
              </span>
            ))}
          </div>
        )}
      </div>
      <div className="rounded-xl border border-line bg-panel-muted px-3 py-3">
        <div className="mb-2 flex items-center gap-2 text-xs uppercase tracking-wide text-ink-faint">
          <Clock className="h-3.5 w-3.5" />
          Bottleneck
        </div>
        <p className="text-sm text-foreground">
          {slowest
            ? `${slowest.name} (${formatDurationMs(slowest.duration_ms ?? 0)})`
            : "No timed spans"}
        </p>
      </div>
      <div className="rounded-xl border border-line bg-panel-muted px-3 py-3">
        <div className="mb-2 flex items-center gap-2 text-xs uppercase tracking-wide text-ink-faint">
          <Zap className="h-3.5 w-3.5" />
          Heaviest LLM Call
        </div>
        <p className="text-sm text-foreground">
          {mostExpensive
            ? `${mostExpensive.name} (${formatTokenCount(numberValue(mostExpensive.metrics["tokens.total"]) ?? 0)} tokens)`
            : "No LLM calls"}
        </p>
      </div>
    </SectionCard>
  );
}

function messageRole(message: unknown): string {
  if (isRecord(message)) {
    return stringValue(message.role) ?? stringValue(message.type) ?? "message";
  }
  return "message";
}

function messageContent(message: unknown): unknown {
  if (isRecord(message)) {
    return message.content ?? message.text ?? message;
  }
  return message;
}

function LlmCallInspector({ trace }: { trace: TraceDetail }) {
  const spans = llmSpans(trace);
  return (
    <SectionCard title="LLM Call Inspector" bodyClassName="space-y-3 px-4 py-4">
      {spans.length === 0 ? (
        <div className="rounded-xl border border-dashed border-line px-4 py-6 text-sm text-ink-muted">
          No LLM call details recorded for this trace.
        </div>
      ) : (
        spans.map((span, index) => {
          const details = span.llm_details ?? {};
          const messages = Array.isArray(details.messages) ? details.messages : [];
          const tools = Array.isArray(details.tools) ? details.tools : [];
          const calls = responseToolCalls(span);
          return (
            <details key={span.span_id} className="rounded-xl border border-line bg-panel px-3 py-3">
              <summary className="cursor-pointer list-none">
                <div className="flex flex-wrap items-center gap-2">
                  <Bot className="h-4 w-4 text-success" />
                  <span className="text-sm font-medium text-foreground">
                    {span.name || `LLM call ${index + 1}`}
                  </span>
                  <span className="rounded-full border border-line px-2 py-0.5 text-[11px] text-ink-muted">
                    seq {spanSequence(span) ?? "-"}
                  </span>
                  <span className="rounded-full border border-line px-2 py-0.5 text-[11px] text-ink-muted">
                    {messages.length} messages
                  </span>
                  <span className="rounded-full border border-line px-2 py-0.5 text-[11px] text-ink-muted">
                    {tools.length} tools available
                  </span>
                  {stringValue(details.finish_reason) ? (
                    <span className="rounded-full border border-line px-2 py-0.5 text-[11px] uppercase tracking-wide text-ink-muted">
                      {stringValue(details.finish_reason)}
                    </span>
                  ) : null}
                </div>
                <p className="mt-2 text-sm text-ink-muted">
                  {compact(details.response_content ?? span.output_preview ?? "No response preview")}
                </p>
              </summary>
              <div className="mt-3 space-y-4">
                <div>
                  <div className="mb-2 text-xs uppercase tracking-wide text-ink-faint">
                    Prompt Messages
                  </div>
                  <div className="space-y-2">
                    {messages.slice(0, 20).map((message, messageIndex) => (
                      <div
                        key={messageIndex}
                        className="rounded-lg border border-line bg-panel-muted px-3 py-2"
                      >
                        <div className="mb-1 text-[11px] uppercase tracking-wide text-ink-faint">
                          {messageRole(message)}
                        </div>
                        <StepContentPreview
                          value={messageContent(message)}
                          emptyLabel="Empty message"
                        />
                      </div>
                    ))}
                    {messages.length > 20 ? (
                      <p className="text-xs text-ink-faint">
                        {messages.length - 20} more messages in raw details.
                      </p>
                    ) : null}
                  </div>
                </div>
                <div>
                  <div className="mb-2 text-xs uppercase tracking-wide text-ink-faint">
                    Model Response
                  </div>
                  <StepContentPreview
                    value={details.response_content}
                    emptyLabel="No assistant response content"
                  />
                  {calls.length > 0 ? <ToolCallPreviewList toolCalls={calls} /> : null}
                </div>
                <JsonDisclosure
                  label="Raw LLM details JSON"
                  value={details}
                  className="bg-panel"
                  contentClassName="bg-panel"
                />
              </div>
            </details>
          );
        })
      )}
    </SectionCard>
  );
}

function ToolTransactions({ trace }: { trace: TraceDetail }) {
  const toolsByCallId = new Map<string, SpanResponse>();
  for (const span of toolSpans(trace)) {
    const id = toolCallId(span);
    if (id) {
      toolsByCallId.set(id, span);
    }
  }

  const transactions = llmSpans(trace).flatMap((llmSpan) =>
    responseToolCalls(llmSpan).map((call) => ({
      llmSpan,
      call,
      toolSpan: call.id ? toolsByCallId.get(call.id) ?? null : null,
    })),
  );
  const unmatchedTools = toolSpans(trace).filter((span) => {
    const id = toolCallId(span);
    return !id || !transactions.some((item) => item.call.id === id);
  });

  return (
    <SectionCard title="Tool Transactions" bodyClassName="space-y-3 px-4 py-4">
      {transactions.length === 0 && unmatchedTools.length === 0 ? (
        <div className="rounded-xl border border-dashed border-line px-4 py-6 text-sm text-ink-muted">
          No tool transactions recorded for this trace.
        </div>
      ) : null}
      {transactions.map(({ llmSpan, call, toolSpan }, index) => {
        const output =
          toolSpan?.tool_details?.content_for_user ??
          toolSpan?.tool_details?.output ??
          toolSpan?.tool_details?.error;
        return (
          <div key={`${call.id ?? "call"}-${index}`} className="rounded-xl border border-line bg-panel px-3 py-3">
            <div className="flex flex-wrap items-center gap-2">
              <Wrench className="h-4 w-4 text-warning" />
              <span className="text-sm font-medium text-foreground">
                {call.function?.name ?? toolSpan?.name ?? "tool_call"}
              </span>
              {call.id ? <MonoText>{call.id}</MonoText> : null}
              <span className="rounded-full border border-line px-2 py-0.5 text-[11px] text-ink-muted">
                from seq {spanSequence(llmSpan) ?? "-"}
              </span>
              <span
                className={`rounded-full border px-2 py-0.5 text-[11px] uppercase tracking-wide ${
                  toolSpan?.status === "error"
                    ? "border-red-500/40 bg-red-500/10 text-red-200"
                    : toolSpan
                      ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-200"
                      : "border-amber-500/40 bg-amber-500/10 text-amber-200"
                }`}
              >
                {toolSpan ? toolSpan.status : "missing result"}
              </span>
              {toolSpan?.duration_ms != null ? (
                <span className="rounded-full border border-line bg-panel-muted px-2 py-1 text-[11px] text-ink-muted">
                  {formatDurationMs(toolSpan.duration_ms)}
                </span>
              ) : null}
            </div>
            <div className="mt-3 grid gap-3 lg:grid-cols-2">
              <div>
                <div className="mb-1 text-xs font-medium text-ink-faint">Arguments</div>
                <StructuredValuePreview value={call.function?.arguments ?? toolSpan?.tool_details?.input_args ?? {}} />
              </div>
              <div>
                <div className="mb-1 text-xs font-medium text-ink-faint">Result</div>
                <StepContentPreview value={output} emptyLabel="No matching tool result" />
              </div>
            </div>
          </div>
        );
      })}
      {unmatchedTools.map((span) => (
        <div key={span.span_id} className="rounded-xl border border-amber-500/30 bg-amber-500/10 px-3 py-3">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <Wrench className="h-4 w-4 text-warning" />
            <span className="text-sm font-medium text-foreground">{toolName(span)}</span>
            <span className="text-xs text-amber-200">Tool result without matched assistant call</span>
          </div>
          <StepContentPreview
            value={span.tool_details?.output ?? span.tool_details?.error}
            emptyLabel="No tool result content"
          />
        </div>
      ))}
    </SectionCard>
  );
}

function ExecutionChain({ trace }: { trace: TraceDetail }) {
  const spanEvents = trace.spans.map((span) => ({
    id: `span:${span.span_id}`,
    time: span.start_time,
    sequence: spanSequence(span),
    kind: span.kind,
    title: span.name,
    summary:
      span.kind === "tool_call"
        ? compact(span.tool_details?.output ?? span.tool_details?.error ?? span.output_preview ?? "")
        : compact(span.output_preview ?? span.llm_details?.response_content ?? span.error_message ?? ""),
    status: span.status,
    duration: span.duration_ms,
  }));
  const decisionEvents = trace.runtime_decisions.map((decision) => ({
    id: `decision:${decision.kind}:${decision.sequence}:${decision.run_id}`,
    time: decision.created_at,
    sequence: decision.sequence,
    kind: decision.kind,
    title: decision.kind.replaceAll("_", " "),
    summary: decision.summary,
    status: decision.kind.includes("failed") ? "error" : "ok",
    duration: null,
  }));
  const events = [...spanEvents, ...decisionEvents].sort((a, b) => {
    const byTime = new Date(a.time ?? 0).getTime() - new Date(b.time ?? 0).getTime();
    if (byTime !== 0) {
      return byTime;
    }
    return (a.sequence ?? 0) - (b.sequence ?? 0);
  });

  return (
    <SectionCard title="Execution Chain" bodyClassName="space-y-2 px-4 py-4">
      {events.length === 0 ? (
        <div className="rounded-xl border border-dashed border-line px-4 py-6 text-sm text-ink-muted">
          No execution chain events available.
        </div>
      ) : (
        <div className="relative space-y-2 before:absolute before:left-[1.05rem] before:top-3 before:h-[calc(100%-1.5rem)] before:w-px before:bg-line">
          {events.map((event) => (
            <div key={event.id} className="relative ml-10 rounded-xl border border-line bg-panel px-3 py-3">
              <div className="absolute -left-10 top-3 flex h-8 w-8 items-center justify-center rounded-full border border-line bg-panel">
                {event.kind === "llm_call" ? (
                  <Zap className="h-3.5 w-3.5 text-success" />
                ) : event.kind === "tool_call" ? (
                  <Wrench className="h-3.5 w-3.5 text-warning" />
                ) : (
                  <GitBranch className="h-3.5 w-3.5 text-ink-muted" />
                )}
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-sm font-medium text-foreground">{event.title}</span>
                <span className="rounded-full border border-line px-2 py-0.5 text-[11px] uppercase tracking-wide text-ink-muted">
                  {event.kind}
                </span>
                <span className="rounded-full border border-line px-2 py-0.5 text-[11px] text-ink-muted">
                  seq {event.sequence ?? "-"}
                </span>
                {event.duration != null ? (
                  <span className="rounded-full border border-line bg-panel-muted px-2 py-1 text-[11px] text-ink-muted">
                    {formatDurationMs(event.duration)}
                  </span>
                ) : null}
                <span className="ml-auto text-xs text-ink-muted">{formatLocalDateTime(event.time)}</span>
              </div>
              {event.summary ? (
                <p className="mt-2 text-sm leading-5 text-ink-muted">{event.summary}</p>
              ) : null}
            </div>
          ))}
        </div>
      )}
    </SectionCard>
  );
}

export function TraceDiagnostics({ trace }: { trace: TraceDetail }) {
  return (
    <div className="space-y-6">
      <DiagnosticsOverview trace={trace} />
      <ExecutionChain trace={trace} />
      <ToolTransactions trace={trace} />
      <LlmCallInspector trace={trace} />
    </div>
  );
}

export default TraceDiagnostics;
