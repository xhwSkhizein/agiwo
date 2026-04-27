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
import { TraceStatusBadge } from "@/components/trace-status-badge";
import type { RuntimeDecisionEvent, SpanResponse, ToolCallPayload, TraceDetail } from "@/lib/api";
import { formatDurationMs, formatTokenCount } from "@/lib/metrics";
import { formatLocalDateTime } from "@/lib/time";

interface SpanChainEvent {
  id: string;
  time: string | null;
  sequence: number | null;
  type: "span";
  span: SpanResponse;
}

interface DecisionChainEvent {
  id: string;
  time: string | null;
  sequence: number | null;
  type: "decision";
  decision: RuntimeDecisionEvent;
}

type ChainEvent = SpanChainEvent | DecisionChainEvent;

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

function toolName(span: SpanResponse): string {
  return stringValue(span.tool_details?.tool_name) ?? span.name;
}

function responseToolCalls(span: SpanResponse): ToolCallPayload[] {
  const calls = span.llm_details?.response_tool_calls;
  return Array.isArray(calls) ? (calls as ToolCallPayload[]) : [];
}

function traceRiskSignals(trace: TraceDetail): string[] {
  const signals: string[] = [];
  const errors = trace.spans.filter((span) => span.status === "error" || span.error_message);
  if (errors.length > 0) {
    signals.push(`${errors.length} errored span${errors.length === 1 ? "" : "s"}`);
  }

  const slowest = [...trace.spans]
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

function pluralize(count: number, singular: string, plural: string): string {
  return count === 1 ? singular : plural;
}

function eventIcon(event: ChainEvent) {
  if (event.type === "decision") {
    return <GitBranch className="h-3.5 w-3.5 text-ink-muted" />;
  }
  if (event.span.kind === "llm_call") {
    return <Zap className="h-3.5 w-3.5 text-success" />;
  }
  if (event.span.kind === "tool_call") {
    return <Wrench className="h-3.5 w-3.5 text-warning" />;
  }
  return <Bot className="h-3.5 w-3.5 text-accent" />;
}

function ChainHeader({ event }: { event: ChainEvent }) {
  const title =
    event.type === "decision"
      ? event.decision.kind.replaceAll("_", " ")
      : event.span.kind === "tool_call"
        ? toolName(event.span)
        : event.span.name;
  const status = event.type === "decision" ? null : event.span.status;
  const duration = event.type === "decision" ? null : event.span.duration_ms;

  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-sm font-medium text-foreground">{title}</span>
      {status ? <TraceStatusBadge status={status} /> : null}
      {event.sequence !== null ? (
        <span className="rounded-full border border-line px-2 py-0.5 text-[11px] text-ink-muted">
          seq {event.sequence}
        </span>
      ) : null}
      {duration !== null ? (
        <span className="rounded-full border border-line bg-panel-muted px-2 py-0.5 text-[11px] text-ink-muted">
          {formatDurationMs(duration)}
        </span>
      ) : null}
      <span className="ml-auto text-xs text-ink-faint">{formatLocalDateTime(event.time)}</span>
    </div>
  );
}

function LlmEventBody({ span }: { span: SpanResponse }) {
  const details = span.llm_details ?? {};
  const messages = Array.isArray(details.messages) ? details.messages : [];
  const tools = Array.isArray(details.tools) ? details.tools : [];
  const calls = responseToolCalls(span);

  return (
    <div className="space-y-3">
      <div>
        <div className="mb-1 text-xs font-medium text-ink-faint">Assistant message</div>
        <StepContentPreview
          value={details.response_content ?? span.output_preview}
          emptyLabel="No assistant response content"
        />
      </div>
      {calls.length > 0 ? <ToolCallPreviewList toolCalls={calls} /> : null}
      <details className="rounded-lg border border-line bg-panel-muted">
        <summary className="cursor-pointer list-none px-3 py-2 text-xs text-ink-muted">
          Prompt context ({messages.length}{" "}
          {pluralize(messages.length, "message", "messages")}, {tools.length}{" "}
          {pluralize(tools.length, "tool", "tools")} available)
        </summary>
        <div className="max-h-96 space-y-2 overflow-auto border-t border-line px-3 py-3">
          {messages.slice(0, 12).map((message, index) => (
            <div key={index} className="rounded-lg border border-line bg-panel px-3 py-2">
              <div className="mb-1 text-[11px] uppercase tracking-wide text-ink-faint">
                {messageRole(message)}
              </div>
              <StepContentPreview value={messageContent(message)} emptyLabel="Empty message" />
            </div>
          ))}
          {messages.length > 12 ? (
            <p className="text-xs text-ink-faint">
              {messages.length - 12} more messages in raw LLM details.
            </p>
          ) : null}
        </div>
      </details>
      <JsonDisclosure
        label="Raw LLM details JSON"
        value={details}
        className="bg-panel"
        contentClassName="bg-panel"
      />
    </div>
  );
}

function ToolEventBody({ span }: { span: SpanResponse }) {
  const details = span.tool_details ?? {};
  const output = details.content_for_user ?? details.output ?? details.error ?? span.output_preview;

  return (
    <div className="grid gap-3 lg:grid-cols-2">
      <div>
        <div className="mb-1 text-xs font-medium text-ink-faint">Arguments</div>
        <StructuredValuePreview value={details.input_args ?? {}} />
      </div>
      <div>
        <div className="mb-1 text-xs font-medium text-ink-faint">Result</div>
        <StepContentPreview value={output} emptyLabel="No tool result content" />
      </div>
      <JsonDisclosure
        label="Raw tool details JSON"
        value={details}
        className="bg-panel lg:col-span-2"
        contentClassName="bg-panel"
      />
    </div>
  );
}

function DecisionBody({ decision }: { decision: RuntimeDecisionEvent }) {
  return (
    <div className="space-y-3">
      <p className="text-sm leading-6 text-foreground">{decision.summary}</p>
      <div className="flex flex-wrap gap-2 text-[11px] text-ink-muted">
        <span>run {decision.run_id}</span>
        <span>agent {decision.agent_id}</span>
      </div>
      <JsonDisclosure
        label="Decision details JSON"
        value={decision.details}
        className="bg-panel"
        contentClassName="bg-panel"
      />
    </div>
  );
}

function SpanFallbackBody({ span }: { span: SpanResponse }) {
  const preview = span.output_preview ?? span.input_preview ?? span.error_message;
  return (
    <div className="space-y-3">
      <StepContentPreview value={preview} emptyLabel="No span preview" />
      {span.error_message ? (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-200">
          {span.error_message}
        </div>
      ) : null}
      <JsonDisclosure
        label="Span attributes JSON"
        value={span.attributes}
        className="bg-panel"
        contentClassName="bg-panel"
      />
    </div>
  );
}

function ChainEventRow({ event }: { event: ChainEvent }) {
  return (
    <div className="relative ml-10 rounded-xl border border-line bg-panel px-3 py-3">
      <div className="absolute -left-10 top-3 flex h-8 w-8 items-center justify-center rounded-full border border-line bg-panel">
        {eventIcon(event)}
      </div>
      <div className="space-y-3">
        <ChainHeader event={event} />
        {event.type === "decision" ? (
          <DecisionBody decision={event.decision} />
        ) : event.span.kind === "llm_call" && event.span.llm_details ? (
          <LlmEventBody span={event.span} />
        ) : event.span.kind === "tool_call" && event.span.tool_details ? (
          <ToolEventBody span={event.span} />
        ) : (
          <SpanFallbackBody span={event.span} />
        )}
      </div>
    </div>
  );
}

function buildChainEvents(trace: TraceDetail): ChainEvent[] {
  const spanEvents = trace.spans.map((span) => ({
    id: `span:${span.span_id}`,
    time: span.start_time,
    sequence: spanSequence(span),
    type: "span" as const,
    span,
  }));
  const decisionEvents = trace.runtime_decisions.map((decision) => ({
    id: `decision:${decision.kind}:${decision.sequence}:${decision.run_id}`,
    time: decision.created_at,
    sequence: decision.sequence,
    type: "decision" as const,
    decision,
  }));

  const eventTime = (time: string | null): number => {
    if (!time) {
      return Number.MAX_SAFE_INTEGER;
    }
    const parsed = new Date(time).getTime();
    return Number.isFinite(parsed) ? parsed : Number.MAX_SAFE_INTEGER;
  };

  return [...spanEvents, ...decisionEvents].sort((a, b) => {
    const byTime = eventTime(a.time) - eventTime(b.time);
    if (byTime !== 0) {
      return byTime;
    }
    return (a.sequence ?? Number.MAX_SAFE_INTEGER) - (b.sequence ?? Number.MAX_SAFE_INTEGER);
  });
}

function DiagnosticSummaryStrip({ trace }: { trace: TraceDetail }) {
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
    <div className="grid gap-3 border-b border-line bg-panel-muted px-4 py-3 lg:grid-cols-3">
      <div className="space-y-2">
        <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-ink-faint">
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
      <div>
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
      <div>
        <div className="mb-2 flex items-center gap-2 text-xs uppercase tracking-wide text-ink-faint">
          <Zap className="h-3.5 w-3.5" />
          Heaviest LLM Call
        </div>
        <p className="text-sm text-foreground">
          {mostExpensive ? (
            <>
              {mostExpensive.name}{" "}
              <MonoText>
                {formatTokenCount(numberValue(mostExpensive.metrics["tokens.total"]) ?? 0)}
              </MonoText>
            </>
          ) : (
            "No LLM calls"
          )}
        </p>
      </div>
    </div>
  );
}

export function TraceDiagnostics({ trace }: { trace: TraceDetail }) {
  const events = buildChainEvents(trace);

  return (
    <SectionCard
      title="Agent Execution Diagnostics"
      headerClassName="border-b border-line px-4 py-3"
    >
      <DiagnosticSummaryStrip trace={trace} />
      <div className="px-4 py-4">
        {events.length === 0 ? (
          <div className="rounded-xl border border-dashed border-line px-4 py-6 text-sm text-ink-muted">
            No execution chain events available.
          </div>
        ) : (
          <div className="relative space-y-3 before:absolute before:left-[1.05rem] before:top-3 before:h-[calc(100%-1.5rem)] before:w-px before:bg-line">
            {events.map((event) => (
              <ChainEventRow key={event.id} event={event} />
            ))}
          </div>
        )}
      </div>
    </SectionCard>
  );
}

export default TraceDiagnostics;
