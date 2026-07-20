"use client";

import { useMemo, useState } from "react";

import { JsonDisclosure } from "@/components/json-disclosure";
import { MonoText } from "@/components/mono-text";
import {
  StepContentPreview,
  StructuredValuePreview,
  ToolCallPreviewList,
  contentText,
  stringifyPretty,
} from "@/components/step-content-preview";
import { TraceStatusBadge } from "@/components/trace-status-badge";
import type { SpanResponse, ToolCallPayload, TraceDetail } from "@/lib/api";
import { formatDurationMs, formatTokenCount } from "@/lib/metrics";
import { cn } from "@/lib/utils";

type DetailLayer = "summary" | "llm" | "tools" | "raw";

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function stringValue(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function numberValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function compact(value: unknown, maxLength = 220): string {
  const text =
    typeof value === "string" ? value : contentText(value) ?? stringifyPretty(value);
  const normalized = text.replace(/\s+/g, " ").trim();
  if (normalized.length <= maxLength) {
    return normalized;
  }
  return `${normalized.slice(0, maxLength - 3)}...`;
}

function barKind(kind: string): "agent" | "llm" | "tool" | "other" {
  if (kind === "llm_call") return "llm";
  if (kind === "tool_call") return "tool";
  if (kind === "agent") return "agent";
  return "other";
}

function toolName(span: SpanResponse): string {
  return stringValue(span.tool_details?.tool_name) ?? span.name;
}

function responseToolCalls(span: SpanResponse): ToolCallPayload[] {
  const calls = span.llm_details?.response_tool_calls;
  return Array.isArray(calls) ? (calls as ToolCallPayload[]) : [];
}

function sortSpans(spans: SpanResponse[]): SpanResponse[] {
  return [...spans].sort((a, b) => {
    const aTime = a.start_time ? Date.parse(a.start_time) : Number.POSITIVE_INFINITY;
    const bTime = b.start_time ? Date.parse(b.start_time) : Number.POSITIVE_INFINITY;
    if (aTime !== bTime) {
      return aTime - bTime;
    }
    return a.depth - b.depth || a.name.localeCompare(b.name);
  });
}

type WindowedSpan = {
  span: SpanResponse;
  leftPct: number;
  widthPct: number;
};

function windowSpans(spans: SpanResponse[], totalDurationMs: number | null): WindowedSpan[] {
  const ordered = sortSpans(spans);
  if (ordered.length === 0) {
    return [];
  }
  const starts = ordered
    .map((span) => (span.start_time ? Date.parse(span.start_time) : null))
    .filter((value): value is number => value !== null);
  const origin = starts.length > 0 ? Math.min(...starts) : 0;
  const inferredEnd = Math.max(
    ...ordered.map((span) => {
      const start = span.start_time ? Date.parse(span.start_time) : origin;
      return start + (span.duration_ms ?? 0);
    }),
    origin + 1,
  );
  const total =
    totalDurationMs && totalDurationMs > 0
      ? totalDurationMs
      : Math.max(inferredEnd - origin, 1);

  return ordered.map((span) => {
    const start = span.start_time ? Date.parse(span.start_time) : origin;
    const duration = Math.max(span.duration_ms ?? 0, 1);
    const leftPct = Math.max(0, Math.min(100, ((start - origin) / total) * 100));
    const widthPct = Math.max(0.4, Math.min(100 - leftPct, (duration / total) * 100));
    return { span, leftPct, widthPct };
  });
}

const layerHints: Record<DetailLayer, string> = {
  summary: "Narrative summary: what this span means in the run.",
  llm: "LLM layer: phase, tokens, output preview.",
  tools: "Tool IO layer: args and condensed results.",
  raw: "Raw JSON: collapsed by default; expand only when debugging.",
};

type TraceFlameExplorerProps = {
  trace: TraceDetail;
};

export function TraceFlameExplorer({ trace }: TraceFlameExplorerProps) {
  const windowed = useMemo(
    () => windowSpans(trace.spans, trace.duration_ms),
    [trace.duration_ms, trace.spans],
  );
  const [selectedId, setSelectedId] = useState<string | null>(
    () => trace.root_span_id ?? windowed[0]?.span.span_id ?? null,
  );
  const [layer, setLayer] = useState<DetailLayer>("summary");

  const selected =
    windowed.find((item) => item.span.span_id === selectedId)?.span ??
    windowed[0]?.span ??
    null;

  const selectSpan = (spanId: string) => {
    setSelectedId(spanId);
    if (layer === "raw") {
      setLayer("summary");
    }
  };

  const totalLabel = formatDurationMs(trace.duration_ms || 0);

  return (
    <div className="grid min-h-[28rem] gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(18rem,22rem)]">
      <section className="flex min-h-0 flex-col overflow-hidden rounded-xl border border-line bg-panel">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line bg-panel-muted px-3 py-2.5">
          <div>
            <h2 className="text-sm font-semibold">Execution timeline</h2>
            <p className="text-xs text-ink-muted">
              Depth via indent · click a span for details
            </p>
          </div>
          <span className="font-mono text-[11px] text-ink-faint">
            max depth {trace.max_depth} · {totalLabel}
          </span>
        </div>
        <div className="min-h-0 flex-1 overflow-auto px-3 py-3">
          {windowed.length === 0 ? (
            <p className="py-10 text-center text-sm text-ink-muted">No spans in this trace.</p>
          ) : (
            <>
              <div className="sticky top-0 z-[1] mb-2 grid grid-cols-[9rem_minmax(0,1fr)_3.25rem] gap-2 border-b border-line bg-panel pb-1.5 text-[10px] uppercase tracking-wide text-ink-faint">
                <span>Span</span>
                <span className="flex justify-between font-mono normal-case tracking-normal">
                  <span>0</span>
                  <span>{totalLabel}</span>
                </span>
                <span>Dur</span>
              </div>
              <div className="space-y-0.5">
                {windowed.map(({ span, leftPct, widthPct }) => {
                  const kind = barKind(span.kind);
                  const selectedRow = span.span_id === selected?.span_id;
                  return (
                    <button
                      key={span.span_id}
                      type="button"
                      onClick={() => selectSpan(span.span_id)}
                      className={cn(
                        "grid w-full grid-cols-[9rem_minmax(0,1fr)_3.25rem] items-center gap-2 rounded-md border px-1 py-1 text-left transition-colors",
                        selectedRow
                          ? "border-accent/40 bg-accent/10"
                          : "border-transparent hover:border-line hover:bg-panel-muted",
                      )}
                    >
                      <span
                        className="truncate font-mono text-[11px] text-ink-muted"
                        style={{ paddingLeft: `${Math.min(span.depth, 6) * 10}px` }}
                        title={span.name}
                      >
                        <span
                          className={cn(
                            "mr-1.5 inline-block h-1.5 w-1.5 rounded-full align-middle",
                            kind === "agent" && "bg-blue-400",
                            kind === "llm" && "bg-violet-300",
                            kind === "tool" && "bg-emerald-300",
                            kind === "other" && "bg-zinc-500",
                          )}
                        />
                        {span.name}
                      </span>
                      <span className="relative h-5 overflow-hidden rounded border border-line bg-background">
                        <span
                          className={cn(
                            "absolute bottom-0.5 top-0.5 rounded-sm",
                            kind === "agent" && "bg-blue-400/70",
                            kind === "llm" && "bg-violet-300/75",
                            kind === "tool" && "bg-emerald-300/75",
                            kind === "other" && "bg-zinc-500/70",
                            selectedRow && "ring-1 ring-accent",
                          )}
                          style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
                        />
                      </span>
                      <span className="font-mono text-[10px] text-ink-faint">
                        {formatDurationMs(span.duration_ms || 0)}
                      </span>
                    </button>
                  );
                })}
              </div>
              <div className="mt-3 flex flex-wrap gap-3 text-[11px] text-ink-faint">
                <span>agent</span>
                <span>llm_call</span>
                <span>tool_call</span>
              </div>
            </>
          )}
        </div>
      </section>

      <aside className="flex max-h-[min(72vh,44rem)] min-h-0 flex-col overflow-hidden rounded-xl border border-line bg-panel lg:sticky lg:top-4">
        <div className="space-y-2 border-b border-line bg-panel-muted px-3 py-2.5">
          <div className="min-w-0">
            <h2 className="truncate text-sm font-semibold">
              {selected?.name ?? "Span detail"}
            </h2>
            <MonoText className="text-[11px] text-ink-faint">
              {selected
                ? `${selected.span_id} · depth ${selected.depth} · ${selected.kind}`
                : "—"}
            </MonoText>
          </div>
          <div className="inline-flex overflow-hidden rounded-md border border-line">
            {(["summary", "llm", "tools", "raw"] as DetailLayer[]).map((item) => (
              <button
                key={item}
                type="button"
                onClick={() => setLayer(item)}
                className={cn(
                  "border-l border-line px-2.5 py-1 text-xs capitalize first:border-l-0",
                  layer === item
                    ? "bg-panel-strong text-foreground"
                    : "text-ink-muted hover:bg-panel",
                )}
              >
                {item}
              </button>
            ))}
          </div>
        </div>
        <div className="min-h-0 flex-1 overflow-auto px-3 py-3">
          <p className="mb-3 text-xs text-ink-faint">{layerHints[layer]}</p>
          {!selected ? (
            <p className="text-sm text-ink-muted">Select a span on the timeline.</p>
          ) : (
            <SpanDetailBody span={selected} layer={layer} />
          )}
        </div>
      </aside>
    </div>
  );
}

function SpanDetailBody({
  span,
  layer,
}: {
  span: SpanResponse;
  layer: DetailLayer;
}) {
  if (layer === "summary") {
    return (
      <dl className="grid grid-cols-[6.5rem_minmax(0,1fr)] gap-x-3 gap-y-2 text-sm">
        <dt className="text-ink-faint">Status</dt>
        <dd>
          <TraceStatusBadge status={span.status} />
        </dd>
        <dt className="text-ink-faint">Duration</dt>
        <dd className="text-ink-muted">{formatDurationMs(span.duration_ms || 0)}</dd>
        <dt className="text-ink-faint">Run</dt>
        <dd>
          <MonoText className="text-xs">{span.run_id || "—"}</MonoText>
        </dd>
        <dt className="text-ink-faint">Input</dt>
        <dd className="text-ink-muted">{compact(span.input_preview || "—")}</dd>
        <dt className="text-ink-faint">Output</dt>
        <dd className="text-ink-muted">{compact(span.output_preview || "—")}</dd>
        {span.error_message ? (
          <>
            <dt className="text-ink-faint">Error</dt>
            <dd className="text-red-300">{span.error_message}</dd>
          </>
        ) : null}
      </dl>
    );
  }

  if (layer === "llm") {
    if (span.kind !== "llm_call" || !span.llm_details) {
      return (
        <p className="text-sm text-ink-muted">
          This span is not an LLM call. Pick an llm_call bar, or open Tools for tool IO.
        </p>
      );
    }
    const details = span.llm_details;
    const phase = stringValue(details.phase) ?? stringValue(details.model_call_phase);
    const attempt = numberValue(details.attempt_no);
    return (
      <div className="space-y-3 text-sm">
        <dl className="grid grid-cols-[6.5rem_minmax(0,1fr)] gap-x-3 gap-y-2">
          <dt className="text-ink-faint">Phase</dt>
          <dd className="font-mono text-xs text-ink-muted">{phase || "assistant"}</dd>
          <dt className="text-ink-faint">Attempt</dt>
          <dd className="text-ink-muted">{attempt ?? 1}</dd>
          <dt className="text-ink-faint">Tokens</dt>
          <dd className="text-ink-muted">
            in {formatTokenCount(span.metrics["tokens.input"] ?? 0)} · out{" "}
            {formatTokenCount(span.metrics["tokens.output"] ?? 0)}
          </dd>
          <dt className="text-ink-faint">Model</dt>
          <dd className="text-ink-muted">
            {stringValue(span.metrics.model) || stringValue(details.model) || "—"}
          </dd>
        </dl>
        {span.output_preview ? (
          <div>
            <p className="mb-1 text-xs text-ink-faint">Output preview</p>
            <p className="whitespace-pre-wrap text-ink-muted">{compact(span.output_preview, 400)}</p>
          </div>
        ) : null}
        {responseToolCalls(span).length > 0 ? (
          <ToolCallPreviewList toolCalls={responseToolCalls(span)} />
        ) : null}
      </div>
    );
  }

  if (layer === "tools") {
    if (span.kind !== "tool_call" || !span.tool_details) {
      if (span.kind === "agent") {
        return (
          <p className="text-sm text-ink-muted">
            Agent spans aggregate child tools. Click a tool_call bar to inspect IO.
          </p>
        );
      }
      return (
        <p className="text-sm text-ink-muted">
          This span is not a tool call. Open LLM for model turns, or pick a tool bar.
        </p>
      );
    }
    const args = span.tool_details.arguments ?? span.tool_details.args;
    const result = span.tool_details.result ?? span.tool_details.output;
    return (
      <div className="space-y-3 text-sm">
        <dl className="grid grid-cols-[6.5rem_minmax(0,1fr)] gap-x-3 gap-y-2">
          <dt className="text-ink-faint">Tool</dt>
          <dd className="font-mono text-xs text-emerald-300/90">{toolName(span)}</dd>
          <dt className="text-ink-faint">Status</dt>
          <dd className="text-ink-muted">{span.status}</dd>
        </dl>
        {args !== undefined ? (
          <div>
            <p className="mb-1 text-xs text-ink-faint">Args</p>
            {isRecord(args) || Array.isArray(args) ? (
              <StructuredValuePreview value={args} />
            ) : (
              <p className="whitespace-pre-wrap text-ink-muted">{compact(args, 500)}</p>
            )}
          </div>
        ) : null}
        {result !== undefined ? (
          <div>
            <p className="mb-1 text-xs text-ink-faint">Result</p>
            <StepContentPreview content={result} />
          </div>
        ) : null}
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <JsonDisclosure
        label="Span JSON"
        value={{
          span_id: span.span_id,
          parent_span_id: span.parent_span_id,
          depth: span.depth,
          kind: span.kind,
          name: span.name,
          status: span.status,
          duration_ms: span.duration_ms,
          attributes: span.attributes,
          llm_details: span.llm_details,
          tool_details: span.tool_details,
          metrics: span.metrics,
        }}
      />
      <p className="text-xs text-ink-faint">
        Raw stays collapsed unless you expand it for debugging.
      </p>
    </div>
  );
}

export default TraceFlameExplorer;
