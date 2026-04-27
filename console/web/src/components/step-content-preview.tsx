"use client";

import { Wrench } from "lucide-react";

import { JsonDisclosure } from "@/components/json-disclosure";
import type { ToolCallPayload } from "@/lib/api";
import { cn } from "@/lib/utils";

const JSON_PARSE_FAILED = Symbol("json_parse_failed");

function tryParseJson(value: string): unknown | typeof JSON_PARSE_FAILED {
  try {
    return JSON.parse(value);
  } catch {
    return JSON_PARSE_FAILED;
  }
}

function displayValue(value: unknown): unknown {
  if (typeof value !== "string") {
    return value;
  }
  const parsed = tryParseJson(value);
  return parsed !== JSON_PARSE_FAILED ? parsed : value;
}

export function stringifyPretty(value: unknown): string {
  if (value === undefined) {
    return "undefined";
  }
  if (typeof value === "string") {
    const parsed = tryParseJson(value);
    if (parsed !== JSON_PARSE_FAILED) {
      return JSON.stringify(parsed, null, 2);
    }
    return value;
  }
  return JSON.stringify(value, null, 2);
}

export function contentText(value: unknown): string | null {
  if (typeof value === "string") {
    return value.trim().length > 0 ? value : null;
  }
  if (value === null || value === undefined) {
    return null;
  }
  if (Array.isArray(value)) {
    const parts = value
      .map((item) =>
        item && typeof item === "object" && "text" in item && typeof item.text === "string"
          ? item.text
          : "",
      )
      .filter(Boolean);
    return parts.length > 0 ? parts.join("\n") : null;
  }
  return null;
}

function compactText(value: string, maxLength = 220): string {
  const normalized = value.replace(/\s+/g, " ").trim();
  if (normalized.length <= maxLength) {
    return normalized;
  }
  return `${normalized.slice(0, maxLength - 3)}...`;
}

function inlineValue(value: unknown): string {
  if (value === null) {
    return "null";
  }
  if (value === undefined) {
    return "undefined";
  }
  if (typeof value === "string") {
    return compactText(value);
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (Array.isArray(value)) {
    if (value.length === 0) {
      return "[]";
    }
    if (value.every((item) => ["string", "number", "boolean"].includes(typeof item))) {
      return compactText(value.join(", "));
    }
    return `${value.length} items`;
  }
  if (typeof value === "object") {
    return `${Object.keys(value).length} fields`;
  }
  return String(value);
}

export function StructuredValuePreview({
  value,
  className,
}: {
  value: unknown;
  className?: string;
}) {
  if (Array.isArray(value)) {
    if (value.length === 0) {
      return <div className="text-sm text-ink-faint">Empty list</div>;
    }
    return (
      <div className={cn("space-y-2 text-sm", className)}>
        {value.slice(0, 6).map((item, index) => (
          <div key={index} className="rounded-md bg-panel-muted px-3 py-2 text-ink-soft break-words">
            {inlineValue(item)}
          </div>
        ))}
        {value.length > 6 ? (
          <div className="text-xs text-ink-faint">{value.length - 6} more items in raw JSON</div>
        ) : null}
      </div>
    );
  }

  if (!value || typeof value !== "object") {
    return (
      <div className={cn("text-sm leading-6 text-ink-soft whitespace-pre-wrap break-words", className)}>
        {inlineValue(value)}
      </div>
    );
  }

  const entries = Object.entries(value).filter(([, item]) => item !== null && item !== undefined);
  if (entries.length === 0) {
    return <div className="text-sm text-ink-faint">Empty object</div>;
  }

  return (
    <dl className={cn("grid gap-2 text-sm", className)}>
      {entries.slice(0, 8).map(([key, item]) => (
        <div key={key} className="grid gap-1 rounded-md bg-panel-muted px-3 py-2 sm:grid-cols-[9rem_1fr]">
          <dt className="font-mono text-xs text-ink-faint">{key}</dt>
          <dd className="min-w-0 text-ink-soft break-words">{inlineValue(item)}</dd>
        </div>
      ))}
      {entries.length > 8 ? (
        <div className="text-xs text-ink-faint">{entries.length - 8} more fields in raw JSON</div>
      ) : null}
    </dl>
  );
}

export function StepContentPreview({
  value,
  emptyLabel,
}: {
  value: unknown;
  emptyLabel: string;
}) {
  const text = contentText(value);
  if (text) {
    return (
      <div className="max-h-96 overflow-auto text-sm leading-6 text-ink-soft whitespace-pre-wrap break-words">
        {text}
      </div>
    );
  }

  if (value !== null && value !== undefined) {
    return (
      <div className="max-h-96 overflow-auto">
        <StructuredValuePreview value={value} />
      </div>
    );
  }

  return <div className="text-sm text-ink-faint">{emptyLabel}</div>;
}

export function ToolCallPreviewList({ toolCalls }: { toolCalls: ToolCallPayload[] }) {
  if (toolCalls.length === 0) {
    return null;
  }

  return (
    <div className="mt-3 space-y-2">
      {toolCalls.map((toolCall, index) => {
        const toolName = toolCall.function?.name || "tool_call";
        const args = toolCall.function?.arguments ?? toolCall;
        const parsedArgs = displayValue(args);
        return (
          <div
            key={`${toolName}-${toolCall.id ?? index}`}
            className="rounded-lg border border-line bg-panel-muted"
          >
            <div className="flex items-center gap-2 border-b border-line px-3 py-2 text-xs">
              <Wrench className="h-3.5 w-3.5 text-warning" />
              <span className="font-medium text-ink-soft">{toolName}</span>
              {toolCall.id ? (
                <span className="font-mono text-[11px] text-ink-faint">{toolCall.id}</span>
              ) : null}
            </div>
            <div className="px-3 py-2">
              <StructuredValuePreview value={parsedArgs} />
            </div>
            <JsonDisclosure
              label="Raw arguments"
              value={parsedArgs}
              className="m-2 mt-0 bg-panel"
              contentClassName="bg-panel"
            />
          </div>
        );
      })}
    </div>
  );
}

export function RawJsonBlock({
  label,
  value,
  className,
}: {
  label: string;
  value: unknown;
  className?: string;
}) {
  return <JsonDisclosure className={className} label={label} value={value} />;
}
