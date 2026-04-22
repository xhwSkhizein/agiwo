"use client";

import { useCallback, useState } from "react";
import { Bot, ChevronDown, ChevronRight, User, Wrench } from "lucide-react";

import { TokenMetricsBadges } from "@/components/token-metrics-badges";
import { UserInputDetail } from "@/components/user-input-detail";
import type { StepResponse, ToolCallPayload } from "@/lib/api";
import { parseGenericMetrics } from "@/lib/metrics";

function StepCardOriginalToggle({ content }: { content: string }) {
  const [expanded, setExpanded] = useState(false);
  const toggle = useCallback(() => setExpanded((prev) => !prev), []);
  return (
    <div className="mt-1">
      <button
        type="button"
        onClick={toggle}
        className="inline-flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
      >
        {expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        {expanded ? "Hide original result" : "View original result"}
      </button>
      {expanded && (
        <div className="mt-1 max-h-64 overflow-auto rounded bg-zinc-800/50 px-3 py-2 text-xs text-zinc-400 whitespace-pre-wrap">
          {content}
        </div>
      )}
    </div>
  );
}

export function SessionStepCard({ step }: { step: StepResponse }) {
  const isUser = step.role === "user";
  const isAssistant = step.role === "assistant";
  const isTool = step.role === "tool";
  const metrics = parseGenericMetrics(step.metrics ?? undefined);

  const getToolLabel = (toolCall: ToolCallPayload): string =>
    toolCall.function?.name || "tool_call";
  const getToolArgs = (toolCall: ToolCallPayload): string =>
    toolCall.function?.arguments || JSON.stringify(toolCall);

  const hasStructuredUserInput =
    isUser && step.user_input !== null && step.user_input !== undefined;
  const hasCondensed = isTool && typeof step.condensed_content === "string";
  const displayContent = hasCondensed ? step.condensed_content : step.content;
  const originalContent =
    hasCondensed && typeof step.content === "string" ? step.content : null;

  return (
    <div
      className={`rounded-lg border p-4 ${
        isUser
          ? "border-blue-800/50 bg-blue-950/20"
          : isTool
            ? "border-amber-800/50 bg-amber-950/20"
            : "border-zinc-800 bg-zinc-900"
      }`}
    >
      <div className="mb-2 flex items-center gap-2">
        {isUser && <User className="h-4 w-4 text-blue-400" />}
        {isAssistant && <Bot className="h-4 w-4 text-green-400" />}
        {isTool && <Wrench className="h-4 w-4 text-amber-400" />}
        <span className="text-xs font-medium uppercase tracking-wide text-zinc-400">
          {step.role}
          {isTool && step.name && ` — ${step.name}`}
          {step.agent_id && ` — ${step.agent_id}`}
        </span>
        <span className="ml-auto text-xs text-zinc-600">#{step.sequence}</span>
      </div>

      {step.reasoning_content && (
        <div className="mb-2 max-h-48 overflow-auto rounded bg-zinc-800/50 px-3 py-2 text-xs text-zinc-400 whitespace-pre-wrap">
          <span className="font-medium text-zinc-500">Thinking: </span>
          {step.reasoning_content}
        </div>
      )}

      {hasStructuredUserInput && (
        <div className="max-h-96 overflow-auto">
          <UserInputDetail input={step.user_input} maxTextLength={2000} />
        </div>
      )}

      {!hasStructuredUserInput && Boolean(displayContent) && (
        <div className="max-h-96 overflow-auto">
          <div className="text-sm whitespace-pre-wrap break-words">
            {typeof displayContent === "string"
              ? displayContent
              : JSON.stringify(displayContent, null, 2)}
          </div>
        </div>
      )}

      {originalContent && <StepCardOriginalToggle content={originalContent} />}

      {step.tool_calls && step.tool_calls.length > 0 && (
        <div className="mt-2 space-y-1">
          {step.tool_calls.map((toolCall, index) => (
            <div
              key={index}
              className="max-h-48 overflow-auto rounded bg-zinc-800/50 px-3 py-2 font-mono text-xs"
            >
              <span className="text-amber-400">{getToolLabel(toolCall)}</span>
              <span className="ml-2 text-zinc-500">{getToolArgs(toolCall)}</span>
            </div>
          ))}
        </div>
      )}

      {step.metrics && (
        <div className="mt-3">
          <TokenMetricsBadges
            metrics={metrics}
            showDuration={true}
            showModelName={true}
            modelName={step.metrics?.model_name ?? null}
          />
        </div>
      )}
    </div>
  );
}

export default SessionStepCard;
