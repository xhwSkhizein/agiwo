"use client";

import { Bot, User, Wrench } from "lucide-react";

import {
  RawJsonBlock,
  StepContentPreview,
  ToolCallPreviewList,
  contentText,
} from "@/components/step-content-preview";
import { TokenMetricsBadges } from "@/components/token-metrics-badges";
import { UserInputDetail } from "@/components/user-input-detail";
import type { StepResponse } from "@/lib/api";
import { parseGenericMetrics } from "@/lib/metrics";

export function SessionStepCard({ step }: { step: StepResponse }) {
  const isUser = step.role === "user";
  const isAssistant = step.role === "assistant";
  const isTool = step.role === "tool";
  const metrics = parseGenericMetrics(step.metrics ?? undefined);

  const hasStructuredUserInput =
    isUser && step.user_input !== null && step.user_input !== undefined;
  const hasCondensed = isTool && typeof step.condensed_content === "string";
  const assistantContent =
    step.content_for_user ?? contentText(step.content) ?? step.content;
  const toolContent = step.condensed_content ?? step.content_for_user ?? step.content;
  const displayContent = isAssistant
    ? assistantContent
    : isTool
      ? toolContent
      : step.content;
  const originalContent = hasCondensed ? step.content : null;

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

      {!hasStructuredUserInput && (
        <StepContentPreview
          value={displayContent}
          emptyLabel={
            isAssistant
              ? "No assistant message content"
              : isTool
                ? "No tool result content"
                : "No step content"
          }
        />
      )}

      {originalContent !== null && originalContent !== undefined && (
        <RawJsonBlock className="mt-3" label="Original result" value={originalContent} />
      )}

      {step.tool_calls && step.tool_calls.length > 0 && (
        <ToolCallPreviewList toolCalls={step.tool_calls} />
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

      <RawJsonBlock className="mt-3" label="Raw step JSON" value={step} />
    </div>
  );
}

export default SessionStepCard;
