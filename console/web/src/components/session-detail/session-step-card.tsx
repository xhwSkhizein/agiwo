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
      className={`rounded-lg border ${
        isUser
          ? "border-blue-800/50 bg-blue-950/20"
          : isTool
            ? "border-amber-800/50 bg-amber-950/20"
            : "border-zinc-800 bg-zinc-900"
      }`}
    >
      <div className="border-b border-line px-4 py-2.5">
        <div className="flex items-center gap-2">
          {isUser && <User className="h-4 w-4 text-blue-400" />}
          {isAssistant && <Bot className="h-4 w-4 text-green-400" />}
          {isTool && <Wrench className="h-4 w-4 text-amber-400" />}
          <span className="text-xs font-medium uppercase tracking-wide text-zinc-400">
            {isTool && step.name ? step.name : step.role}
          </span>
          <span className="ml-auto rounded-full border border-line px-2 py-0.5 text-[11px] text-zinc-500">
            #{step.sequence}
          </span>
        </div>
      </div>

      <div className="space-y-3 px-4 py-4">
        {step.reasoning_content && (
          <details className="rounded-lg border border-line bg-panel-muted">
            <summary className="cursor-pointer list-none px-3 py-2 text-xs font-medium text-ink-muted">
              Thinking
            </summary>
            <div className="max-h-48 overflow-auto border-t border-line px-3 py-2 text-xs text-zinc-400 whitespace-pre-wrap">
              {step.reasoning_content}
            </div>
          </details>
        )}

        <div className="rounded-lg border border-line bg-panel px-3 py-3">
          {hasStructuredUserInput ? (
            <div className="max-h-96 overflow-auto">
              <UserInputDetail input={step.user_input} maxTextLength={2000} />
            </div>
          ) : (
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
        </div>

        {step.tool_calls && step.tool_calls.length > 0 && (
          <ToolCallPreviewList toolCalls={step.tool_calls} />
        )}

        {originalContent !== null && originalContent !== undefined && (
          <RawJsonBlock label="Original result" value={originalContent} />
        )}

        <details className="rounded-lg border border-line bg-panel-muted">
          <summary className="cursor-pointer list-none px-3 py-2 text-xs font-medium text-ink-muted">
            Step details
          </summary>
          <div className="space-y-3 border-t border-line px-3 py-3">
            <div className="flex flex-wrap gap-3 text-xs text-ink-muted">
              {step.agent_id ? <span>Agent {step.agent_id}</span> : null}
              <span>Run {step.run_id}</span>
              {step.tool_call_id ? <span>Tool call {step.tool_call_id}</span> : null}
              {step.created_at ? <span>{step.created_at}</span> : null}
            </div>
            {step.metrics && (
              <TokenMetricsBadges
                metrics={metrics}
                showDuration={true}
                showModelName={true}
                modelName={step.metrics?.model_name ?? null}
                chipClassName="bg-panel-strong"
              />
            )}

            <RawJsonBlock label="Raw step JSON" value={step} />
          </div>
        </details>
      </div>
    </div>
  );
}

export default SessionStepCard;
