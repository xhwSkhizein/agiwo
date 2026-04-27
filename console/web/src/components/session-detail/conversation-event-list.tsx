"use client";

import { useState } from "react";
import { Bot, GitBranch, User, Wrench } from "lucide-react";

import { JsonDisclosure } from "@/components/json-disclosure";
import { SectionCard } from "@/components/section-card";
import {
  StepContentPreview,
  ToolCallPreviewList,
} from "@/components/step-content-preview";
import { UserInputDetail } from "@/components/user-input-detail";
import type { ConversationEvent, ToolCallPayload, UserInput } from "@/lib/api";

type ConversationFilter = "dialogue" | "key-events" | "all";

function filterEvents(
  events: ConversationEvent[],
  filter: ConversationFilter,
): ConversationEvent[] {
  if (filter === "dialogue") {
    return events.filter(
      (event) =>
        event.kind === "user_message" || event.kind === "assistant_message",
    );
  }
  if (filter === "key-events") {
    return events.filter((event) => event.priority !== "muted");
  }
  return events;
}

function eventCardClass(event: ConversationEvent): string {
  if (event.priority === "primary") {
    return "border-line bg-panel";
  }
  if (event.priority === "secondary") {
    return "border-line bg-panel-muted";
  }
  return "border-line bg-panel-muted/60";
}

function eventIcon(event: ConversationEvent) {
  if (event.kind === "user_message") {
    return <User className="h-3.5 w-3.5 text-blue-400" />;
  }
  if (event.kind === "assistant_message") {
    return <Bot className="h-3.5 w-3.5 text-green-400" />;
  }
  if (event.kind.includes("tool") || event.kind.includes("milestone")) {
    return <Wrench className="h-3.5 w-3.5 text-amber-400" />;
  }
  return <GitBranch className="h-3.5 w-3.5 text-ink-muted" />;
}

function firstMeaningfulContent(...values: unknown[]): unknown {
  for (const value of values) {
    if (typeof value === "string") {
      if (value.trim().length > 0) {
        return value;
      }
      continue;
    }
    if (value !== null && value !== undefined) {
      return value;
    }
  }
  return null;
}

function eventContent(event: ConversationEvent): unknown {
  return firstMeaningfulContent(
    event.details.content_for_user,
    event.details.condensed_content,
    event.details.content,
    event.summary,
  );
}

function eventToolCalls(event: ConversationEvent): ToolCallPayload[] {
  return Array.isArray(event.details.tool_calls)
    ? (event.details.tool_calls as ToolCallPayload[])
    : [];
}

function eventUserInput(event: ConversationEvent): UserInput | null {
  return event.details.user_input ? (event.details.user_input as UserInput) : null;
}

export function ConversationEventList({
  events,
}: {
  events: ConversationEvent[];
}) {
  const [filter, setFilter] = useState<ConversationFilter>("key-events");
  const filteredEvents = filterEvents(events, filter);

  return (
    <SectionCard
      title="Conversation"
      action={
        <div className="flex flex-wrap gap-2">
          {[
            ["dialogue", "Dialogue"],
            ["key-events", "Dialogue + Key Events"],
            ["all", "All Events"],
          ].map(([value, label]) => (
            <button
              key={value}
              type="button"
              onClick={() => setFilter(value as ConversationFilter)}
              className={`rounded-full border px-3 py-1 text-xs transition-colors ${
                filter === value
                  ? "border-accent bg-panel-strong text-foreground"
                  : "border-line text-ink-muted hover:border-line-strong hover:text-foreground"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      }
      bodyClassName="space-y-3 px-4 py-4"
    >
      {filteredEvents.length === 0 ? (
        <div className="rounded-xl border border-dashed border-line px-4 py-6 text-sm text-ink-muted">
          No conversation events for the selected filter.
        </div>
      ) : (
        <div className="relative space-y-3 before:absolute before:left-[1.05rem] before:top-3 before:h-[calc(100%-1.5rem)] before:w-px before:bg-line">
          {filteredEvents.map((event) => {
            const toolCalls = eventToolCalls(event);
            const userInput = eventUserInput(event);
            const content = eventContent(event);
            return (
              <div
                key={event.id}
                className={`relative ml-10 rounded-xl border px-3 py-3 ${eventCardClass(event)}`}
              >
                <div className="absolute -left-10 top-3 flex h-8 w-8 items-center justify-center rounded-full border border-line bg-panel font-mono text-[11px] text-ink-muted">
                  {eventIcon(event)}
                </div>
                <div className="space-y-3">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-sm font-medium text-foreground">
                      {event.title}
                    </span>
                    {event.sequence !== null ? (
                      <span className="rounded-full border border-line px-2 py-0.5 text-[11px] text-ink-muted">
                        #{event.sequence}
                      </span>
                    ) : null}
                  </div>
                  {event.kind === "user_message" && userInput ? (
                    <UserInputDetail input={userInput} maxTextLength={2000} />
                  ) : (
                    <StepContentPreview value={content} emptyLabel="No event content" />
                  )}
                  {toolCalls.length > 0 ? (
                    <ToolCallPreviewList toolCalls={toolCalls} />
                  ) : null}
                  <div className="flex flex-wrap gap-2 text-[11px] text-ink-faint">
                    <span>{event.kind}</span>
                    <span>{event.priority}</span>
                    {event.run_id ? <span>run {event.run_id}</span> : null}
                  </div>
                </div>
                <JsonDisclosure
                  label="Raw event JSON"
                  value={event.details}
                  className="mt-3 bg-panel"
                  contentClassName="bg-panel"
                />
              </div>
            );
          })}
        </div>
      )}
    </SectionCard>
  );
}

export default ConversationEventList;
