"use client";

import { useState } from "react";

import { JsonDisclosure } from "@/components/json-disclosure";
import { SectionCard } from "@/components/section-card";
import type { ConversationEvent } from "@/lib/api";

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
        filteredEvents.map((event) => (
          <details
            key={event.id}
            className={`rounded-xl border px-3 py-3 ${eventCardClass(event)}`}
          >
            <summary className="cursor-pointer list-none">
              <div className="space-y-2">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm font-medium text-foreground">
                    {event.title}
                  </span>
                  <span className="rounded-full border border-line px-2 py-0.5 text-[11px] uppercase tracking-wide text-ink-muted">
                    {event.kind}
                  </span>
                  <span className="rounded-full border border-line px-2 py-0.5 text-[11px] uppercase tracking-wide text-ink-muted">
                    {event.priority}
                  </span>
                  <span className="text-xs text-ink-muted">
                    seq {event.sequence ?? "-"}
                  </span>
                </div>
                <p className="text-sm text-foreground">{event.summary}</p>
              </div>
            </summary>
            <div className="mt-3">
              <JsonDisclosure
                label="Details"
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

export default ConversationEventList;
