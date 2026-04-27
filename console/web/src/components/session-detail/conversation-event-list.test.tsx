import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";

import { ConversationEventList } from "./conversation-event-list";

describe("ConversationEventList", () => {
  test("filters dialogue, key events, and all events", () => {
    render(
      <ConversationEventList
        events={[
          {
            id: "evt-user",
            session_id: "sess-1",
            run_id: "run-1",
            sequence: 1,
            kind: "user_message",
            priority: "primary",
            title: "User",
            summary: "Please inspect auth",
            details: {
              content: "Please inspect auth with full context",
            },
          },
          {
            id: "evt-tool",
            session_id: "sess-1",
            run_id: "run-1",
            sequence: 2,
            kind: "tool_event",
            priority: "secondary",
            title: "Tool: web_search",
            summary: "Searched the repo",
            details: {
              content_for_user: "Searched the repo and found auth.py",
            },
          },
          {
            id: "evt-review",
            session_id: "sess-1",
            run_id: "run-1",
            sequence: 3,
            kind: "review_event",
            priority: "muted",
            title: "Review",
            summary: "Review misaligned; 2 steps condensed",
            details: {},
          },
        ]}
      />,
    );

    expect(screen.getByText("Please inspect auth with full context")).toBeInTheDocument();
    expect(screen.getByText("Searched the repo and found auth.py")).toBeInTheDocument();
    expect(screen.queryByText("Review misaligned; 2 steps condensed")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Dialogue" }));
    expect(screen.getByText("Please inspect auth with full context")).toBeInTheDocument();
    expect(screen.queryByText("Searched the repo and found auth.py")).not.toBeInTheDocument();
    expect(screen.queryByText("Review misaligned; 2 steps condensed")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "All Events" }));
    expect(screen.getByText("Searched the repo and found auth.py")).toBeInTheDocument();
    expect(screen.getByText("Review misaligned; 2 steps condensed")).toBeInTheDocument();
  });

  test("shows assistant tool calls from event details without opening raw JSON", () => {
    render(
      <ConversationEventList
        events={[
          {
            id: "evt-assistant",
            session_id: "sess-1",
            run_id: "run-1",
            sequence: 4,
            kind: "assistant_message",
            priority: "primary",
            title: "Assistant",
            summary: "I will search",
            details: {
              content: "I will search the repository.",
              tool_calls: [
                {
                  id: "call-1",
                  type: "function",
                  function: {
                    name: "grep",
                    arguments: '{"pattern":"auth"}',
                  },
                },
              ],
            },
          },
        ]}
      />,
    );

    expect(screen.getByText("I will search the repository.")).toBeInTheDocument();
    expect(screen.getByText("grep")).toBeInTheDocument();
    expect(screen.getByText("auth")).toBeInTheDocument();
  });
});
