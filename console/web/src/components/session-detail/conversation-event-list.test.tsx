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
            details: {},
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
            details: {},
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

    expect(screen.getByText("Please inspect auth")).toBeInTheDocument();
    expect(screen.getByText("Searched the repo")).toBeInTheDocument();
    expect(screen.queryByText("Review misaligned; 2 steps condensed")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Dialogue" }));
    expect(screen.getByText("Please inspect auth")).toBeInTheDocument();
    expect(screen.queryByText("Searched the repo")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "All Events" }));
    expect(screen.getByText("Review misaligned; 2 steps condensed")).toBeInTheDocument();
  });
});
