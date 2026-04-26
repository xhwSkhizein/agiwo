import { render, screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";

import type { StepResponse } from "@/lib/api";

import { SessionStepCard } from "./session-step-card";

function buildStep(partial: Partial<StepResponse>): StepResponse {
  return {
    id: "step-1",
    session_id: "session-1",
    run_id: "run-1",
    sequence: 3,
    role: "assistant",
    agent_id: "agent-1",
    content: null,
    content_for_user: null,
    reasoning_content: null,
    user_input: null,
    tool_calls: null,
    tool_call_id: null,
    name: null,
    metrics: null,
    created_at: null,
    parent_run_id: null,
    depth: 0,
    ...partial,
  };
}

describe("SessionStepCard", () => {
  test("shows assistant message content before raw JSON", () => {
    render(
      <SessionStepCard
        step={buildStep({
          content: [{ type: "text", text: "Assistant visible answer" }],
          content_for_user: "User-facing answer",
        })}
      />,
    );

    expect(screen.getByText("User-facing answer")).toBeInTheDocument();
    expect(screen.getByText("Raw step JSON")).toBeInTheDocument();
  });

  test("shows tool result content and tool call arguments semantically", () => {
    render(
      <SessionStepCard
        step={buildStep({
          role: "assistant",
          content: "",
          tool_calls: [
            {
              id: "call-1",
              type: "function",
              function: {
                name: "web_search",
                arguments: '{"query":"agiwo console"}',
              },
            },
          ],
        })}
      />,
    );

    expect(screen.getByText("web_search")).toBeInTheDocument();
    expect(screen.getAllByText(/agiwo console/).length).toBeGreaterThan(0);

    render(
      <SessionStepCard
        step={buildStep({
          id: "tool-step",
          role: "tool",
          sequence: 4,
          name: "web_search",
          content: { status: "ok", result: "Search result body" },
        })}
      />,
    );

    expect(screen.getByText("result")).toBeInTheDocument();
    expect(screen.getByText("Search result body")).toBeInTheDocument();
  });
});
