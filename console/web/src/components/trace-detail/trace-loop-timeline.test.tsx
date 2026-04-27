import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";

import type { TraceTimelineEvent } from "@/lib/api";

import { TraceLoopTimeline } from "./trace-loop-timeline";

describe("TraceLoopTimeline", () => {
  test("renders LLM and tool details as readable content before raw JSON", () => {
    const events: TraceTimelineEvent[] = [
      {
        kind: "llm_call",
        timestamp: "2026-04-25T12:00:00Z",
        sequence: 1,
        run_id: "run-1",
        agent_id: "agent-1",
        span_id: "span-llm",
        step_id: "step-llm",
        title: "LLM Call",
        summary: "gpt-4.1 · tool_calls",
        status: "ok",
        details: {
          finish_reason: "tool_calls",
          messages: [{ role: "user", content: "Find docs" }],
          response_content: "I will search the docs.",
          response_tool_calls: [
            {
              id: "call-1",
              type: "function",
              function: {
                name: "web_search",
                arguments: '{"query":"docs"}',
              },
            },
          ],
        },
      },
      {
        kind: "tool_call",
        timestamp: "2026-04-25T12:00:01Z",
        sequence: 2,
        run_id: "run-1",
        agent_id: "agent-1",
        span_id: "span-tool",
        step_id: "step-tool",
        title: "Tool Call: web_search",
        summary: "completed",
        status: "ok",
        details: {
          tool_name: "web_search",
          tool_call_id: "call-1",
          input_args: { query: "docs" },
          output: "Search result body",
          status: "completed",
        },
      },
    ];

    render(<TraceLoopTimeline events={events} />);

    fireEvent.click(screen.getByText("LLM Call"));
    expect(screen.getByText("I will search the docs.")).toBeInTheDocument();
    expect(screen.getAllByText("web_search").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Raw details JSON")).toHaveLength(2);

    fireEvent.click(screen.getByText("Tool Call: web_search"));
    expect(screen.getByText("Arguments")).toBeInTheDocument();
    expect(screen.getByText("Search result body")).toBeInTheDocument();
    expect(screen.getAllByText("Raw details JSON")).toHaveLength(2);
  });
});
