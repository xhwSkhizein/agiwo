import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, test, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  getTrace: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useParams: () => ({ id: "trace-1" }),
}));

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    getTrace: apiMocks.getTrace,
  };
});

import TraceDetailPage from "./page";

describe("TraceDetailPage", () => {
  test("renders clickable timeline with right-hand span detail", async () => {
    apiMocks.getTrace.mockResolvedValue({
      trace_id: "trace-1",
      agent_id: "agent-1",
      session_id: "sess-1",
      user_id: null,
      start_time: "2026-04-25T12:00:00Z",
      end_time: "2026-04-25T12:00:02Z",
      duration_ms: 2000,
      status: "ok",
      root_span_id: "root-1",
      max_depth: 1,
      total_tokens: 10,
      total_input_tokens: 4,
      total_output_tokens: 6,
      total_cache_read_tokens: 0,
      total_cache_creation_tokens: 0,
      total_token_cost: 0.01,
      total_llm_calls: 1,
      total_tool_calls: 1,
      input_query: "fix the bug",
      final_output: "done",
      spans: [
        {
          span_id: "root-1",
          trace_id: "trace-1",
          parent_span_id: null,
          kind: "agent",
          name: "agent.run",
          start_time: "2026-04-25T12:00:00Z",
          end_time: "2026-04-25T12:00:02Z",
          duration_ms: 2000,
          status: "ok",
          error_message: null,
          depth: 0,
          attributes: {},
          input_preview: "fix the bug",
          output_preview: "done",
          metrics: {},
          llm_details: null,
          tool_details: null,
          run_id: "run-1",
          step_id: null,
        },
        {
          span_id: "llm-1",
          trace_id: "trace-1",
          parent_span_id: "root-1",
          kind: "llm_call",
          name: "llm.assistant#1",
          start_time: "2026-04-25T12:00:00.200Z",
          end_time: "2026-04-25T12:00:01Z",
          duration_ms: 800,
          status: "ok",
          error_message: null,
          depth: 1,
          attributes: {},
          input_preview: null,
          output_preview: "I will inspect auth",
          metrics: { "tokens.input": 4, "tokens.output": 6, model: "test" },
          llm_details: { phase: "assistant", attempt_no: 1, model: "test" },
          tool_details: null,
          run_id: "run-1",
          step_id: null,
        },
        {
          span_id: "tool-1",
          trace_id: "trace-1",
          parent_span_id: "root-1",
          kind: "tool_call",
          name: "tool.bash",
          start_time: "2026-04-25T12:00:01Z",
          end_time: "2026-04-25T12:00:01.500Z",
          duration_ms: 500,
          status: "ok",
          error_message: null,
          depth: 1,
          attributes: {},
          input_preview: null,
          output_preview: "ok",
          metrics: {},
          llm_details: null,
          tool_details: {
            tool_name: "bash",
            arguments: { command: "ls" },
            result: "ok",
          },
          run_id: "run-1",
          step_id: null,
        },
      ],
      mainline_events: [],
      review_cycles: [],
      runtime_decisions: [
        {
          kind: "compaction_failed",
          summary: "compaction failed once",
          sequence: 2,
          created_at: "2026-04-25T12:00:01Z",
          run_id: "run-1",
          agent_id: "agent-1",
          details: {},
        },
      ],
      timeline_events: [],
      llm_calls: [],
    });

    render(<TraceDetailPage />);

    await waitFor(() => {
      expect(screen.getByText("Execution timeline")).toBeInTheDocument();
    });

    expect(screen.getAllByText("fix the bug").length).toBeGreaterThan(0);
    expect(screen.getAllByText("agent.run").length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: /llm\.assistant#1/i })).toBeInTheDocument();
    expect(screen.queryByText("Agent Execution Diagnostics")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /llm\.assistant#1/i }));
    fireEvent.click(screen.getByRole("button", { name: "llm" }));
    expect(await screen.findByText("assistant")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /tool\.bash/i }));
    fireEvent.click(screen.getByRole("button", { name: "tools" }));
    expect(await screen.findByText("bash")).toBeInTheDocument();
  });
});
