import { render, screen, waitFor } from "@testing-library/react";
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
  test("renders runtime decisions and loop timeline above the waterfall", async () => {
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
      total_tool_calls: 2,
      input_query: "fix the bug",
      final_output: "done",
      spans: [],
      runtime_decisions: [
        {
          kind: "step_back",
          sequence: 9,
          run_id: "run-1",
          agent_id: "agent-1",
          created_at: "2026-04-25T12:00:01Z",
          summary: "2 results condensed after checkpoint seq 4",
          details: {
            affected_count: 2,
            checkpoint_seq: 4,
            experience: "switch plan",
          },
        },
      ],
      timeline_events: [
        {
          kind: "review_checkpoint",
          timestamp: "2026-04-25T12:00:00Z",
          sequence: 8,
          run_id: "run-1",
          agent_id: "agent-1",
          span_id: "span-1",
          step_id: "step-1",
          title: "Review Checkpoint",
          summary: "triggered by step_interval after 8 steps",
          status: "ok",
          details: {
            trigger_reason: "step_interval",
            steps_since_last_review: 8,
          },
        },
      ],
    });

    render(<TraceDetailPage />);

    await waitFor(() => {
      expect(screen.getAllByText("Runtime Decisions")).toHaveLength(2);
    });

    expect(screen.getByText("Loop Timeline")).toBeInTheDocument();
    expect(screen.getByText("Review Checkpoint")).toBeInTheDocument();
    expect(screen.getByText("Span Waterfall (0 spans)")).toBeInTheDocument();
  });
});
