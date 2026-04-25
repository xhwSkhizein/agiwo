import { screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";

import { renderWithProviders } from "@/test/render";

import { SessionObservabilityPanel } from "./session-observability-panel";

describe("SessionObservabilityPanel", () => {
  test("renders recent traces and runtime decisions together", () => {
    renderWithProviders(
      <SessionObservabilityPanel
        sessionId="sess-1"
        observability={{
          recent_traces: [
            {
              trace_id: "trace-1",
              agent_id: "agent-1",
              session_id: "sess-1",
              user_id: null,
              start_time: "2026-04-22T12:00:00Z",
              duration_ms: 123,
              status: "ok",
              total_tokens: 12,
              total_input_tokens: 5,
              total_output_tokens: 7,
              total_cache_read_tokens: 0,
              total_cache_creation_tokens: 0,
              total_token_cost: 0.01,
              total_llm_calls: 1,
              total_tool_calls: 0,
              input_query: "hello",
              final_output: "done",
            },
          ],
          decision_events: [
            {
              kind: "step_back",
              sequence: 8,
              run_id: "run-1",
              agent_id: "agent-1",
              created_at: "2026-04-22T12:01:00Z",
              summary: "2 results condensed after checkpoint seq 4",
              details: {
                affected_count: 2,
                checkpoint_seq: 4,
                experience: "switch plan",
              },
            },
          ],
        }}
      />,
    );

    expect(screen.getByText("Observability")).toBeInTheDocument();
    expect(screen.getByText("Trace Context")).toBeInTheDocument();
    expect(screen.getByText("Runtime Decisions")).toBeInTheDocument();
    expect(screen.getByText("hello")).toBeInTheDocument();
    expect(screen.getByText("2 results condensed after checkpoint seq 4")).toBeInTheDocument();
    expect(screen.getByText("checkpoint_seq 4")).toBeInTheDocument();
    expect(screen.getByText("switch plan")).toBeInTheDocument();
    expect(screen.getByText("Recent Traces")).toBeInTheDocument();
  });
});
