import { render, screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";

import type { TraceDetail } from "@/lib/api";

import { TraceDiagnostics } from "./trace-diagnostics";

function buildTrace(): TraceDetail {
  return {
    trace_id: "trace-1",
    agent_id: "agent-1",
    session_id: "session-1",
    user_id: null,
    start_time: "2026-04-27T10:00:00Z",
    end_time: "2026-04-27T10:00:12Z",
    duration_ms: 12000,
    status: "ok",
    root_span_id: "root",
    total_tokens: 95000,
    total_input_tokens: 70000,
    total_output_tokens: 25000,
    total_token_cost: 0.12,
    total_llm_calls: 1,
    total_tool_calls: 1,
    total_cache_read_tokens: 0,
    total_cache_creation_tokens: 0,
    max_depth: 1,
    input_query: "diagnose",
    final_output: "done",
    runtime_decisions: [
      {
        kind: "step_back",
        sequence: 6,
        run_id: "run-1",
        agent_id: "agent-1",
        created_at: "2026-04-27T10:00:11Z",
        summary: "2 results condensed after checkpoint seq 4",
        details: { affected_count: 2 },
      },
    ],
    timeline_events: [],
    mainline_events: [],
    review_cycles: [],
    llm_calls: [],
    spans: [
      {
        span_id: "llm-1",
        trace_id: "trace-1",
        parent_span_id: null,
        kind: "llm_call",
        name: "gpt-5.4",
        start_time: "2026-04-27T10:00:00Z",
        end_time: "2026-04-27T10:00:02Z",
        duration_ms: 2000,
        status: "ok",
        error_message: null,
        depth: 1,
        attributes: { sequence: 2 },
        input_preview: null,
        output_preview: "I will inspect the file.",
        metrics: { "tokens.total": 90000 },
        llm_details: {
          messages: [
            { role: "system", content: "You are an agent." },
            { role: "user", content: "Find the bug." },
          ],
          tools: [{ name: "read_file" }],
          response_content: "I will inspect the file.",
          response_tool_calls: [
            {
              id: "call-1",
              type: "function",
              function: { name: "read_file", arguments: '{"path":"app.py"}' },
            },
          ],
          finish_reason: "tool_calls",
        },
        tool_details: null,
        run_id: "run-1",
        step_id: "step-2",
      },
      {
        span_id: "tool-1",
        trace_id: "trace-1",
        parent_span_id: null,
        kind: "tool_call",
        name: "read_file",
        start_time: "2026-04-27T10:00:02Z",
        end_time: "2026-04-27T10:00:12Z",
        duration_ms: 10000,
        status: "error",
        error_message: "file missing",
        depth: 1,
        attributes: { sequence: 3, tool_call_id: "call-1" },
        input_preview: null,
        output_preview: "file missing",
        metrics: {},
        llm_details: null,
        tool_details: {
          tool_name: "read_file",
          tool_call_id: "call-1",
          input_args: { path: "app.py" },
          output: "file missing",
          error: "file missing",
          status: "error",
        },
        run_id: "run-1",
        step_id: "step-3",
      },
    ],
  };
}

describe("TraceDiagnostics", () => {
  test("renders risk signals, paired tool transactions, and LLM prompt messages", () => {
    render(<TraceDiagnostics trace={buildTrace()} />);

    expect(screen.getByText("Diagnostic Summary")).toBeInTheDocument();
    expect(screen.getByText(/errored span/)).toBeInTheDocument();
    expect(screen.getByText(/high token usage/)).toBeInTheDocument();
    expect(screen.getByText("Execution Chain")).toBeInTheDocument();
    expect(screen.getByText("Tool Transactions")).toBeInTheDocument();
    expect(screen.getByText("LLM Call Inspector")).toBeInTheDocument();
    expect(screen.getByText("Find the bug.")).toBeInTheDocument();
    expect(screen.getAllByText("read_file").length).toBeGreaterThan(0);
    expect(screen.getAllByText("file missing").length).toBeGreaterThan(0);
  });
});
