import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, test, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  getSessionDetail: vi.fn(),
  getSessionSteps: vi.fn(),
  listRuns: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useParams: () => ({ id: "sess-1" }),
  useRouter: () => ({ replace: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    getSessionDetail: apiMocks.getSessionDetail,
    getSessionSteps: apiMocks.getSessionSteps,
    listRuns: apiMocks.listRuns,
  };
});

import SessionDetailPage from "./page";

describe("SessionDetailPage", () => {
  test("shows runs and steps as the main stage", async () => {
    apiMocks.getSessionDetail.mockResolvedValue({
      summary: {
        session_id: "sess-1",
        agent_id: "agent-1",
        last_user_input: null,
        last_response: "done",
        run_count: 1,
        step_count: 2,
        metrics: {
          run_count: 1,
          completed_run_count: 1,
          step_count: 2,
          tool_calls_count: 0,
          duration_ms: 10,
          input_tokens: 5,
          output_tokens: 6,
          total_tokens: 11,
          cache_read_tokens: 0,
          cache_creation_tokens: 0,
          token_cost: 0.01,
        },
        created_at: "2026-04-22T12:00:00Z",
        updated_at: "2026-04-22T12:01:00Z",
        chat_context_scope_id: null,
        created_by: "test",
        base_agent_id: "agent-1",
        root_state_status: "idle",
        source_session_id: null,
        fork_context_summary: null,
        archived_at: null,
      },
      session: {
        id: "sess-1",
        chat_context_scope_id: null,
        base_agent_id: "agent-1",
        created_by: "test",
        created_at: "2026-04-22T12:00:00Z",
        updated_at: "2026-04-22T12:01:00Z",
        source_session_id: null,
        fork_context_summary: null,
        archived_at: null,
      },
      chat_context: null,
      scheduler_state: null,
      milestone_board: {
        session_id: "sess-1",
        run_id: "run-1",
        milestones: [
          {
            id: "inspect",
            description: "Inspect auth flow",
            status: "active",
            declared_at_seq: 3,
            completed_at_seq: null,
          },
        ],
        active_milestone_id: "inspect",
        latest_checkpoint: null,
        latest_review_outcome: null,
        pending_review_reason: null,
      },
      review_cycles: [],
      conversation_events: [
        {
          id: "evt-1",
          session_id: "sess-1",
          run_id: "run-1",
          sequence: 1,
          kind: "assistant_message",
          priority: "primary",
          title: "Assistant",
          summary: "I will inspect auth",
          details: {},
        },
      ],
      observability: {
        recent_traces: [
          {
            trace_id: "trace-1",
            agent_id: "agent-1",
            session_id: "sess-1",
            user_id: null,
            start_time: "2026-04-22T12:00:00Z",
            duration_ms: 123,
            status: "ok",
            total_tokens: 11,
            total_input_tokens: 5,
            total_output_tokens: 6,
            total_cache_read_tokens: 0,
            total_cache_creation_tokens: 0,
            total_token_cost: 0.01,
            total_llm_calls: 1,
            total_tool_calls: 0,
            input_query: "hello",
            final_output: "done",
          },
        ],
        decision_events: [],
      },
    });
    apiMocks.listRuns.mockResolvedValue({
      items: [
        {
          id: "run-1",
          agent_id: "agent-1",
          session_id: "sess-1",
          user_id: null,
          user_input: "hello",
          status: "completed",
          response_content: "done",
          metrics: {
            steps_count: 2,
            tool_calls_count: 0,
            duration_ms: 10,
            token_cost: 0.01,
          },
          created_at: "2026-04-22T12:00:00Z",
          updated_at: "2026-04-22T12:01:00Z",
          parent_run_id: null,
        },
      ],
      limit: 50,
      offset: 0,
      has_more: false,
      total: 1,
    });
    apiMocks.getSessionSteps.mockResolvedValue({
      items: [
        {
          id: "step-1",
          session_id: "sess-1",
          run_id: "run-1",
          sequence: 1,
          role: "user",
          agent_id: "agent-1",
          content: "hello",
          content_for_user: "hello",
          reasoning_content: null,
          user_input: null,
          tool_calls: null,
          tool_call_id: null,
          name: null,
          metrics: null,
          created_at: "2026-04-22T12:00:00Z",
          parent_run_id: null,
          depth: 0,
        },
      ],
      limit: 100,
      offset: 0,
      has_more: false,
      total: 1,
    });

    render(<SessionDetailPage />);

    await waitFor(() => {
      expect(screen.getByText("Agent Runs & Steps")).toBeInTheDocument();
    });

    expect(screen.getByText("hello")).toBeInTheDocument();
    expect(apiMocks.listRuns).toHaveBeenCalled();
    expect(apiMocks.getSessionSteps).toHaveBeenCalled();

    expect(screen.queryByText("Milestone Board")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Show debug extras" }));
    expect(await screen.findByText("Milestone Board")).toBeInTheDocument();
    expect(screen.getByText("Conversation")).toBeInTheDocument();
  });
});
