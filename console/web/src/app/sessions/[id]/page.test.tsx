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
  test("switches between mainline and debug session views", async () => {
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
        latest_checkpoint: {
          seq: 8,
          milestone_id: "inspect",
          confirmed_at: "2026-04-22T12:00:01Z",
        },
        latest_review_outcome: {
          aligned: false,
          experience: "switch plan",
          step_back_applied: true,
          affected_count: 2,
          trigger_reason: "step_interval",
          active_milestone: "Inspect auth flow",
          resolved_at: "2026-04-22T12:00:02Z",
        },
        pending_review_reason: null,
      },
      review_cycles: [
        {
          cycle_id: "run-1:8",
          run_id: "run-1",
          agent_id: "agent-1",
          trigger_reason: "step_interval",
          steps_since_last_review: 8,
          active_milestone: "Inspect auth flow",
          active_milestone_id: "inspect",
          hook_advice: "narrow the search",
          aligned: false,
          experience: "switch plan",
          step_back_applied: true,
          rollback_range: null,
          affected_count: 2,
          started_at: "2026-04-22T12:00:01Z",
          resolved_at: "2026-04-22T12:00:02Z",
          raw_notice: "Trigger: step_interval",
        },
      ],
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
        decision_events: [
          {
            kind: "termination",
            sequence: 8,
            run_id: "run-1",
            agent_id: "agent-1",
            created_at: "2026-04-22T12:00:01Z",
            summary: "completed via finished",
            details: {
              reason: "completed",
              source: "finished",
            },
          },
        ],
      },
    });
    apiMocks.listRuns.mockResolvedValue({
      items: [],
      limit: 50,
      offset: 0,
      has_more: false,
      total: 0,
    });
    apiMocks.getSessionSteps.mockResolvedValue({
      items: [],
      limit: 100,
      offset: 0,
      has_more: false,
      total: 0,
    });

    render(<SessionDetailPage />);

    await waitFor(() => {
      expect(screen.getByText("Milestone Board")).toBeInTheDocument();
    });

    expect(screen.getAllByText("Inspect auth flow").length).toBeGreaterThan(0);
    expect(screen.getByText("Conversation")).toBeInTheDocument();
    expect(screen.queryByText("Observability")).not.toBeInTheDocument();
    expect(apiMocks.listRuns).not.toHaveBeenCalled();
    expect(apiMocks.getSessionSteps).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Debug" }));

    expect(await screen.findByText("Observability")).toBeInTheDocument();
    expect(screen.getByText("Trace Context")).toBeInTheDocument();
    expect(screen.getByText("Runtime Decisions")).toBeInTheDocument();
    expect(screen.getByText("completed via finished")).toBeInTheDocument();
    expect(screen.getByText("hello")).toBeInTheDocument();
  });
});
