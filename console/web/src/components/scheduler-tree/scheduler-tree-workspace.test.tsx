import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";

import { ApiError } from "@/lib/api";
import { renderWithProviders } from "@/test/render";

import { SchedulerTreeWorkspace } from "./scheduler-tree-workspace";

const apiMocks = vi.hoisted(() => ({
  getSchedulerTree: vi.fn(),
  getAgentState: vi.fn(),
  getPendingEvents: vi.fn(),
  steerAgent: vi.fn(),
  cancelAgent: vi.fn(),
  resumeAgent: vi.fn(),
}));

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    getSchedulerTree: apiMocks.getSchedulerTree,
    getAgentState: apiMocks.getAgentState,
    getPendingEvents: apiMocks.getPendingEvents,
    steerAgent: apiMocks.steerAgent,
    cancelAgent: apiMocks.cancelAgent,
    resumeAgent: apiMocks.resumeAgent,
  };
});

const tree = {
  root_state_id: "root-1",
  root_session_id: "sess-1",
  generated_at: "2026-04-02T00:00:00Z",
  stats: {
    total: 2,
    running: 1,
    waiting: 0,
    queued: 0,
    idle: 0,
    completed: 0,
    failed: 0,
    cancelled: 0,
  },
  nodes: [
    {
      state_id: "root-1",
      root_state_id: "root-1",
      parent_state_id: null,
      child_ids: ["child-1"],
      session_id: "sess-1",
      agent_id: "root-1",
      task_id: "task-1",
      status: "running",
      depth: 0,
      created_at: "2026-04-02T00:00:00Z",
      updated_at: "2026-04-02T00:00:01Z",
      completed_at: null,
      wake_condition: null,
      pending_event_count: 1,
      last_error: null,
      result_summary: null,
      last_run_result: null,
    },
    {
      state_id: "child-1",
      root_state_id: "root-1",
      parent_state_id: "root-1",
      child_ids: [],
      session_id: "sess-1",
      agent_id: "child-1",
      task_id: "task-1",
      status: "completed",
      depth: 1,
      created_at: "2026-04-02T00:00:02Z",
      updated_at: "2026-04-02T00:00:03Z",
      completed_at: "2026-04-02T00:00:03Z",
      wake_condition: null,
      pending_event_count: 0,
      last_error: null,
      result_summary: "done",
      last_run_result: {
        run_id: "run-child-1",
        termination_reason: "completed",
        summary: "done",
        error: null,
        completed_at: "2026-04-02T00:00:03Z",
      },
    },
  ],
} as const;

const rootDetail = {
  id: "root-1",
  root_state_id: "root-1",
  session_id: "sess-1",
  status: "running",
  task: "Root task",
  parent_id: null,
  pending_input: null,
  config_overrides: {},
  wake_condition: null,
  result_summary: null,
  last_run_result: null,
  signal_propagated: false,
  agent_config_id: null,
  is_persistent: true,
  depth: 0,
  wake_count: 0,
  metrics: {
    run_count: 0,
    completed_run_count: 0,
    step_count: 0,
    tool_calls_count: 0,
    duration_ms: 0,
    input_tokens: 0,
    output_tokens: 0,
    total_tokens: 0,
    cache_read_tokens: 0,
    cache_creation_tokens: 0,
    token_cost: 0,
  },
  created_at: "2026-04-02T00:00:00Z",
  updated_at: "2026-04-02T00:00:05Z",
};

describe("SchedulerTreeWorkspace", () => {
  test("renders root-only actions when the root node is selected", async () => {
    apiMocks.getSchedulerTree.mockResolvedValue(tree);
    apiMocks.getAgentState.mockResolvedValue(rootDetail);
    apiMocks.getPendingEvents.mockResolvedValue([]);

    renderWithProviders(
      <SchedulerTreeWorkspace
        rootStateId="root-1"
        selectedStateId="root-1"
        onSelectedStateIdChange={vi.fn()}
      />,
    );

    expect(await screen.findByText("Scheduler Tree")).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: "Cancel Root" })).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: "Send Steering" })).toBeInTheDocument();
  });

  test("hides root actions when a child node is selected and notifies selection changes", async () => {
    apiMocks.getSchedulerTree.mockResolvedValue(tree);
    apiMocks.getAgentState.mockResolvedValue({
      ...rootDetail,
      id: "child-1",
      root_state_id: "root-1",
      status: "completed",
      parent_id: "root-1",
      is_persistent: false,
      result_summary: "done",
      last_run_result: {
        run_id: "run-child-1",
        termination_reason: "completed",
        summary: "done",
        error: null,
        completed_at: "2026-04-02T00:00:03Z",
      },
    });
    apiMocks.getPendingEvents.mockResolvedValue([]);
    const onSelectedStateIdChange = vi.fn();

    renderWithProviders(
      <SchedulerTreeWorkspace
        rootStateId="root-1"
        selectedStateId="child-1"
        onSelectedStateIdChange={onSelectedStateIdChange}
      />,
    );

    expect(await screen.findByText("child-1")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Cancel Root" })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Select root-1" }));
    expect(onSelectedStateIdChange).toHaveBeenCalledWith("root-1");
  });

  test("shows a not-found state for 404 tree responses", async () => {
    apiMocks.getSchedulerTree.mockRejectedValue(
      new ApiError(404, "Agent state not found"),
    );

    renderWithProviders(
      <SchedulerTreeWorkspace
        rootStateId="missing-root"
        selectedStateId={null}
        onSelectedStateIdChange={vi.fn()}
      />,
    );

    await waitFor(() => {
      expect(
        screen.getByText("Scheduler state not found or already removed."),
      ).toBeInTheDocument();
    });
  });
});
