import { describe, expect, test } from "vitest";

import {
  buildSchedulerTreeIndex,
  collectAutoExpandedNodeIds,
  hasNonTerminalSchedulerNodes,
} from "./scheduler-tree";
import type { SchedulerTree } from "./api";

const tree: SchedulerTree = {
  root_state_id: "root-1",
  root_session_id: "sess-1",
  generated_at: "2026-04-02T00:00:00Z",
  stats: {
    total: 4,
    running: 1,
    waiting: 1,
    queued: 0,
    idle: 0,
    completed: 1,
    failed: 0,
    cancelled: 1,
  },
  nodes: [
    {
      state_id: "root-1",
      root_state_id: "root-1",
      parent_state_id: null,
      child_ids: ["child-1", "child-2"],
      session_id: "sess-1",
      agent_id: "root-1",
      task_id: "task-1",
      status: "waiting",
      depth: 0,
      created_at: "2026-04-02T00:00:00Z",
      updated_at: "2026-04-02T00:00:05Z",
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
      child_ids: ["grandchild-1"],
      session_id: "sess-1",
      agent_id: "child-1",
      task_id: "task-1",
      status: "running",
      depth: 1,
      created_at: "2026-04-02T00:01:00Z",
      updated_at: "2026-04-02T00:01:05Z",
      completed_at: null,
      wake_condition: null,
      pending_event_count: 2,
      last_error: null,
      result_summary: null,
      last_run_result: null,
    },
    {
      state_id: "child-2",
      root_state_id: "root-1",
      parent_state_id: "root-1",
      child_ids: [],
      session_id: "sess-1",
      agent_id: "child-2",
      task_id: "task-1",
      status: "failed",
      depth: 1,
      created_at: "2026-04-02T00:02:00Z",
      updated_at: "2026-04-02T00:02:05Z",
      completed_at: "2026-04-02T00:02:05Z",
      wake_condition: null,
      pending_event_count: 0,
      last_error: "Cancelled by operator",
      result_summary: "Cancelled by operator",
      last_run_result: {
        run_id: "run-cancelled",
        termination_reason: "cancelled",
        summary: "Cancelled by operator",
        error: null,
        completed_at: "2026-04-02T00:02:05Z",
      },
    },
    {
      state_id: "grandchild-1",
      root_state_id: "root-1",
      parent_state_id: "child-1",
      child_ids: [],
      session_id: "sess-1",
      agent_id: "grandchild-1",
      task_id: "task-1",
      status: "completed",
      depth: 2,
      created_at: "2026-04-02T00:03:00Z",
      updated_at: "2026-04-02T00:03:05Z",
      completed_at: "2026-04-02T00:03:05Z",
      wake_condition: null,
      pending_event_count: 0,
      last_error: null,
      result_summary: "done",
      last_run_result: {
        run_id: "run-completed",
        termination_reason: "completed",
        summary: "done",
        error: null,
        completed_at: "2026-04-02T00:03:05Z",
      },
    },
  ],
};

describe("scheduler-tree selectors", () => {
  test("builds a parent-before-child index with quick lookup maps", () => {
    const index = buildSchedulerTreeIndex(tree);

    expect(index.root?.state_id).toBe("root-1");
    expect(index.orderedIds).toEqual([
      "root-1",
      "child-1",
      "child-2",
      "grandchild-1",
    ]);
    expect(index.nodesById["grandchild-1"]?.parent_state_id).toBe("child-1");
  });

  test("auto-expands the selected node ancestry chain", () => {
    const index = buildSchedulerTreeIndex(tree);

    expect(collectAutoExpandedNodeIds(index, "grandchild-1")).toEqual([
      "root-1",
      "child-1",
      "grandchild-1",
    ]);
  });

  test("detects whether the tree still has active work", () => {
    const index = buildSchedulerTreeIndex(tree);
    expect(hasNonTerminalSchedulerNodes(index)).toBe(true);

    const terminalIndex = buildSchedulerTreeIndex({
      ...tree,
      nodes: tree.nodes.map((node) =>
        node.status === "running" || node.status === "waiting"
          ? {
              ...node,
              status: "completed",
              completed_at: node.updated_at,
            }
          : node,
      ),
      stats: {
        ...tree.stats,
        running: 0,
        waiting: 0,
        completed: 3,
      },
    });

    expect(hasNonTerminalSchedulerNodes(terminalIndex)).toBe(false);
  });
});
