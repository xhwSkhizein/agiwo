import type { SchedulerTree, SchedulerTreeNode } from "./api";

export interface SchedulerTreeIndex {
  root: SchedulerTreeNode | null;
  nodesById: Record<string, SchedulerTreeNode>;
  orderedIds: string[];
}

const TERMINAL_STATUSES = new Set(["completed", "failed", "cancelled"]);

export function buildSchedulerTreeIndex(tree: SchedulerTree): SchedulerTreeIndex {
  const nodesById = Object.fromEntries(
    tree.nodes.map((node) => [node.state_id, node]),
  ) as Record<string, SchedulerTreeNode>;
  const orderedIds: string[] = [];
  const seen = new Set<string>();
  const queue: string[] = [tree.root_state_id];

  while (queue.length > 0) {
    const stateId = queue.shift();
    if (!stateId || seen.has(stateId)) {
      continue;
    }
    const node = nodesById[stateId];
    if (!node) {
      continue;
    }
    seen.add(stateId);
    orderedIds.push(stateId);
    for (const childId of node.child_ids) {
      if (!seen.has(childId)) {
        queue.push(childId);
      }
    }
  }

  for (const node of tree.nodes) {
    if (!seen.has(node.state_id)) {
      orderedIds.push(node.state_id);
    }
  }

  return {
    root: nodesById[tree.root_state_id] ?? null,
    nodesById,
    orderedIds,
  };
}

export function collectAutoExpandedNodeIds(
  index: SchedulerTreeIndex,
  selectedStateId: string | null | undefined,
): string[] {
  const targetId = selectedStateId ?? index.root?.state_id;
  if (!targetId) {
    return [];
  }
  const lineage: string[] = [];
  let cursor: SchedulerTreeNode | null = index.nodesById[targetId] ?? null;
  while (cursor) {
    lineage.push(cursor.state_id);
    cursor = cursor.parent_state_id
      ? (index.nodesById[cursor.parent_state_id] ?? null)
      : null;
  }
  return lineage.reverse();
}

export function hasNonTerminalSchedulerNodes(index: SchedulerTreeIndex): boolean {
  return Object.values(index.nodesById).some(
    (node) => !TERMINAL_STATUSES.has(node.status),
  );
}
