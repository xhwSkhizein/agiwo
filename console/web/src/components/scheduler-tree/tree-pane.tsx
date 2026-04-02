"use client";

import { ChevronDown, ChevronRight } from "lucide-react";

import { SchedulerStatusBadge } from "@/components/scheduler-status-badge";
import type { SchedulerTreeNode } from "@/lib/api";
import type { SchedulerTreeIndex } from "@/lib/scheduler-tree";

type TreePaneProps = {
  index: SchedulerTreeIndex;
  selectedStateId: string | null;
  expandedIds: Set<string>;
  onSelect: (stateId: string) => void;
  onToggle: (stateId: string) => void;
};

function flattenVisibleNodeIds(
  index: SchedulerTreeIndex,
  expandedIds: Set<string>,
): string[] {
  const visible: string[] = [];
  const root = index.root;
  if (!root) {
    return visible;
  }

  const queue: Array<{ node: SchedulerTreeNode; parentVisible: boolean }> = [
    { node: root, parentVisible: true },
  ];
  while (queue.length > 0) {
    const current = queue.shift();
    if (!current || !current.parentVisible) {
      continue;
    }
    visible.push(current.node.state_id);
    const isExpanded = expandedIds.has(current.node.state_id);
    if (!isExpanded) {
      continue;
    }
    for (const childId of current.node.child_ids) {
      const child = index.nodesById[childId];
      if (child) {
        queue.push({ node: child, parentVisible: true });
      }
    }
  }

  return visible;
}

export function TreePane({
  index,
  selectedStateId,
  expandedIds,
  onSelect,
  onToggle,
}: TreePaneProps) {
  const visibleIds = flattenVisibleNodeIds(index, expandedIds);

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900">
      <div className="border-b border-zinc-800 px-4 py-3">
        <h2 className="text-sm font-medium">Tree</h2>
      </div>
      <div className="max-h-[calc(100vh-18rem)] overflow-auto p-2">
        {visibleIds.map((stateId) => {
          const node = index.nodesById[stateId];
          const isExpanded = expandedIds.has(stateId);
          const isSelected = stateId === selectedStateId;
          const hasChildren = node.child_ids.length > 0;

          return (
            <div
              key={stateId}
              className={`rounded-md px-2 py-1.5 ${
                isSelected ? "bg-zinc-800" : "hover:bg-zinc-800/60"
              }`}
            >
              <div
                className="flex items-center gap-2"
                style={{ paddingLeft: `${node.depth * 16}px` }}
              >
                {hasChildren ? (
                  <button
                    type="button"
                    aria-label={isExpanded ? `Collapse ${stateId}` : `Expand ${stateId}`}
                    onClick={() => onToggle(stateId)}
                    className="rounded p-0.5 text-zinc-400 hover:bg-zinc-700 hover:text-zinc-100"
                  >
                    {isExpanded ? (
                      <ChevronDown className="h-3.5 w-3.5" />
                    ) : (
                      <ChevronRight className="h-3.5 w-3.5" />
                    )}
                  </button>
                ) : (
                  <span className="w-4" />
                )}

                <button
                  type="button"
                  aria-label={`Select ${stateId}`}
                  onClick={() => onSelect(stateId)}
                  className="flex min-w-0 flex-1 items-center gap-2 text-left"
                >
                  <span className="truncate font-mono text-xs text-zinc-100">
                    {stateId}
                  </span>
                  <SchedulerStatusBadge status={node.status} />
                  {node.pending_event_count > 0 && (
                    <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-300">
                      {node.pending_event_count} pending
                    </span>
                  )}
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
