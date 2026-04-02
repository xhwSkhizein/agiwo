"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { BackHeader } from "@/components/back-header";
import { MetricCard } from "@/components/metric-card";
import { MonoLink } from "@/components/mono-text";
import { ErrorStateMessage, FullPageMessage } from "@/components/state-message";
import {
  ApiError,
  getAgentState,
  getPendingEvents,
  getSchedulerTree,
} from "@/lib/api";
import type { AgentStateDetail, PendingEventItem, SchedulerTree } from "@/lib/api";
import {
  buildSchedulerTreeIndex,
  collectAutoExpandedNodeIds,
  hasNonTerminalSchedulerNodes,
} from "@/lib/scheduler-tree";
import { formatLocalDateTime } from "@/lib/time";

import { NodeInspector } from "./node-inspector";
import { RootActions } from "./root-actions";
import { TreePane } from "./tree-pane";

type SchedulerTreeWorkspaceProps = {
  rootStateId: string;
  selectedStateId: string | null;
  onSelectedStateIdChange: (stateId: string) => void;
};

function describeTreeError(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 404) {
      return "Scheduler state not found or already removed.";
    }
    return error.detail;
  }
  return error instanceof Error ? error.message : "Failed to load scheduler tree.";
}

export function SchedulerTreeWorkspace({
  rootStateId,
  selectedStateId,
  onSelectedStateIdChange,
}: SchedulerTreeWorkspaceProps) {
  const [tree, setTree] = useState<SchedulerTree | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const [selectedDetail, setSelectedDetail] = useState<AgentStateDetail | null>(null);
  const [selectedPendingEvents, setSelectedPendingEvents] = useState<PendingEventItem[]>(
    [],
  );
  const [selectedLoading, setSelectedLoading] = useState(false);
  const [selectedError, setSelectedError] = useState<string | null>(null);

  const index = useMemo(
    () => (tree ? buildSchedulerTreeIndex(tree) : null),
    [tree],
  );

  const effectiveSelectedStateId = useMemo(() => {
    if (selectedStateId && index?.nodesById[selectedStateId]) {
      return selectedStateId;
    }
    return tree?.root_state_id ?? null;
  }, [index, selectedStateId, tree]);

  const refreshTree = useCallback(
    async (background = false) => {
      if (background) {
        setRefreshing(true);
      } else {
        setLoading(true);
      }
      try {
        const nextTree = await getSchedulerTree(rootStateId);
        setTree(nextTree);
        setError(null);
      } catch (err) {
        setTree(null);
        setError(describeTreeError(err));
      } finally {
        setLoading(false);
        setRefreshing(false);
      }
    },
    [rootStateId],
  );

  useEffect(() => {
    void refreshTree();
  }, [refreshTree]);

  useEffect(() => {
    if (!index) {
      return;
    }
    const autoExpanded = collectAutoExpandedNodeIds(index, effectiveSelectedStateId);
    setExpandedIds((current) => {
      const next = new Set(current);
      for (const stateId of autoExpanded) {
        next.add(stateId);
      }
      return next;
    });
  }, [effectiveSelectedStateId, index]);

  useEffect(() => {
    if (!index || !effectiveSelectedStateId) {
      setSelectedDetail(null);
      setSelectedPendingEvents([]);
      setSelectedError(null);
      return;
    }

    let cancelled = false;
    setSelectedLoading(true);
    setSelectedError(null);
    void Promise.all([
      getAgentState(effectiveSelectedStateId),
      getPendingEvents(effectiveSelectedStateId),
    ])
      .then(([detail, pendingEvents]) => {
        if (cancelled) {
          return;
        }
        setSelectedDetail(detail);
        setSelectedPendingEvents(pendingEvents);
      })
      .catch((err) => {
        if (cancelled) {
          return;
        }
        setSelectedDetail(null);
        setSelectedPendingEvents([]);
        setSelectedError(
          err instanceof ApiError ? err.detail : err instanceof Error ? err.message : "Failed to load selected node detail.",
        );
      })
      .finally(() => {
        if (!cancelled) {
          setSelectedLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [effectiveSelectedStateId, index]);

  useEffect(() => {
    if (!index || !autoRefresh || !hasNonTerminalSchedulerNodes(index)) {
      return;
    }
    const timerId = window.setInterval(() => {
      void refreshTree(true);
    }, 3000);
    return () => {
      window.clearInterval(timerId);
    };
  }, [autoRefresh, index, refreshTree]);

  const selectedNode = effectiveSelectedStateId
    ? (index?.nodesById[effectiveSelectedStateId] ?? null)
    : null;
  const showRootActions =
    !!selectedDetail &&
    !!tree &&
    selectedDetail.id === tree.root_state_id &&
    selectedNode?.state_id === tree.root_state_id;

  if (loading && !tree) {
    return <FullPageMessage>Loading scheduler tree...</FullPageMessage>;
  }

  if (!tree || !index) {
    return <FullPageMessage>{error ?? "Scheduler tree unavailable."}</FullPageMessage>;
  }

  return (
    <div className="mx-auto max-w-7xl space-y-6 p-6">
      <BackHeader
        href={`/scheduler/${rootStateId}`}
        title="Scheduler Tree"
        subtitle={rootStateId}
        rightContent={
          <>
            <label className="flex items-center gap-2 text-xs text-zinc-400">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(event) => setAutoRefresh(event.target.checked)}
                className="rounded border-zinc-700 bg-zinc-950 text-zinc-200"
              />
              Auto refresh
            </label>
            <button
              type="button"
              onClick={() => void refreshTree(true)}
              className="rounded border border-zinc-800 px-3 py-1.5 text-xs text-zinc-300 hover:border-zinc-700 hover:text-zinc-100"
            >
              Refresh
            </button>
          </>
        }
      />

      <div className="flex flex-wrap gap-2 text-xs text-zinc-500">
        <MonoLink href={`/scheduler/${rootStateId}`}>Open compact detail</MonoLink>
        {tree.root_session_id && (
          <MonoLink href={`/sessions/${tree.root_session_id}`}>
            Open session detail
          </MonoLink>
        )}
        <span className="text-zinc-600">
          Last tree snapshot: {formatLocalDateTime(tree.generated_at)}
        </span>
        {refreshing && <span className="text-zinc-600">Refreshing…</span>}
      </div>

      {error && <ErrorStateMessage>{error}</ErrorStateMessage>}

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-8">
        <MetricCard label="Total" value={String(tree.stats.total)} />
        <MetricCard label="Running" value={String(tree.stats.running)} />
        <MetricCard label="Waiting" value={String(tree.stats.waiting)} />
        <MetricCard label="Queued" value={String(tree.stats.queued)} />
        <MetricCard label="Idle" value={String(tree.stats.idle)} />
        <MetricCard label="Completed" value={String(tree.stats.completed)} />
        <MetricCard label="Failed" value={String(tree.stats.failed)} />
        <MetricCard label="Cancelled" value={String(tree.stats.cancelled)} />
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)]">
        <TreePane
          index={index}
          selectedStateId={effectiveSelectedStateId}
          expandedIds={expandedIds}
          onSelect={onSelectedStateIdChange}
          onToggle={(stateId) => {
            setExpandedIds((current) => {
              const next = new Set(current);
              if (next.has(stateId)) {
                next.delete(stateId);
              } else {
                next.add(stateId);
              }
              return next;
            });
          }}
        />

        <div className="space-y-4">
          {showRootActions && selectedDetail && (
            <RootActions
              state={selectedDetail}
              onActionComplete={async () => {
                await refreshTree(true);
              }}
            />
          )}

          <NodeInspector
            node={selectedNode}
            detail={selectedDetail}
            pendingEvents={selectedPendingEvents}
            loading={selectedLoading}
            error={selectedError}
          />
        </div>
      </div>
    </div>
  );
}
