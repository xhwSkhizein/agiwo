"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { RefreshCw } from "lucide-react";
import { PaginationControls } from "@/components/pagination-controls";
import { MonoLink, MonoText } from "@/components/mono-text";
import { SchedulerStatusBadge } from "@/components/scheduler-status-badge";
import {
  EmptyStateMessage,
  ErrorStateMessage,
  FullPageMessage,
} from "@/components/state-message";
import { UserInputCompact } from "@/components/user-input-detail";
import { cn } from "@/lib/utils";
import {
  listAgentStates,
  getSchedulerStats,
} from "@/lib/api";
import type { AgentStateListItem, SchedulerStats } from "@/lib/api";
import {
  formatTokenCount,
  formatUsd,
  normalizeRunMetricsSummary,
} from "@/lib/metrics";
import { formatLocalDateTime } from "@/lib/time";
import { formatWakeConditionSummary } from "@/lib/wake-condition";
import { usePageVisibility } from "@/hooks/use-page-visibility";

/**
 * Render a clickable metric tile that displays an uppercase label and a large numeric value.
 *
 * @param label - Short label shown above the value (displayed in uppercase).
 * @param value - Numeric metric to display prominently.
 * @param active - If true, visually marks the tile as active and sets `aria-pressed` accordingly.
 * @param onClick - Callback invoked when the tile is clicked.
 * @returns The rendered button element representing the metric tile.
 */
function StatMini({
  label,
  value,
  active,
  onClick,
}: {
  label: string;
  value: number;
  active?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-pressed={active}
      onClick={onClick}
      className={cn(
        "rounded-lg border p-4 text-left transition-all duration-150",
        active
          ? "border-line-strong bg-panel-strong"
          : "border-line bg-panel hover:border-line-strong hover:bg-panel-muted"
      )}
    >
      <p className="text-xs font-medium uppercase tracking-wide text-ink-faint">{label}</p>
      <p className={cn("mt-1 text-2xl font-semibold", active ? "text-foreground" : "text-ink-soft")}>
        {value}
      </p>
    </button>
  );
}

/**
 * Render a compact, styled summary of an agent state's wake condition.
 *
 * @param wc - The wake condition value from an agent state (may be `null` or `undefined`)
 * @returns A <span> element containing the formatted wake-condition text; uses muted styling when `wc` is present and faint styling when absent.
 */
function WakeInfo({ wc }: { wc: AgentStateListItem["wake_condition"] }) {
  return (
    <span className={wc ? "text-xs text-ink-muted" : "text-ink-faint"}>
      {formatWakeConditionSummary(wc)}
    </span>
  );
}

/**
 * Render the scheduler page content including agent state metrics, the agent states table, filters, pagination, and refresh controls.
 *
 * Manages loading scheduler stats and agent states, applies status filtering and pagination, and supports optional auto-refresh when the page is visible.
 *
 * @returns The rendered scheduler page content as a React element.
 */
function SchedulerPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const isPageVisible = usePageVisibility();
  const [states, setStates] = useState<AgentStateListItem[]>([]);
  const [stats, setStats] = useState<SchedulerStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState(searchParams.get("status") || "");
  const [pageSize, setPageSize] = useState(Number(searchParams.get("limit") || "25"));
  const [offset, setOffset] = useState(Number(searchParams.get("offset") || "0"));
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [hasMore, setHasMore] = useState(false);
  const hasLoadedRef = useRef(false);

  const updateQuery = useCallback(
    (next: Record<string, string | number | null>) => {
      const query = new URLSearchParams(searchParams.toString());
      for (const [key, value] of Object.entries(next)) {
        if (value === null || value === "") {
          query.delete(key);
        } else {
          query.set(key, String(value));
        }
      }
      router.replace(`/scheduler?${query.toString()}`);
    },
    [router, searchParams],
  );

  const loadData = useCallback(async (background = false) => {
    if (background) {
      setRefreshing(true);
    } else {
      setLoading(true);
    }
    setError(null);
    const params: { status?: string; limit: number; offset: number } = {
      limit: pageSize,
      offset,
    };
    if (filter) {
      const statusFilter = filter;
      params.status = statusFilter;
    }

    try {
      const [nextStates, nextStats] = await Promise.all([
        listAgentStates(params),
        getSchedulerStats(),
      ]);
      setStates(nextStates.items);
      setHasMore(nextStates.has_more);
      setStats(nextStats);
    } catch (err) {
      setStates([]);
      setHasMore(false);
      setError(err instanceof Error ? err.message : "Failed to load scheduler states");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [filter, offset, pageSize]);

  useEffect(() => {
    if (hasLoadedRef.current) {
      void loadData();
    }
  }, [filter, offset, pageSize]);

  useEffect(() => {
    if (hasLoadedRef.current) {
      return;
    }
    hasLoadedRef.current = true;
    void loadData();
  }, []);

  useEffect(() => {
    if (!autoRefresh || !isPageVisible) {
      return;
    }
    const timerId = window.setInterval(() => {
      void loadData(true);
    }, 10000);
    return () => {
      window.clearInterval(timerId);
    };
  }, [autoRefresh, isPageVisible, loadData]);

  const applyFilter = (nextFilter: string) => {
    setFilter(nextFilter);
    setOffset(0);
    updateQuery({ status: nextFilter || null, offset: 0, limit: pageSize });
  };

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Scheduler</h1>
          <p className="mt-1 text-sm text-ink-muted">
            Agent execution states and orchestration overview
          </p>
        </div>
        <div className="flex items-center gap-3 text-sm text-ink-muted">
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(event) => {
                setAutoRefresh(event.target.checked);
              }}
              className="h-4 w-4 rounded border-line bg-panel-strong text-accent"
            />
            Auto refresh
          </label>
          {!isPageVisible && autoRefresh && (
            <span className="text-ink-faint">Paused in background</span>
          )}
          {refreshing && isPageVisible && <span className="text-ink-faint">Refreshing…</span>}
        </div>
      </div>

      {stats && (
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3">
          <StatMini label="Total" value={stats.total} active={filter === ""} onClick={() => applyFilter("")} />
          <StatMini label="Pending" value={stats.pending} active={filter === "pending"} onClick={() => applyFilter("pending")} />
          <StatMini label="Running" value={stats.running} active={filter === "running"} onClick={() => applyFilter("running")} />
          <StatMini label="Waiting" value={stats.waiting} active={filter === "waiting"} onClick={() => applyFilter("waiting")} />
          <StatMini label="Idle" value={stats.idle} active={filter === "idle"} onClick={() => applyFilter("idle")} />
          <StatMini label="Queued" value={stats.queued} active={filter === "queued"} onClick={() => applyFilter("queued")} />
          <StatMini label="Completed" value={stats.completed} active={filter === "completed"} onClick={() => applyFilter("completed")} />
          <StatMini label="Failed" value={stats.failed} active={filter === "failed"} onClick={() => applyFilter("failed")} />
        </div>
      )}

      <div className="flex items-center gap-2">
        <span className="text-xs text-ink-muted">
          {filter ? `Filtered by ${filter}` : "All scheduler states"}
        </span>
        <button
          type="button"
          onClick={() => {
            void loadData();
          }}
          disabled={loading || refreshing}
          className="ui-button ui-button-secondary min-h-9 px-3 py-1.5 text-xs"
        >
          <RefreshCw className={cn("w-3 h-3", (loading || refreshing) && "animate-spin")} />
          Refresh
        </button>
      </div>

      {error && <ErrorStateMessage>{error}</ErrorStateMessage>}

      {loading ? (
        <FullPageMessage loading>Loading scheduler states...</FullPageMessage>
      ) : states.length === 0 ? (
        <EmptyStateMessage>
          <div className="space-y-2">
            <p>No agent states found</p>
            {filter && <p className="text-ink-faint">Try clearing the status filter</p>}
          </div>
        </EmptyStateMessage>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-line bg-panel">
          <table className="w-full text-sm">
            <thead className="bg-panel-muted text-xs uppercase tracking-wide text-ink-faint">
              <tr>
                <th className="text-left px-4 py-3">Agent</th>
                <th className="text-left px-4 py-3">Task</th>
                <th className="text-center px-4 py-3">Status</th>
                <th className="text-right px-4 py-3">Cost</th>
                <th className="text-right px-4 py-3">Tokens(In/Out/Total)</th>
                <th className="text-right px-4 py-3">Cache R/C</th>
                <th className="text-right px-4 py-3">Runs</th>
                <th className="text-left px-4 py-3">Wake Condition</th>
                <th className="text-left px-4 py-3">Parent</th>
                <th className="text-right px-4 py-3">Updated</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line">
              {states.map((s) => {
                const metrics = normalizeRunMetricsSummary(s.metrics);
                return (
                  <tr
                    key={s.id}
                    className="transition-colors hover:bg-panel-muted"
                  >
                    <td className="px-4 py-3">
                      <div className="flex flex-col gap-1">
                        <MonoLink href={`/scheduler/${s.id}`}>
                          {s.id}
                        </MonoLink>
                        <MonoLink
                          href={`/scheduler/${s.root_state_id ?? s.id}/tree${
                            s.root_state_id && s.root_state_id !== s.id
                              ? `?selected=${s.id}`
                              : ""
                          }`}
                          className="text-[11px] text-ink-faint hover:text-foreground"
                        >
                          tree
                        </MonoLink>
                      </div>
                    </td>
                    <td className="px-4 py-3 max-w-xs">
                      <UserInputCompact input={s.task} maxLength={60} />
                    </td>
                    <td className="px-4 py-3 text-center">
                      <SchedulerStatusBadge status={s.status} />
                    </td>
                    <td className="px-4 py-3 text-right text-ink-soft">
                      {formatUsd(metrics.token_cost)}
                    </td>
                    <td className="px-4 py-3 text-right text-xs text-ink-muted">
                      {formatTokenCount(metrics.input_tokens)} / {formatTokenCount(metrics.output_tokens)} / {formatTokenCount(metrics.total_tokens)}
                    </td>
                    <td className="px-4 py-3 text-right text-xs text-ink-faint">
                      {formatTokenCount(metrics.cache_read_tokens)} / {formatTokenCount(metrics.cache_creation_tokens)}
                    </td>
                    <td className="px-4 py-3 text-right text-xs text-ink-faint">
                      {metrics.run_count}
                    </td>
                    <td className="px-4 py-3">
                      <WakeInfo wc={s.wake_condition} />
                    </td>
                    <td className="px-4 py-3 text-ink-faint">
                      <MonoText className="text-xs font-mono text-ink-faint">
                        {s.parent_id || "-"}
                      </MonoText>
                    </td>
                    <td className="px-4 py-3 text-right text-xs text-ink-faint">
                      {formatLocalDateTime(s.updated_at)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      <PaginationControls
        offset={offset}
        pageSize={pageSize}
        itemCount={states.length}
        hasMore={hasMore}
        itemLabel="states"
        disabled={loading}
        onPageSizeChange={(nextPageSize) => {
          setPageSize(nextPageSize);
          setOffset(0);
          updateQuery({ limit: nextPageSize, offset: 0 });
        }}
        onPrevious={() => {
          setOffset((current) => {
            const nextOffset = Math.max(0, current - pageSize);
            updateQuery({ offset: nextOffset, limit: pageSize });
            return nextOffset;
          });
        }}
        onNext={() => {
          setOffset((current) => {
            const nextOffset = current + pageSize;
            updateQuery({ offset: nextOffset, limit: pageSize });
            return nextOffset;
          });
        }}
      />
    </div>
  );
}

/**
 * Renders the Scheduler page wrapped in a Suspense fallback.
 *
 * @returns A React element containing the scheduler content and a loading fallback message.
 */
export default function SchedulerPage() {
  return (
    <Suspense fallback={<FullPageMessage loading>Loading scheduler...</FullPageMessage>}>
      <SchedulerPageContent />
    </Suspense>
  );
}
