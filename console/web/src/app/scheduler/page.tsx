"use client";

import { Suspense, useCallback, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { PaginationControls } from "@/components/pagination-controls";
import { MonoLink, MonoText } from "@/components/mono-text";
import { SchedulerStatusBadge } from "@/components/scheduler-status-badge";
import {
  EmptyStateMessage,
  ErrorStateMessage,
  TextStateMessage,
} from "@/components/state-message";
import { UserInputCompact } from "@/components/user-input-detail";
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

function StatMini({ label, value, active }: { label: string; value: number; active?: boolean }) {
  return (
    <div
      className={`rounded-lg border p-4 transition-colors ${
        active ? "border-zinc-600 bg-zinc-800" : "border-zinc-800 bg-zinc-900"
      }`}
    >
      <p className="text-xs text-zinc-500 uppercase tracking-wide">{label}</p>
      <p className="text-2xl font-semibold mt-1">{value}</p>
    </div>
  );
}

function WakeInfo({ wc }: { wc: AgentStateListItem["wake_condition"] }) {
  return (
    <span className={wc ? "text-xs text-zinc-400" : "text-zinc-600"}>
      {formatWakeConditionSummary(wc)}
    </span>
  );
}

const STATUS_FILTERS = [
  { label: "All", value: "" },
  { label: "Pending", value: "pending" },
  { label: "Running", value: "running" },
  { label: "Waiting", value: "waiting" },
  { label: "Idle", value: "idle" },
  { label: "Queued", value: "queued" },
  { label: "Completed", value: "completed" },
  { label: "Failed", value: "failed" },
];

function SchedulerPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
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
    void loadData();
  }, [loadData]);

  useEffect(() => {
    if (!autoRefresh) {
      return;
    }
    const timerId = window.setInterval(() => {
      void loadData(true);
    }, 10000);
    return () => {
      window.clearInterval(timerId);
    };
  }, [autoRefresh, loadData]);

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Scheduler</h1>
          <p className="text-sm text-zinc-400 mt-1">
            Agent execution states and orchestration overview
          </p>
        </div>
        <div className="flex items-center gap-3 text-sm text-zinc-400">
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(event) => {
                setAutoRefresh(event.target.checked);
              }}
              className="rounded border-zinc-700 bg-zinc-950 text-zinc-200"
            />
            Auto refresh
          </label>
          {refreshing && <span className="text-zinc-500">Refreshing…</span>}
        </div>
      </div>

      {stats && (
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3">
          <StatMini label="Total" value={stats.total} />
          <StatMini label="Pending" value={stats.pending} active={filter === "pending"} />
          <StatMini label="Running" value={stats.running} active={filter === "running"} />
          <StatMini label="Waiting" value={stats.waiting} active={filter === "waiting"} />
          <StatMini label="Idle" value={stats.idle} active={filter === "idle"} />
          <StatMini label="Queued" value={stats.queued} active={filter === "queued"} />
          <StatMini label="Completed" value={stats.completed} active={filter === "completed"} />
          <StatMini label="Failed" value={stats.failed} active={filter === "failed"} />
        </div>
      )}

      <div className="flex items-center gap-2">
        {STATUS_FILTERS.map((f) => (
          <button
            key={f.value}
            onClick={() => {
              setFilter(f.value);
              setOffset(0);
              updateQuery({ status: f.value || null, offset: 0, limit: pageSize });
            }}
            className={`px-3 py-1.5 text-xs rounded-md transition-colors ${
              filter === f.value
                ? "bg-zinc-700 text-white"
                : "bg-zinc-900 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
            }`}
          >
            {f.label}
          </button>
        ))}
        <button
          onClick={() => {
            void loadData();
          }}
          className="ml-auto px-3 py-1.5 text-xs rounded-md bg-zinc-900 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 transition-colors"
        >
          Refresh
        </button>
      </div>

      {error && <ErrorStateMessage>{error}</ErrorStateMessage>}

      {loading ? (
        <TextStateMessage>Loading...</TextStateMessage>
      ) : states.length === 0 ? (
        <EmptyStateMessage>
          No agent states found
        </EmptyStateMessage>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-zinc-800">
          <table className="w-full text-sm">
            <thead className="bg-zinc-900 text-zinc-400 text-xs uppercase tracking-wide">
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
            <tbody className="divide-y divide-zinc-800">
              {states.map((s) => {
                const metrics = normalizeRunMetricsSummary(s.metrics);
                return (
                  <tr
                    key={s.id}
                    className="hover:bg-zinc-900/50 transition-colors"
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
                          className="text-[11px] text-zinc-500 hover:text-zinc-200"
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
                    <td className="px-4 py-3 text-right text-zinc-200">
                      {formatUsd(metrics.token_cost)}
                    </td>
                    <td className="px-4 py-3 text-right text-zinc-400 text-xs">
                      {formatTokenCount(metrics.input_tokens)} / {formatTokenCount(metrics.output_tokens)} / {formatTokenCount(metrics.total_tokens)}
                    </td>
                    <td className="px-4 py-3 text-right text-zinc-500 text-xs">
                      {formatTokenCount(metrics.cache_read_tokens)} / {formatTokenCount(metrics.cache_creation_tokens)}
                    </td>
                    <td className="px-4 py-3 text-right text-zinc-500 text-xs">
                      {metrics.run_count}
                    </td>
                    <td className="px-4 py-3">
                      <WakeInfo wc={s.wake_condition} />
                    </td>
                    <td className="px-4 py-3 text-zinc-500">
                      <MonoText className="text-zinc-500 text-xs font-mono">
                        {s.parent_id || "-"}
                      </MonoText>
                    </td>
                    <td className="px-4 py-3 text-right text-zinc-500 text-xs">
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

export default function SchedulerPage() {
  return (
    <Suspense fallback={<TextStateMessage>Loading...</TextStateMessage>}>
      <SchedulerPageContent />
    </Suspense>
  );
}
