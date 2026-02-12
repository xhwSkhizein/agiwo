"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  listAgentStates,
  getSchedulerStats,
} from "@/lib/api";
import type { AgentStateListItem, SchedulerStats } from "@/lib/api";

const STATUS_STYLES: Record<string, string> = {
  pending: "bg-yellow-900/50 text-yellow-400",
  running: "bg-blue-900/50 text-blue-400",
  sleeping: "bg-purple-900/50 text-purple-400",
  completed: "bg-green-900/50 text-green-400",
  failed: "bg-red-900/50 text-red-400",
};

function StatusBadge({ status }: { status: string }) {
  const cls = STATUS_STYLES[status] || "bg-zinc-800 text-zinc-400";
  return (
    <span className={`text-xs px-1.5 py-0.5 rounded ${cls}`}>{status}</span>
  );
}

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
  if (!wc) return <span className="text-zinc-600">-</span>;
  if (wc.type === "children_complete") {
    return (
      <span className="text-xs text-zinc-400">
        children {wc.completed_children}/{wc.total_children}
      </span>
    );
  }
  if (wc.type === "delay" || wc.type === "interval") {
    const label = wc.time_value != null && wc.time_unit
      ? `${wc.time_value} ${wc.time_unit}`
      : wc.wakeup_at
      ? new Date(wc.wakeup_at).toLocaleTimeString()
      : wc.type;
    return <span className="text-xs text-zinc-400">{wc.type}: {label}</span>;
  }
  return <span className="text-xs text-zinc-400">{wc.type}</span>;
}

const STATUS_FILTERS = [
  { label: "All", value: "" },
  { label: "Pending", value: "pending" },
  { label: "Running", value: "running" },
  { label: "Sleeping", value: "sleeping" },
  { label: "Completed", value: "completed" },
  { label: "Failed", value: "failed" },
];

export default function SchedulerPage() {
  const [states, setStates] = useState<AgentStateListItem[]>([]);
  const [stats, setStats] = useState<SchedulerStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("");

  const fetchData = (statusFilter: string) => {
    setLoading(true);
    const params: { status?: string; limit?: number } = { limit: 200 };
    if (statusFilter) params.status = statusFilter;
    Promise.all([
      listAgentStates(params).catch(() => []),
      getSchedulerStats().catch(() => null),
    ]).then(([s, st]) => {
      setStates(s);
      if (st) setStats(st);
      setLoading(false);
    });
  };

  useEffect(() => {
    fetchData(filter);
  }, [filter]);

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Scheduler</h1>
        <p className="text-sm text-zinc-400 mt-1">
          Agent execution states and orchestration overview
        </p>
      </div>

      {stats && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          <StatMini label="Total" value={stats.total} />
          <StatMini label="Pending" value={stats.pending} active={filter === "pending"} />
          <StatMini label="Running" value={stats.running} active={filter === "running"} />
          <StatMini label="Sleeping" value={stats.sleeping} active={filter === "sleeping"} />
          <StatMini label="Completed" value={stats.completed} active={filter === "completed"} />
          <StatMini label="Failed" value={stats.failed} active={filter === "failed"} />
        </div>
      )}

      <div className="flex items-center gap-2">
        {STATUS_FILTERS.map((f) => (
          <button
            key={f.value}
            onClick={() => setFilter(f.value)}
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
          onClick={() => fetchData(filter)}
          className="ml-auto px-3 py-1.5 text-xs rounded-md bg-zinc-900 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 transition-colors"
        >
          Refresh
        </button>
      </div>

      {loading ? (
        <div className="text-zinc-500">Loading...</div>
      ) : states.length === 0 ? (
        <div className="text-zinc-500 text-center py-12">
          No agent states found
        </div>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-zinc-800">
          <table className="w-full text-sm">
            <thead className="bg-zinc-900 text-zinc-400 text-xs uppercase tracking-wide">
              <tr>
                <th className="text-left px-4 py-3">Agent</th>
                <th className="text-left px-4 py-3">Task</th>
                <th className="text-center px-4 py-3">Status</th>
                <th className="text-left px-4 py-3">Wake Condition</th>
                <th className="text-left px-4 py-3">Parent</th>
                <th className="text-right px-4 py-3">Updated</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800">
              {states.map((s) => (
                <tr
                  key={s.id}
                  className="hover:bg-zinc-900/50 transition-colors"
                >
                  <td className="px-4 py-3">
                    <Link
                      href={`/scheduler/${s.id}`}
                      className="text-zinc-200 hover:text-white font-mono text-xs"
                    >
                      {s.agent_id}
                    </Link>
                  </td>
                  <td className="px-4 py-3 max-w-xs">
                    <span className="truncate block text-zinc-300">
                      {s.task}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-center">
                    <StatusBadge status={s.status} />
                  </td>
                  <td className="px-4 py-3">
                    <WakeInfo wc={s.wake_condition} />
                  </td>
                  <td className="px-4 py-3 text-zinc-500 text-xs font-mono">
                    {s.parent_state_id || "-"}
                  </td>
                  <td className="px-4 py-3 text-right text-zinc-500 text-xs">
                    {s.updated_at
                      ? new Date(s.updated_at).toLocaleString()
                      : "-"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
