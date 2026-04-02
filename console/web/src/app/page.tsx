"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { MessageSquare, Activity, Bot, Zap, CalendarClock } from "lucide-react";
import { getDashboardOverview, listSessions, listTraces } from "@/lib/api";
import {
  EmptyStateMessage,
  ErrorStateMessage,
  FullPageMessage,
} from "@/components/state-message";
import { SectionCard } from "@/components/section-card";
import { TraceStatusBadge } from "@/components/trace-status-badge";
import { UserInputCompact } from "@/components/user-input-detail";
import type {
  DashboardOverview,
  SessionSummary,
  TraceListItem,
} from "@/lib/api";
import { formatRoundedMs } from "@/lib/time";

function StatCard({
  label,
  value,
  icon: Icon,
  href,
}: {
  label: string;
  value: number | string;
  icon: React.ComponentType<{ className?: string }>;
  href: string;
}) {
  return (
    <Link
      href={href}
      className="rounded-lg border border-zinc-800 bg-zinc-900 p-5 hover:border-zinc-700 transition-colors"
    >
      <div className="flex items-center justify-between">
        <p className="text-sm text-zinc-400">{label}</p>
        <Icon className="w-4 h-4 text-zinc-500" />
      </div>
      <p className="text-2xl font-semibold mt-2">{value}</p>
    </Link>
  );
}

export default function DashboardPage() {
  const [overview, setOverview] = useState<DashboardOverview | null>(null);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [traces, setTraces] = useState<TraceListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadDashboard = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [nextOverview, nextSessions, nextTraces] = await Promise.all([
        getDashboardOverview(),
        listSessions(5, 0),
        listTraces({ limit: 5, offset: 0 }),
      ]);
      setOverview(nextOverview);
      setSessions(nextSessions);
      setTraces(nextTraces);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load dashboard");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadDashboard();
  }, [loadDashboard]);

  if (loading && overview === null) {
    return <FullPageMessage>Loading...</FullPageMessage>;
  }

  const schedulerStats = overview?.scheduler;

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-8">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Dashboard</h1>
          <p className="text-sm text-zinc-400 mt-1">
            Real-time control plane overview and recent activity
          </p>
        </div>
        <button
          type="button"
          onClick={() => {
            void loadDashboard();
          }}
          className="rounded-md border border-zinc-700 px-3 py-1.5 text-sm text-zinc-300 transition-colors hover:border-zinc-500 hover:text-white"
        >
          Refresh
        </button>
      </div>

      {error && <ErrorStateMessage>{error}</ErrorStateMessage>}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        <StatCard
          label="Sessions"
          value={overview?.total_sessions ?? 0}
          icon={MessageSquare}
          href="/sessions"
        />
        <StatCard
          label="Traces"
          value={overview?.total_traces ?? 0}
          icon={Activity}
          href="/traces"
        />
        <StatCard
          label="Agents"
          value={overview?.total_agents ?? 0}
          icon={Bot}
          href="/agents"
        />
        <StatCard
          label="Scheduled"
          value={
            schedulerStats
              ? `${schedulerStats.running + schedulerStats.waiting + schedulerStats.queued} active`
              : "0"
          }
          icon={CalendarClock}
          href="/scheduler"
        />
        <StatCard
          label="Total Tokens"
          value={(overview?.total_tokens ?? 0).toLocaleString()}
          icon={Zap}
          href="/traces"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <SectionCard
          title="Recent Sessions"
          action={
            <Link href="/sessions" className="text-xs text-zinc-500 hover:text-zinc-300">
              View all
            </Link>
          }
        >
          <div className="divide-y divide-zinc-800">
            {sessions.length === 0 ? (
              <EmptyStateMessage className="px-4 py-6 text-sm text-zinc-500 text-center">
                No sessions yet
              </EmptyStateMessage>
            ) : (
              sessions.map((s) => (
                <Link
                  key={s.session_id}
                  href={`/sessions/${s.session_id}`}
                  className="block px-4 py-3 hover:bg-zinc-800/50 transition-colors"
                >
                  <div className="text-sm">
                    <UserInputCompact input={s.last_user_input} maxLength={80} />
                  </div>
                  <p className="text-xs text-zinc-500 mt-1">
                    {s.run_count} runs &middot; {s.agent_id || "unknown"}
                  </p>
                </Link>
              ))
            )}
          </div>
        </SectionCard>

        <SectionCard
          title="Recent Traces"
          action={
            <Link href="/traces" className="text-xs text-zinc-500 hover:text-zinc-300">
              View all
            </Link>
          }
        >
          <div className="divide-y divide-zinc-800">
            {traces.length === 0 ? (
              <EmptyStateMessage className="px-4 py-6 text-sm text-zinc-500 text-center">
                No traces yet
              </EmptyStateMessage>
            ) : (
              traces.map((t) => (
                <Link
                  key={t.trace_id}
                  href={`/traces/${t.trace_id}`}
                  className="block px-4 py-3 hover:bg-zinc-800/50 transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <p className="text-sm truncate flex-1">
                      {t.input_query || t.trace_id}
                    </p>
                    <div className="ml-2">
                      <TraceStatusBadge status={t.status} />
                    </div>
                  </div>
                  <p className="text-xs text-zinc-500 mt-1">
                    {t.total_tokens} tokens &middot; {t.total_tool_calls} tools &middot;{" "}
                    {formatRoundedMs(t.duration_ms)}
                  </p>
                </Link>
              ))
            )}
          </div>
        </SectionCard>
      </div>
    </div>
  );
}
