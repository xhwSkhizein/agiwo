"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { MessageSquare, Activity, Bot, Zap, CalendarClock, RefreshCw } from "lucide-react";
import { getDashboardOverview, listSessions, listTraces } from "@/lib/api";
import {
  EmptyStateMessage,
  ErrorStateMessage,
  FullPageMessage,
} from "@/components/state-message";
import { SectionCard } from "@/components/section-card";
import { TraceStatusBadge } from "@/components/trace-status-badge";
import { UserInputCompact } from "@/components/user-input-detail";
import { PillBadge } from "@/components/pill-badge";
import { cn } from "@/lib/utils";
import type {
  DashboardOverview,
  SessionSummary,
  TraceListItem,
} from "@/lib/api";
import { formatRoundedMs } from "@/lib/time";

/**
 * Renders a clickable statistic card that links to `href` and displays a label, an icon, and a value.
 *
 * @param label - Text label shown above the value
 * @param value - Numeric or string value displayed prominently
 * @param icon - Icon component to render to the right of the label
 * @param href - Destination URL for the card link
 * @returns A link element styled as a statistic card containing the provided label, icon, and value
 */
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
      className={cn(
        "group rounded-lg border border-zinc-800 bg-zinc-900/50 p-5",
        "transition-all duration-200 hover:border-zinc-700 hover:bg-zinc-900"
      )}
    >
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium text-zinc-400">{label}</p>
        <Icon className="w-4 h-4 text-zinc-500 transition-colors duration-150 group-hover:text-zinc-400" />
      </div>
      <p className="text-2xl font-semibold mt-2 text-zinc-100">{value}</p>
    </Link>
  );
}

/**
 * Renders the dashboard page with top-level statistics, recent sessions, and recent traces.
 *
 * Loads overview, recent sessions, and recent traces on mount; shows a full-page loading message while initially loading, displays an error message if loading fails, and provides a manual refresh control.
 *
 * @returns The rendered dashboard page element containing stats, recent activity lists, and controls.
 */
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
      setSessions(nextSessions.items);
      setTraces(nextTraces.items);
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
    return <FullPageMessage loading>Loading dashboard...</FullPageMessage>;
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
          disabled={loading}
          className={cn(
            "inline-flex items-center gap-2 rounded-md border px-3 py-1.5 text-sm",
            "transition-all duration-150",
            "border-zinc-700 text-zinc-300 hover:border-zinc-500 hover:text-white",
            "disabled:opacity-50 disabled:cursor-not-allowed"
          )}
        >
          <RefreshCw className={cn("w-3.5 h-3.5", loading && "animate-spin")} />
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
                  className={cn(
                    "group block px-4 py-3 transition-colors duration-150",
                    "hover:bg-zinc-800/50"
                  )}
                >
                  <div className="text-sm text-zinc-200">
                    <UserInputCompact input={s.last_user_input} maxLength={80} />
                  </div>
                  <div className="flex items-center gap-2 mt-1.5">
                    <PillBadge variant="default">{s.run_count} runs</PillBadge>
                    <span className="text-xs text-zinc-500">
                      {s.agent_id || "unknown"}
                    </span>
                  </div>
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
                  className={cn(
                    "group block px-4 py-3 transition-colors duration-150",
                    "hover:bg-zinc-800/50"
                  )}
                >
                  <div className="flex items-center justify-between gap-3">
                    <p className="text-sm text-zinc-200 truncate flex-1">
                      {t.input_query || t.trace_id}
                    </p>
                    <TraceStatusBadge status={t.status} />
                  </div>
                  <div className="flex items-center gap-2 mt-1.5 text-xs text-zinc-500">
                    <span>{t.total_tokens.toLocaleString()} tokens</span>
                    <span className="text-zinc-700">·</span>
                    <span>{t.total_tool_calls} tools</span>
                    <span className="text-zinc-700">·</span>
                    <span>{formatRoundedMs(t.duration_ms)}</span>
                  </div>
                </Link>
              ))
            )}
          </div>
        </SectionCard>
      </div>
    </div>
  );
}
