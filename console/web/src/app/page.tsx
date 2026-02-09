"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { MessageSquare, Activity, Bot, Zap } from "lucide-react";
import { listSessions, listTraces, listAgents } from "@/lib/api";
import type { SessionSummary, TraceListItem, AgentConfig } from "@/lib/api";

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
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [traces, setTraces] = useState<TraceListItem[]>([]);
  const [agents, setAgents] = useState<AgentConfig[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      listSessions(5).catch(() => []),
      listTraces({ limit: 5 }).catch(() => []),
      listAgents().catch(() => []),
    ]).then(([s, t, a]) => {
      setSessions(s);
      setTraces(t);
      setAgents(a);
      setLoading(false);
    });
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-zinc-500">Loading...</div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-8">
      <div>
        <h1 className="text-2xl font-semibold">Dashboard</h1>
        <p className="text-sm text-zinc-400 mt-1">
          Agiwo Agent SDK overview
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Sessions" value={sessions.length} icon={MessageSquare} href="/sessions" />
        <StatCard label="Traces" value={traces.length} icon={Activity} href="/traces" />
        <StatCard label="Agents" value={agents.length} icon={Bot} href="/agents" />
        <StatCard
          label="Total Tokens"
          value={traces.reduce((sum, t) => sum + t.total_tokens, 0).toLocaleString()}
          icon={Zap}
          href="/traces"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="rounded-lg border border-zinc-800 bg-zinc-900">
          <div className="px-4 py-3 border-b border-zinc-800 flex items-center justify-between">
            <h2 className="text-sm font-medium">Recent Sessions</h2>
            <Link href="/sessions" className="text-xs text-zinc-500 hover:text-zinc-300">
              View all
            </Link>
          </div>
          <div className="divide-y divide-zinc-800">
            {sessions.length === 0 ? (
              <p className="px-4 py-6 text-sm text-zinc-500 text-center">No sessions yet</p>
            ) : (
              sessions.map((s) => (
                <Link
                  key={s.session_id}
                  href={`/sessions/${s.session_id}`}
                  className="block px-4 py-3 hover:bg-zinc-800/50 transition-colors"
                >
                  <p className="text-sm truncate">
                    {s.last_user_input || s.session_id}
                  </p>
                  <p className="text-xs text-zinc-500 mt-1">
                    {s.run_count} runs &middot; {s.agent_id || "unknown"}
                  </p>
                </Link>
              ))
            )}
          </div>
        </div>

        <div className="rounded-lg border border-zinc-800 bg-zinc-900">
          <div className="px-4 py-3 border-b border-zinc-800 flex items-center justify-between">
            <h2 className="text-sm font-medium">Recent Traces</h2>
            <Link href="/traces" className="text-xs text-zinc-500 hover:text-zinc-300">
              View all
            </Link>
          </div>
          <div className="divide-y divide-zinc-800">
            {traces.length === 0 ? (
              <p className="px-4 py-6 text-sm text-zinc-500 text-center">No traces yet</p>
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
                    <span
                      className={`ml-2 text-xs px-1.5 py-0.5 rounded ${
                        t.status === "ok"
                          ? "bg-green-900/50 text-green-400"
                          : t.status === "error"
                          ? "bg-red-900/50 text-red-400"
                          : "bg-zinc-800 text-zinc-400"
                      }`}
                    >
                      {t.status}
                    </span>
                  </div>
                  <p className="text-xs text-zinc-500 mt-1">
                    {t.total_tokens} tokens &middot; {t.total_tool_calls} tools &middot;{" "}
                    {t.duration_ms ? `${Math.round(t.duration_ms)}ms` : "-"}
                  </p>
                </Link>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
