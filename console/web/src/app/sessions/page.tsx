"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { listSessions } from "@/lib/api";
import type { SessionSummary } from "@/lib/api";

export default function SessionsPage() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listSessions(50)
      .then(setSessions)
      .catch(() => setSessions([]))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Sessions</h1>
        <p className="text-sm text-zinc-400 mt-1">All conversation sessions</p>
      </div>

      {loading ? (
        <div className="text-zinc-500">Loading...</div>
      ) : sessions.length === 0 ? (
        <div className="text-zinc-500 text-center py-12">No sessions found</div>
      ) : (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 divide-y divide-zinc-800">
          {sessions.map((s) => (
            <Link
              key={s.session_id}
              href={`/sessions/${s.session_id}`}
              className="block px-5 py-4 hover:bg-zinc-800/50 transition-colors"
            >
              <div className="flex items-center justify-between">
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">
                    {s.last_user_input || s.session_id}
                  </p>
                  {s.last_response && (
                    <p className="text-xs text-zinc-500 mt-1 truncate">
                      {s.last_response}
                    </p>
                  )}
                </div>
                <div className="ml-4 text-right shrink-0">
                  <p className="text-xs text-zinc-400">{s.run_count} runs</p>
                  <p className="text-xs text-zinc-600 mt-0.5">
                    {s.agent_id || "unknown"}
                  </p>
                </div>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
