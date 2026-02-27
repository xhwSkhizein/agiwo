"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { listTraces } from "@/lib/api";
import type { TraceListItem } from "@/lib/api";

function StatusBadge({ status }: { status: string }) {
  const cls =
    status === "ok"
      ? "bg-green-900/50 text-green-400"
      : status === "error"
      ? "bg-red-900/50 text-red-400"
      : status === "running"
      ? "bg-blue-900/50 text-blue-400"
      : "bg-zinc-800 text-zinc-400";
  return (
    <span className={`text-xs px-1.5 py-0.5 rounded ${cls}`}>{status}</span>
  );
}

export default function TracesPage() {
  const [traces, setTraces] = useState<TraceListItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listTraces({ limit: 100 })
      .then(setTraces)
      .catch(() => setTraces([]))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Traces</h1>
        <p className="text-sm text-zinc-400 mt-1">
          Agent execution traces with timing and token metrics
        </p>
      </div>

      {loading ? (
        <div className="text-zinc-500">Loading...</div>
      ) : traces.length === 0 ? (
        <div className="text-zinc-500 text-center py-12">No traces found</div>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-zinc-800">
          <table className="w-full text-sm">
            <thead className="bg-zinc-900 text-zinc-400 text-xs uppercase tracking-wide">
              <tr>
                <th className="text-left px-4 py-3">Input</th>
                <th className="text-left px-4 py-3">Agent</th>
                <th className="text-center px-4 py-3">Status</th>
                <th className="text-right px-4 py-3">Tokens</th>
                <th className="text-right px-4 py-3">LLM</th>
                <th className="text-right px-4 py-3">Tools</th>
                <th className="text-right px-4 py-3">Duration</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800">
              {traces.map((t) => (
                <tr
                  key={t.trace_id}
                  className="hover:bg-zinc-900/50 transition-colors"
                >
                  <td className="px-4 py-3 max-w-xs">
                    <Link
                      href={`/traces/${t.trace_id}`}
                      className="text-zinc-200 hover:text-white block"
                    >
                      {t.input_query || t.trace_id.slice(0, 12)}
                    </Link>
                  </td>
                  <td className="px-4 py-3 text-zinc-400">
                    {t.agent_id || "-"}
                  </td>
                  <td className="px-4 py-3 text-center">
                    <StatusBadge status={t.status} />
                  </td>
                  <td className="px-4 py-3 text-right text-zinc-400">
                    {t.total_tokens.toLocaleString()}
                  </td>
                  <td className="px-4 py-3 text-right text-zinc-400">
                    {t.total_llm_calls}
                  </td>
                  <td className="px-4 py-3 text-right text-zinc-400">
                    {t.total_tool_calls}
                  </td>
                  <td className="px-4 py-3 text-right text-zinc-400">
                    {t.duration_ms ? `${Math.round(t.duration_ms)}ms` : "-"}
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
