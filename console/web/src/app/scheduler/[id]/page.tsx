"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { getAgentState, getAgentStateChildren } from "@/lib/api";
import type { AgentStateDetail, AgentStateListItem } from "@/lib/api";

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

function InfoRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-4 py-2.5 border-b border-zinc-800/50 last:border-0">
      <span className="text-xs text-zinc-500 w-36 shrink-0 pt-0.5">{label}</span>
      <div className="text-sm min-w-0">{children}</div>
    </div>
  );
}

function WakeConditionCard({ wc }: { wc: AgentStateDetail["wake_condition"] }) {
  if (!wc) return null;
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4 space-y-2">
      <h3 className="text-sm font-medium">Wake Condition</h3>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
        <div>
          <span className="text-zinc-500">Type: </span>
          <span className="text-zinc-200">{wc.type}</span>
        </div>
        {wc.type === "children_complete" && (
          <>
            <div>
              <span className="text-zinc-500">Progress: </span>
              <span className="text-zinc-200">
                {wc.completed_children} / {wc.total_children}
              </span>
            </div>
            <div>
              <span className="text-zinc-500">Remaining: </span>
              <span className="text-zinc-200">
                {wc.total_children - wc.completed_children}
              </span>
            </div>
          </>
        )}
        {(wc.type === "delay" || wc.type === "interval") && (
          <>
            {wc.time_value != null && wc.time_unit && (
              <div>
                <span className="text-zinc-500">Duration: </span>
                <span className="text-zinc-200">
                  {wc.time_value} {wc.time_unit}
                </span>
              </div>
            )}
            {wc.wakeup_at && (
              <div>
                <span className="text-zinc-500">Wakeup At: </span>
                <span className="text-zinc-200">
                  {new Date(wc.wakeup_at).toLocaleString()}
                </span>
              </div>
            )}
          </>
        )}
      </div>
      {wc.type === "children_complete" && wc.total_children > 0 && (
        <div className="mt-2">
          <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-green-500 rounded-full transition-all"
              style={{
                width: `${Math.min(
                  (wc.completed_children / wc.total_children) * 100,
                  100
                )}%`,
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function ChildrenTable({ children }: { children: AgentStateListItem[] }) {
  if (children.length === 0) return null;
  return (
    <div className="rounded-lg border border-zinc-800 overflow-hidden">
      <div className="px-4 py-3 bg-zinc-900 border-b border-zinc-800">
        <h3 className="text-sm font-medium">
          Child Agents ({children.length})
        </h3>
      </div>
      <table className="w-full text-sm">
        <thead className="bg-zinc-900/50 text-zinc-500 text-xs uppercase tracking-wide">
          <tr>
            <th className="text-left px-4 py-2">Agent</th>
            <th className="text-left px-4 py-2">Task</th>
            <th className="text-center px-4 py-2">Status</th>
            <th className="text-left px-4 py-2">Result</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-800">
          {children.map((c) => (
            <tr key={c.id} className="hover:bg-zinc-900/50 transition-colors">
              <td className="px-4 py-2.5">
                <Link
                  href={`/scheduler/${c.id}`}
                  className="text-zinc-200 hover:text-white font-mono text-xs"
                >
                  {c.agent_id}
                </Link>
              </td>
              <td className="px-4 py-2.5 max-w-xs">
                <span className="truncate block text-zinc-300 text-xs">
                  {c.task}
                </span>
              </td>
              <td className="px-4 py-2.5 text-center">
                <StatusBadge status={c.status} />
              </td>
              <td className="px-4 py-2.5 text-xs text-zinc-400 max-w-xs truncate">
                {c.result_summary || "-"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function SchedulerDetailPage() {
  const params = useParams();
  const stateId = params.id as string;
  const [state, setState] = useState<AgentStateDetail | null>(null);
  const [children, setChildren] = useState<AgentStateListItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      getAgentState(stateId).catch(() => null),
      getAgentStateChildren(stateId).catch(() => []),
    ]).then(([s, c]) => {
      setState(s);
      setChildren(c);
      setLoading(false);
    });
  }, [stateId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-zinc-500">Loading...</div>
      </div>
    );
  }

  if (!state) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-zinc-500">Agent state not found</div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <div className="flex items-center gap-3">
        <Link
          href="/scheduler"
          className="p-1.5 rounded hover:bg-zinc-800 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
        </Link>
        <div>
          <h1 className="text-xl font-semibold">Agent State</h1>
          <p className="text-xs text-zinc-500 font-mono mt-0.5">{state.id}</p>
        </div>
        <div className="ml-auto">
          <StatusBadge status={state.status} />
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <p className="text-xs text-zinc-500">Status</p>
          <div className="mt-1">
            <StatusBadge status={state.status} />
          </div>
        </div>
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <p className="text-xs text-zinc-500">Agent ID</p>
          <p className="text-sm font-mono mt-1 truncate">{state.agent_id}</p>
        </div>
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <p className="text-xs text-zinc-500">Session</p>
          <p className="text-sm font-mono mt-1 truncate">{state.session_id}</p>
        </div>
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <p className="text-xs text-zinc-500">Children</p>
          <p className="text-lg font-medium mt-1">{children.length}</p>
        </div>
      </div>

      <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
        <h3 className="text-sm font-medium mb-2">Details</h3>
        <InfoRow label="Task">
          <p className="whitespace-pre-wrap">{state.task}</p>
        </InfoRow>
        <InfoRow label="Parent Agent">
          <span className="font-mono text-xs">{state.parent_agent_id}</span>
        </InfoRow>
        {state.parent_state_id && (
          <InfoRow label="Parent State">
            <Link
              href={`/scheduler/${state.parent_state_id}`}
              className="font-mono text-xs text-blue-400 hover:text-blue-300"
            >
              {state.parent_state_id}
            </Link>
          </InfoRow>
        )}
        <InfoRow label="Signal Propagated">
          <span>{state.signal_propagated ? "Yes" : "No"}</span>
        </InfoRow>
        <InfoRow label="Created">
          <span className="text-zinc-400">
            {state.created_at ? new Date(state.created_at).toLocaleString() : "-"}
          </span>
        </InfoRow>
        <InfoRow label="Updated">
          <span className="text-zinc-400">
            {state.updated_at ? new Date(state.updated_at).toLocaleString() : "-"}
          </span>
        </InfoRow>
      </div>

      {state.result_summary && (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <h3 className="text-sm font-medium mb-2">Result Summary</h3>
          <p className="text-sm whitespace-pre-wrap text-zinc-300">
            {state.result_summary}
          </p>
        </div>
      )}

      <WakeConditionCard wc={state.wake_condition} />

      {Object.keys(state.config_overrides).length > 0 && (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <h3 className="text-sm font-medium mb-2">Config Overrides</h3>
          <pre className="text-xs bg-zinc-800/50 rounded px-3 py-2 overflow-auto max-h-48 font-mono">
            {JSON.stringify(state.config_overrides, null, 2)}
          </pre>
        </div>
      )}

      <ChildrenTable children={children} />
    </div>
  );
}
