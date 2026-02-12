"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Plus, MessageSquare, Trash2, Network } from "lucide-react";
import { listAgents, deleteAgent } from "@/lib/api";
import type { AgentConfig } from "@/lib/api";

export default function AgentsPage() {
  const [agents, setAgents] = useState<AgentConfig[]>([]);
  const [loading, setLoading] = useState(true);

  const loadAgents = () => {
    setLoading(true);
    listAgents()
      .then(setAgents)
      .catch(() => setAgents([]))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadAgents();
  }, []);

  const handleDelete = async (id: string) => {
    if (!confirm("Delete this agent configuration?")) return;
    try {
      await deleteAgent(id);
      loadAgents();
    } catch {
      alert("Failed to delete agent");
    }
  };

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Agents</h1>
          <p className="text-sm text-zinc-400 mt-1">
            Create and manage agent configurations
          </p>
        </div>
        <Link
          href="/agents/new"
          className="flex items-center gap-2 px-4 py-2 rounded-md bg-white text-black text-sm font-medium hover:bg-zinc-200 transition-colors"
        >
          <Plus className="w-4 h-4" />
          New Agent
        </Link>
      </div>

      {loading ? (
        <div className="text-zinc-500">Loading...</div>
      ) : agents.length === 0 ? (
        <div className="text-center py-16 space-y-3">
          <p className="text-zinc-500">No agents configured yet</p>
          <Link
            href="/agents/new"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-md border border-zinc-700 text-sm hover:bg-zinc-800 transition-colors"
          >
            <Plus className="w-4 h-4" />
            Create your first agent
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {agents.map((agent) => (
            <div
              key={agent.id}
              className="rounded-lg border border-zinc-800 bg-zinc-900 p-5 flex flex-col"
            >
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="font-medium">{agent.name}</h3>
                  <p className="text-xs text-zinc-500 mt-0.5">
                    {agent.model_provider}/{agent.model_name}
                  </p>
                </div>
                <button
                  onClick={() => handleDelete(agent.id)}
                  className="p-1.5 rounded hover:bg-zinc-800 text-zinc-500 hover:text-red-400 transition-colors"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>

              {agent.description && (
                <p className="text-sm text-zinc-400 mt-2 line-clamp-2">
                  {agent.description}
                </p>
              )}

              {agent.tools && agent.tools.length > 0 && (
                <div className="flex flex-wrap gap-1.5 mt-2">
                  {agent.tools.map((t) => (
                    <span
                      key={t}
                      className={`px-2 py-0.5 rounded text-xs ${
                        t.startsWith("agent:")
                          ? "bg-blue-900/40 text-blue-400 border border-blue-800"
                          : "bg-zinc-800 text-zinc-500"
                      }`}
                    >
                      {t.startsWith("agent:") ? `🤖 ${t.slice(6)}` : t}
                    </span>
                  ))}
                </div>
              )}

              <div className="mt-auto pt-4 flex items-center gap-2">
                <Link
                  href={`/agents/${agent.id}/chat`}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-zinc-800 text-sm hover:bg-zinc-700 transition-colors"
                >
                  <MessageSquare className="w-3.5 h-3.5" />
                  Chat
                </Link>
                <Link
                  href={`/agents/${agent.id}/scheduler-chat`}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-purple-900/30 text-purple-400 text-sm hover:bg-purple-900/50 transition-colors"
                >
                  <Network className="w-3.5 h-3.5" />
                  Scheduler
                </Link>
                <Link
                  href={`/agents/${agent.id}/edit`}
                  className="px-3 py-1.5 rounded-md border border-zinc-700 text-sm hover:bg-zinc-800 transition-colors"
                >
                  Edit
                </Link>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
