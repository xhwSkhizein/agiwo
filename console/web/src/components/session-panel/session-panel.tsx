"use client";

import { useCallback, useEffect, useState } from "react";
import { Plus, RefreshCw } from "lucide-react";
import {
  listAgentSessions,
  createAgentSession,
  forkSession,
} from "@/lib/api";
import type { ChatSessionItem } from "@/lib/api";
import { SessionItem } from "./session-item";
import { ForkDialog } from "./fork-dialog";

interface SessionPanelProps {
  agentId: string;
  currentSessionId: string | null;
  onSwitch: (sessionId: string) => void;
  onCreated: (sessionId: string) => void;
  onForked: (sessionId: string) => void;
}

export function SessionPanel({
  agentId,
  currentSessionId,
  onSwitch,
  onCreated,
  onForked,
}: SessionPanelProps) {
  const [sessions, setSessions] = useState<ChatSessionItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [showForkDialog, setShowForkDialog] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await listAgentSessions(agentId);
      setSessions(result.items);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load sessions");
    } finally {
      setLoading(false);
    }
  }, [agentId]);

  useEffect(() => {
    void refresh();
  }, [refresh, currentSessionId]);

  const handleCreate = async () => {
    if (actionLoading) return;
    setActionLoading(true);
    setError(null);
    try {
      const result = await createAgentSession(agentId);
      onCreated(result.session_id);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create session");
    } finally {
      setActionLoading(false);
    }
  };

  const handleSwitch = async (targetSessionId: string) => {
    if (actionLoading) return;
    onSwitch(targetSessionId);
  };

  const handleFork = async (contextSummary: string) => {
    if (!currentSessionId || actionLoading) return;
    setActionLoading(true);
    setError(null);
    try {
      const result = await forkSession(currentSessionId, contextSummary);
      setShowForkDialog(false);
      onForked(result.session_id);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fork session");
    } finally {
      setActionLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-zinc-800 flex items-center justify-between">
        <h2 className="text-xs font-medium text-zinc-400 uppercase">Sessions</h2>
        <div className="flex items-center gap-1">
          <button
            onClick={refresh}
            disabled={loading}
            className="p-1.5 rounded hover:bg-zinc-800 text-zinc-500 hover:text-zinc-300 transition-colors disabled:opacity-30"
            title="Refresh"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
          </button>
          <button
            onClick={handleCreate}
            disabled={actionLoading}
            className="p-1.5 rounded hover:bg-zinc-800 text-zinc-500 hover:text-zinc-300 transition-colors disabled:opacity-30"
            title="New session"
          >
            <Plus className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-auto px-3 py-3 space-y-2">
        {showForkDialog && (
          <ForkDialog
            onConfirm={handleFork}
            onCancel={() => setShowForkDialog(false)}
          />
        )}

        {error && (
          <p className="rounded border border-red-900/50 bg-red-950/20 px-3 py-2 text-xs text-red-300">
            {error}
          </p>
        )}

        {sessions.length === 0 && !loading && (
          <p className="text-xs text-zinc-600 text-center py-4">
            No sessions yet
          </p>
        )}

        {sessions.map((session) => (
          <SessionItem
            key={session.session_id}
            session={session}
            isCurrent={session.session_id === currentSessionId}
            onSwitch={() => handleSwitch(session.session_id)}
            onFork={() => setShowForkDialog(true)}
          />
        ))}
      </div>
    </div>
  );
}
