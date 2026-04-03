"use client";

import { useCallback, useEffect, useState } from "react";
import { Plus, RefreshCw } from "lucide-react";
import {
  listAgentSessions,
  createAgentSession,
  forkSession,
} from "@/lib/api";
import type { ChatSessionItem } from "@/lib/api";
import { ErrorStateMessage } from "@/components/state-message";
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
      <div className="flex items-center justify-between border-b border-line px-4 py-3">
        <h2 className="text-xs font-medium uppercase tracking-[0.16em] text-ink-muted">
          Sessions
        </h2>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={refresh}
            disabled={loading}
            aria-label="Refresh sessions"
            className="ui-button ui-button-ghost ui-button-icon"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
          </button>
          <button
            type="button"
            onClick={handleCreate}
            disabled={actionLoading}
            aria-label="Create session"
            className="ui-button ui-button-ghost ui-button-icon"
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

        {error && <ErrorStateMessage className="text-xs">{error}</ErrorStateMessage>}

        {sessions.length === 0 && !loading && (
          <p className="py-4 text-center text-xs text-ink-faint">
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
