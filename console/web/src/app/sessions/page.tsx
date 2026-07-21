"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { RefreshCw, Trash2 } from "lucide-react";
import {
  EmptyStateMessage,
  ErrorStateMessage,
  FullPageMessage,
} from "@/components/state-message";
import { PaginationControls } from "@/components/pagination-controls";
import { UserInputCompact } from "@/components/user-input-detail";
import { PillBadge } from "@/components/pill-badge";
import { MonoText } from "@/components/mono-text";
import { cn } from "@/lib/utils";
import { archiveSession, listSessions, type SessionSummary } from "@/lib/api";

function formatRelativeTime(dateStr: string | null): string {
  if (!dateStr) return "";
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

function sessionStatusVariant(
  status: string,
): "running" | "pending" | "success" | "error" | "default" {
  const normalized = status.toLowerCase();
  if (normalized === "running") {
    return "running";
  }
  if (normalized === "completed" || normalized === "idle") {
    return "success";
  }
  if (normalized === "failed" || normalized === "cancelled") {
    return "error";
  }
  return "pending";
}

export default function SessionsPage() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pageSize, setPageSize] = useState(25);
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [total, setTotal] = useState<number | null>(null);

  const loadSessions = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const nextSessions = await listSessions(pageSize, offset);
      setSessions(nextSessions.items);
      setHasMore(nextSessions.has_more);
      setTotal(nextSessions.total);
    } catch (err) {
      setSessions([]);
      setHasMore(false);
      setTotal(null);
      setError(err instanceof Error ? err.message : "Failed to load sessions");
    } finally {
      setLoading(false);
    }
  }, [offset, pageSize]);

  const handleDelete = async (sessionId: string, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!confirm("Archive this session? You can restore it later.")) return;
    try {
      await archiveSession(sessionId);
      setSessions((prev) => prev.filter((s) => s.session_id !== sessionId));
      if (total !== null) {
        setTotal((t) => (t !== null ? t - 1 : null));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to archive session");
    }
  };

  useEffect(() => {
    void loadSessions();
  }, [loadSessions]);

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Sessions</h1>
          <p className="mt-1 text-sm text-zinc-400">
            Recent session activity and run status
          </p>
        </div>
        <button
          type="button"
          onClick={() => {
            void loadSessions();
          }}
          disabled={loading}
          className={cn(
            "inline-flex items-center gap-2 rounded-md border px-3 py-1.5 text-sm",
            "transition-all duration-150",
            "border-zinc-700 text-zinc-300 hover:border-zinc-500 hover:text-white",
            "disabled:cursor-not-allowed disabled:opacity-50",
          )}
        >
          <RefreshCw className={cn("h-3.5 w-3.5", loading && "animate-spin")} />
          Refresh
        </button>
      </div>

      {error && <ErrorStateMessage>{error}</ErrorStateMessage>}

      {loading ? (
        <FullPageMessage loading>Loading sessions...</FullPageMessage>
      ) : sessions.length === 0 ? (
        <EmptyStateMessage>No sessions found</EmptyStateMessage>
      ) : (
        <div className="overflow-hidden rounded-lg border border-zinc-800 bg-zinc-900/50">
          <div className="hidden grid-cols-[minmax(0,1.4fr)_8rem_minmax(0,1fr)_7rem_5.5rem] gap-3 border-b border-zinc-800 px-5 py-2 text-[10px] font-semibold uppercase tracking-wide text-zinc-500 md:grid">
            <span>Work item</span>
            <span>Status</span>
            <span>Last response</span>
            <span>Size</span>
            <span>Updated</span>
          </div>
          <div className="divide-y divide-zinc-800">
            {sessions.map((s) => {
              const status = s.root_state_status || "idle";
              const focus = s.last_response || "No response yet";
              return (
                <div
                  key={s.session_id}
                  className="grid gap-3 px-5 py-4 transition-colors hover:bg-zinc-800/50 md:grid-cols-[minmax(0,1.4fr)_8rem_minmax(0,1fr)_7rem_5.5rem] md:items-start"
                >
                  <Link href={`/sessions/${s.session_id}`} className="min-w-0">
                    <div className="text-sm text-zinc-200">
                      <UserInputCompact
                        input={s.last_user_input}
                        maxLength={140}
                        showContextBadge
                        showMetadata
                        showAttachmentBadge
                      />
                    </div>
                    <MonoText className="mt-1.5 block text-[11px] text-zinc-500">
                      {s.session_id.slice(0, 12)} · {s.agent_id || "unknown"}
                    </MonoText>
                  </Link>
                  <div>
                    <PillBadge variant={sessionStatusVariant(status)} dot>
                      {status}
                    </PillBadge>
                  </div>
                  <Link
                    href={`/sessions/${s.session_id}`}
                    className="min-w-0 text-xs leading-5 text-zinc-400 line-clamp-2"
                  >
                    {focus}
                  </Link>
                  <div className="text-xs text-zinc-500">
                    {s.run_count} runs · {s.step_count} steps
                  </div>
                  <div className="flex items-start justify-between gap-2 md:flex-col md:items-end">
                    <p className="text-[11px] text-zinc-500">
                      {formatRelativeTime(s.updated_at)}
                    </p>
                    <button
                      type="button"
                      onClick={(e) => handleDelete(s.session_id, e)}
                      className={cn(
                        "rounded-md p-1.5 text-zinc-500 transition-colors duration-150",
                        "hover:bg-red-900/20 hover:text-red-400",
                      )}
                      aria-label={`Archive session ${s.session_id.slice(0, 8)}`}
                      title="Archive"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <PaginationControls
        offset={offset}
        pageSize={pageSize}
        itemCount={sessions.length}
        totalCount={total}
        hasMore={hasMore}
        itemLabel="sessions"
        disabled={loading}
        onPageSizeChange={(nextPageSize) => {
          setPageSize(nextPageSize);
          setOffset(0);
        }}
        onPrevious={() => {
          setOffset((current) => Math.max(0, current - pageSize));
        }}
        onNext={() => {
          setOffset((current) => current + pageSize);
        }}
      />
    </div>
  );
}
