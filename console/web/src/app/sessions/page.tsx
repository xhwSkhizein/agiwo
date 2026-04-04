"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { RefreshCw } from "lucide-react";
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
import { listSessions } from "@/lib/api";
import type { SessionSummary } from "@/lib/api";

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

  useEffect(() => {
    void loadSessions();
  }, [loadSessions]);

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Sessions</h1>
          <p className="text-sm text-zinc-400 mt-1">
            Session history and recent activity
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
            "disabled:opacity-50 disabled:cursor-not-allowed"
          )}
        >
          <RefreshCw className={cn("w-3.5 h-3.5", loading && "animate-spin")} />
          Refresh
        </button>
      </div>

      {error && <ErrorStateMessage>{error}</ErrorStateMessage>}

      {loading ? (
        <FullPageMessage loading>Loading sessions...</FullPageMessage>
      ) : sessions.length === 0 ? (
        <EmptyStateMessage>No sessions found</EmptyStateMessage>
      ) : (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 divide-y divide-zinc-800 overflow-hidden">
          {sessions.map((s) => (
            <Link
              key={s.session_id}
              href={`/sessions/${s.session_id}`}
              className={cn(
                "group block px-5 py-4 transition-colors duration-150",
                "hover:bg-zinc-800/50"
              )}
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1 min-w-0">
                  <div className="text-sm text-zinc-200">
                    <UserInputCompact
                      input={s.last_user_input}
                      maxLength={160}
                      showContextBadge={true}
                      showMetadata={true}
                      showAttachmentBadge={true}
                    />
                  </div>
                  {s.last_response && (
                    <p className="text-xs text-zinc-500 mt-2 truncate">
                      {s.last_response}
                    </p>
                  )}
                </div>
                <div className="text-right shrink-0 space-y-1.5">
                  <div className="flex items-center justify-end gap-2">
                    <PillBadge variant="default">{s.run_count} runs</PillBadge>
                    <MonoText truncate className="max-w-[120px]">{s.agent_id || "unknown"}</MonoText>
                  </div>
                  <p className="text-[11px] text-zinc-500 mt-1">
                    {formatRelativeTime(s.updated_at)}
                  </p>
                </div>
              </div>
            </Link>
          ))}
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
