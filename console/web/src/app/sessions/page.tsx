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
import {
  formatTokenCount,
  formatUsd,
  normalizeRunMetricsSummary,
} from "@/lib/metrics";

/**
 * Render the Sessions page which lists session summaries with cost and token rollups, supports pagination, refresh, and error/loading states.
 *
 * Loads session data on mount and whenever pagination changes; displays a refresh control, per-session metadata (run count, agent id, cost, token counts), and pagination controls.
 *
 * @returns The page's JSX element.
 */
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
            Session history with cost and token rollups
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
          {sessions.map((s) => {
            const metrics = normalizeRunMetricsSummary(s.metrics);
            return (
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
                    <div className="mt-2 space-y-0.5 text-[11px]">
                      <p className="text-zinc-400">
                        cost <span className="text-zinc-300">{formatUsd(metrics.token_cost)}</span>
                      </p>
                      <p className="text-zinc-500">
                        in/out <span className="text-zinc-400">{formatTokenCount(metrics.input_tokens)}</span>
                        <span className="text-zinc-700 mx-1">/</span>
                        <span className="text-zinc-400">{formatTokenCount(metrics.output_tokens)}</span>
                      </p>
                      <p className="text-zinc-600">
                        cache r/c <span className="text-zinc-500">{formatTokenCount(metrics.cache_read_tokens)}</span>
                        <span className="text-zinc-700 mx-1">/</span>
                        <span className="text-zinc-500">{formatTokenCount(metrics.cache_creation_tokens)}</span>
                      </p>
                    </div>
                  </div>
                </div>
              </Link>
            );
          })}
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
