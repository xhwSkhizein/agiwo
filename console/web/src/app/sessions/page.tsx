"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  EmptyStateMessage,
  ErrorStateMessage,
  TextStateMessage,
} from "@/components/state-message";
import { PaginationControls } from "@/components/pagination-controls";
import { UserInputCompact } from "@/components/user-input-detail";
import { listSessions } from "@/lib/api";
import type { SessionSummary } from "@/lib/api";
import {
  formatTokenCount,
  formatUsd,
  normalizeRunMetricsSummary,
} from "@/lib/metrics";

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
          className="rounded-md border border-zinc-700 px-3 py-1.5 text-sm text-zinc-300 transition-colors hover:border-zinc-500 hover:text-white"
        >
          Refresh
        </button>
      </div>

      {error && <ErrorStateMessage>{error}</ErrorStateMessage>}

      {loading ? (
        <TextStateMessage>Loading...</TextStateMessage>
      ) : sessions.length === 0 ? (
        <EmptyStateMessage>No sessions found</EmptyStateMessage>
      ) : (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 divide-y divide-zinc-800">
          {sessions.map((s) => {
            const metrics = normalizeRunMetricsSummary(s.metrics);
            return (
              <Link
                key={s.session_id}
                href={`/sessions/${s.session_id}`}
                className="block px-5 py-4 hover:bg-zinc-800/50 transition-colors"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="text-sm">
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
                  <div className="text-right shrink-0">
                    <p className="text-xs text-zinc-400">{s.run_count} runs</p>
                    <p className="text-xs text-zinc-600 mt-0.5">
                      {s.agent_id || "unknown"}
                    </p>
                    <div className="mt-2 space-y-0.5">
                      <p className="text-[11px] text-zinc-400">
                        cost {formatUsd(metrics.token_cost)}
                      </p>
                      <p className="text-[11px] text-zinc-500">
                        in/out {formatTokenCount(metrics.input_tokens)}
                        {" / "}
                        {formatTokenCount(metrics.output_tokens)}
                      </p>
                      <p className="text-[11px] text-zinc-600">
                        cache r/c {formatTokenCount(metrics.cache_read_tokens)}
                        {" / "}
                        {formatTokenCount(metrics.cache_creation_tokens)}
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
