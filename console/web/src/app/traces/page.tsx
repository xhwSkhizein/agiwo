"use client";

import { FormEvent, Suspense, useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import {
  EmptyStateMessage,
  ErrorStateMessage,
  TextStateMessage,
} from "@/components/state-message";
import { CopyButton } from "@/components/copy-button";
import { MonoText } from "@/components/mono-text";
import { PaginationControls } from "@/components/pagination-controls";
import { TraceStatusBadge } from "@/components/trace-status-badge";
import { listTraces } from "@/lib/api";
import type { TraceListItem } from "@/lib/api";
import { formatTokenCount, formatUsd } from "@/lib/metrics";
import { formatLocalDateTime, formatRoundedMs } from "@/lib/time";

const TRACE_STATUS_FILTERS = [
  { label: "All", value: "" },
  { label: "OK", value: "ok" },
  { label: "Error", value: "error" },
  { label: "Running", value: "running" },
  { label: "Unset", value: "unset" },
];

function TracesPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [traces, setTraces] = useState<TraceListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pageSize, setPageSize] = useState(Number(searchParams.get("limit") || "25"));
  const [offset, setOffset] = useState(Number(searchParams.get("offset") || "0"));
  const [statusInput, setStatusInput] = useState(searchParams.get("status") || "");
  const [agentInput, setAgentInput] = useState(searchParams.get("agent_id") || "");
  const [sessionInput, setSessionInput] = useState(searchParams.get("session_id") || "");
  const [filters, setFilters] = useState({
    status: searchParams.get("status") || "",
    agentId: searchParams.get("agent_id") || "",
    sessionId: searchParams.get("session_id") || "",
  });
  const [hasMore, setHasMore] = useState(false);

  const updateQuery = useCallback(
    (next: Record<string, string | number | null>) => {
      const query = new URLSearchParams(searchParams.toString());
      for (const [key, value] of Object.entries(next)) {
        if (value === null || value === "") {
          query.delete(key);
        } else {
          query.set(key, String(value));
        }
      }
      router.replace(`/traces?${query.toString()}`);
    },
    [router, searchParams],
  );

  const loadTraces = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const nextTraces = await listTraces({
        status: filters.status || undefined,
        agent_id: filters.agentId || undefined,
        session_id: filters.sessionId || undefined,
        limit: pageSize,
        offset,
      });
      setTraces(nextTraces.items);
      setHasMore(nextTraces.has_more);
    } catch (err) {
      setTraces([]);
      setHasMore(false);
      setError(err instanceof Error ? err.message : "Failed to load traces");
    } finally {
      setLoading(false);
    }
  }, [filters, offset, pageSize]);

  useEffect(() => {
    void loadTraces();
  }, [loadTraces]);

  function handleApplyFilters(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setOffset(0);
    const nextFilters = {
      status: statusInput,
      agentId: agentInput.trim(),
      sessionId: sessionInput.trim(),
    };
    setFilters(nextFilters);
    updateQuery({
      status: nextFilters.status || null,
      agent_id: nextFilters.agentId || null,
      session_id: nextFilters.sessionId || null,
      offset: 0,
      limit: pageSize,
    });
  }

  function handleResetFilters() {
    setStatusInput("");
    setAgentInput("");
    setSessionInput("");
    setOffset(0);
    setFilters({ status: "", agentId: "", sessionId: "" });
    updateQuery({
      status: null,
      agent_id: null,
      session_id: null,
      offset: 0,
      limit: pageSize,
    });
  }

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Traces</h1>
          <p className="text-sm text-zinc-400 mt-1">
            Agent execution traces with filterable cost and latency breakdowns
          </p>
        </div>
        <button
          type="button"
          onClick={() => {
            void loadTraces();
          }}
          className="rounded-md border border-zinc-700 px-3 py-1.5 text-sm text-zinc-300 transition-colors hover:border-zinc-500 hover:text-white"
        >
          Refresh
        </button>
      </div>

      <form
        onSubmit={handleApplyFilters}
        className="grid gap-3 rounded-lg border border-zinc-800 bg-zinc-900 p-4 sm:grid-cols-2 lg:grid-cols-4"
      >
        <label className="space-y-1">
          <span className="text-xs uppercase tracking-wide text-zinc-500">Status</span>
          <select
            value={statusInput}
            onChange={(event) => {
              setStatusInput(event.target.value);
            }}
            className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-200"
          >
            {TRACE_STATUS_FILTERS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>

        <label className="space-y-1">
          <span className="text-xs uppercase tracking-wide text-zinc-500">Agent ID</span>
          <input
            value={agentInput}
            onChange={(event) => {
              setAgentInput(event.target.value);
            }}
            placeholder="agent-123"
            className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-200 placeholder:text-zinc-600"
          />
        </label>

        <label className="space-y-1">
          <span className="text-xs uppercase tracking-wide text-zinc-500">Session ID</span>
          <input
            value={sessionInput}
            onChange={(event) => {
              setSessionInput(event.target.value);
            }}
            placeholder="session-123"
            className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-200 placeholder:text-zinc-600"
          />
        </label>

        <div className="flex items-end gap-2">
          <button
            type="submit"
            className="rounded-md border border-zinc-700 px-3 py-2 text-sm text-zinc-300 transition-colors hover:border-zinc-500 hover:text-white"
          >
            Apply
          </button>
          <button
            type="button"
            onClick={handleResetFilters}
            className="rounded-md border border-zinc-800 px-3 py-2 text-sm text-zinc-500 transition-colors hover:border-zinc-600 hover:text-zinc-300"
          >
            Clear
          </button>
        </div>
      </form>

      {error && <ErrorStateMessage>{error}</ErrorStateMessage>}

      {loading ? (
        <TextStateMessage>Loading...</TextStateMessage>
      ) : traces.length === 0 ? (
        <EmptyStateMessage>No traces found</EmptyStateMessage>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-zinc-800">
          <table className="w-full text-sm">
            <thead className="bg-zinc-900 text-zinc-400 text-xs uppercase tracking-wide">
              <tr>
                <th className="text-left px-4 py-3">Trace</th>
                <th className="text-left px-4 py-3">Input / Output</th>
                <th className="text-left px-4 py-3">Session</th>
                <th className="text-left px-4 py-3">Agent</th>
                <th className="text-center px-4 py-3">Status</th>
                <th className="text-right px-4 py-3">Cost</th>
                <th className="text-right px-4 py-3">Tokens</th>
                <th className="text-right px-4 py-3">Calls</th>
                <th className="text-right px-4 py-3">Duration</th>
                <th className="text-right px-4 py-3">Started</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800">
              {traces.map((t) => (
                <tr
                  key={t.trace_id}
                  className="hover:bg-zinc-900/50 transition-colors"
                >
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <Link
                        href={`/traces/${t.trace_id}`}
                        className="font-mono text-xs text-zinc-200 transition-colors hover:text-white"
                      >
                        {t.trace_id.slice(0, 12)}
                      </Link>
                      <CopyButton
                        value={t.trace_id}
                        label="Copy"
                        className="h-5 px-1.5"
                      />
                    </div>
                  </td>
                  <td className="px-4 py-3 min-w-[20rem] max-w-md">
                    <Link
                      href={`/traces/${t.trace_id}`}
                      className="block text-zinc-200 transition-colors hover:text-white"
                    >
                      <span className="line-clamp-2">
                        {t.input_query || "Trace detail"}
                      </span>
                      {t.final_output ? (
                        <span className="mt-1 block truncate text-xs text-zinc-500">
                          {t.final_output}
                        </span>
                      ) : null}
                    </Link>
                  </td>
                  <td className="px-4 py-3 text-zinc-400">
                    {t.session_id ? (
                      <Link
                        href={`/sessions/${t.session_id}`}
                        className="transition-colors hover:text-zinc-200"
                      >
                        <MonoText truncate className="max-w-[9rem]">
                          {t.session_id}
                        </MonoText>
                      </Link>
                    ) : (
                      "-"
                    )}
                  </td>
                  <td className="px-4 py-3 text-zinc-400">
                    {t.agent_id ? (
                      <MonoText truncate className="max-w-[9rem]">
                        {t.agent_id}
                      </MonoText>
                    ) : (
                      "-"
                    )}
                  </td>
                  <td className="px-4 py-3 text-center">
                    <TraceStatusBadge status={t.status} />
                  </td>
                  <td className="px-4 py-3 text-right text-zinc-200">
                    {formatUsd(t.total_token_cost)}
                  </td>
                  <td className="px-4 py-3 text-right text-zinc-400">
                    <div>{formatTokenCount(t.total_tokens)}</div>
                    <div className="text-[11px] text-zinc-500">
                      {formatTokenCount(t.total_input_tokens)} in / {formatTokenCount(t.total_output_tokens)} out
                    </div>
                  </td>
                  <td className="px-4 py-3 text-right text-zinc-400">
                    <div>{t.total_llm_calls} llm</div>
                    <div className="text-[11px] text-zinc-500">{t.total_tool_calls} tools</div>
                  </td>
                  <td className="px-4 py-3 text-right text-zinc-400">
                    {formatRoundedMs(t.duration_ms)}
                  </td>
                  <td className="px-4 py-3 text-right text-xs text-zinc-500">
                    {formatLocalDateTime(t.start_time)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <PaginationControls
        offset={offset}
        pageSize={pageSize}
        itemCount={traces.length}
        hasMore={hasMore}
        itemLabel="traces"
        disabled={loading}
        onPageSizeChange={(nextPageSize) => {
          setPageSize(nextPageSize);
          setOffset(0);
          updateQuery({ limit: nextPageSize, offset: 0 });
        }}
        onPrevious={() => {
          setOffset((current) => {
            const nextOffset = Math.max(0, current - pageSize);
            updateQuery({ offset: nextOffset, limit: pageSize });
            return nextOffset;
          });
        }}
        onNext={() => {
          setOffset((current) => {
            const nextOffset = current + pageSize;
            updateQuery({ offset: nextOffset, limit: pageSize });
            return nextOffset;
          });
        }}
      />
    </div>
  );
}

export default function TracesPage() {
  return (
    <Suspense fallback={<TextStateMessage>Loading...</TextStateMessage>}>
      <TracesPageContent />
    </Suspense>
  );
}
