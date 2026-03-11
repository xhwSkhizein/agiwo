"use client";

import { useCallback, useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { BackHeader } from "@/components/back-header";
import { MetricCard } from "@/components/metric-card";
import { MonoLink, MonoText } from "@/components/mono-text";
import { PillBadge } from "@/components/pill-badge";
import { SectionCard } from "@/components/section-card";
import { SchedulerStatusBadge } from "@/components/scheduler-status-badge";
import { FullPageMessage } from "@/components/state-message";
import { TokenSummaryCards } from "@/components/token-summary-cards";
import {
  getAgentState,
  getAgentStateChildren,
  steerAgent,
  cancelAgent,
  resumeAgent,
} from "@/lib/api";
import { UserInputDetail, UserInputCompact } from "@/components/user-input-detail";
import type { AgentStateDetail, AgentStateListItem } from "@/lib/api";
import {
  formatDurationMs,
  formatTokenCount,
  formatUsd,
  normalizeRunMetricsSummary,
} from "@/lib/metrics";
import { formatLocalDateTime } from "@/lib/time";
import {
  formatWakeConditionTimer,
  getWaitsetProgress,
} from "@/lib/wake-condition";

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

  const waitsetProgress = getWaitsetProgress(wc);
  const timerLabel =
    wc.type === "timer" || wc.type === "periodic"
      ? formatWakeConditionTimer(wc)
      : null;

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4 space-y-2">
      <h3 className="text-sm font-medium">Wake Condition</h3>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
        <div>
          <span className="text-zinc-500">Type: </span>
          <span className="text-zinc-200">{wc.type}</span>
        </div>
        {wc.type === "waitset" && (
          <>
            <div>
              <span className="text-zinc-500">Progress: </span>
              <span className="text-zinc-200">
                {waitsetProgress.completed} / {waitsetProgress.total}
              </span>
            </div>
            <div>
              <span className="text-zinc-500">Mode: </span>
              <span className="text-zinc-200">{wc.wait_mode}</span>
            </div>
          </>
        )}
        {(wc.type === "timer" || wc.type === "periodic") && (
          <>
            {timerLabel && (
              <div>
                <span className="text-zinc-500">
                  {wc.type === "periodic" ? "Interval: " : "Duration: "}
                </span>
                <span className="text-zinc-200">{timerLabel}</span>
              </div>
            )}
            {wc.wakeup_at && (
              <div>
                <span className="text-zinc-500">Wakeup At: </span>
                <span className="text-zinc-200">
                  {formatLocalDateTime(wc.wakeup_at)}
                </span>
              </div>
            )}
          </>
        )}
        {wc.type === "task_submitted" && wc.submitted_task !== null && wc.submitted_task !== undefined && (
          <div className="col-span-2 sm:col-span-3">
            <span className="text-zinc-500">Submitted Task: </span>
            <div className="mt-1 p-2 rounded bg-zinc-800/50">
              <UserInputCompact input={wc.submitted_task} maxLength={100} />
            </div>
          </div>
        )}
        {wc.timeout_at && (
          <div>
            <span className="text-zinc-500">Timeout: </span>
            <span className="text-zinc-200">
              {formatLocalDateTime(wc.timeout_at)}
            </span>
          </div>
        )}
      </div>
      {wc.type === "waitset" && wc.wait_for.length > 0 && (
        <div className="mt-2">
          <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-green-500 rounded-full transition-all"
              style={{ width: `${waitsetProgress.percent}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function ChildrenTable({ items }: { items: AgentStateListItem[] }) {
  if (items.length === 0) return null;
  return (
    <div className="rounded-lg border border-zinc-800 overflow-hidden">
      <div className="px-4 py-3 bg-zinc-900 border-b border-zinc-800">
        <h3 className="text-sm font-medium">
          Child Agents ({items.length})
        </h3>
      </div>
      <table className="w-full text-sm">
        <thead className="bg-zinc-900/50 text-zinc-500 text-xs uppercase tracking-wide">
          <tr>
            <th className="text-left px-4 py-2">Agent</th>
            <th className="text-left px-4 py-2">Task</th>
            <th className="text-center px-4 py-2">Status</th>
            <th className="text-right px-4 py-2">Cost</th>
            <th className="text-right px-4 py-2">Tokens</th>
            <th className="text-left px-4 py-2">Result</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-800">
          {items.map((c) => {
            const metrics = normalizeRunMetricsSummary(c.metrics);
            return (
              <tr key={c.id} className="hover:bg-zinc-900/50 transition-colors">
                <td className="px-4 py-2.5">
                  <MonoLink href={`/scheduler/${c.id}`}>
                    {c.id}
                  </MonoLink>
                </td>
                <td className="px-4 py-2.5 max-w-xs">
                  <UserInputCompact input={c.task} maxLength={50} />
                </td>
                <td className="px-4 py-2.5 text-center">
                  <SchedulerStatusBadge status={c.status} />
                </td>
                <td className="px-4 py-2.5 text-right text-xs text-zinc-200">
                  {formatUsd(metrics.token_cost)}
                </td>
                <td className="px-4 py-2.5 text-right text-xs text-zinc-500">
                  {formatTokenCount(metrics.input_tokens)} / {formatTokenCount(metrics.output_tokens)} / {formatTokenCount(metrics.total_tokens)}
                </td>
                <td className="px-4 py-2.5 text-xs text-zinc-400 max-w-xs truncate">
                  {c.result_summary || "-"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ControlPanel({ state, onAction }: { state: AgentStateDetail; onAction: () => void }) {
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);

  const isRoot = state.parent_id === null;
  const isActive = state.status === "running" || state.status === "sleeping";
  const canResume = isRoot && state.is_persistent && ["sleeping", "completed", "failed"].includes(state.status);

  const handle = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    try {
      await fn();
      onAction();
    } catch { /* toast or ignore */ }
    setBusy(false);
  };

  if (!isRoot) return null;

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4 space-y-3">
      <h3 className="text-sm font-medium">Control</h3>
      <div className="flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Message..."
          className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-sm placeholder-zinc-500 focus:outline-none focus:border-zinc-500"
          onKeyDown={(e) => {
            if (e.key === "Enter" && input.trim()) {
              if (canResume) handle(() => resumeAgent(state.id, input.trim()).then(() => setInput("")));
              else if (isActive) handle(() => steerAgent(state.id, input.trim(), true).then(() => setInput("")));
            }
          }}
        />
        {isActive && (
          <button
            disabled={busy || !input.trim()}
            onClick={() => handle(() => steerAgent(state.id, input.trim(), true).then(() => setInput("")))}
            className="px-3 py-1.5 text-xs rounded bg-blue-900/50 text-blue-400 hover:bg-blue-800/50 disabled:opacity-40 transition-colors"
          >
            Steer
          </button>
        )}
        {canResume && (
          <button
            disabled={busy || !input.trim()}
            onClick={() => handle(() => resumeAgent(state.id, input.trim()).then(() => setInput("")))}
            className="px-3 py-1.5 text-xs rounded bg-green-900/50 text-green-400 hover:bg-green-800/50 disabled:opacity-40 transition-colors"
          >
            Resume
          </button>
        )}
        {isActive && (
          <button
            disabled={busy}
            onClick={() => handle(() => cancelAgent(state.id))}
            className="px-3 py-1.5 text-xs rounded bg-red-900/50 text-red-400 hover:bg-red-800/50 disabled:opacity-40 transition-colors"
          >
            Cancel
          </button>
        )}
      </div>
    </div>
  );
}

export default function SchedulerDetailPage() {
  const params = useParams();
  const stateId = params.id as string;
  const [state, setState] = useState<AgentStateDetail | null>(null);
  const [children, setChildren] = useState<AgentStateListItem[]>([]);
  const [loading, setLoading] = useState(true);

  const loadStateData = useCallback(async (showLoading = true) => {
    if (showLoading) {
      setLoading(true);
    }

    try {
      const [nextState, nextChildren] = await Promise.all([
        getAgentState(stateId).catch(() => null),
        getAgentStateChildren(stateId).catch(() => []),
      ]);
      setState(nextState);
      setChildren(nextChildren);
    } finally {
      if (showLoading) {
        setLoading(false);
      }
    }
  }, [stateId]);

  const refresh = useCallback(() => {
    void loadStateData(false);
  }, [loadStateData]);

  useEffect(() => {
    void loadStateData();
  }, [loadStateData]);

  if (loading) {
    return <FullPageMessage>Loading...</FullPageMessage>;
  }

  if (!state) {
    return <FullPageMessage>Agent state not found</FullPageMessage>;
  }

  const stateMetrics = normalizeRunMetricsSummary(state.metrics);

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <BackHeader
        href="/scheduler"
        title="Agent State"
        subtitle={state.id}
        rightContent={
          <>
            {state.is_persistent && (
              <PillBadge className="text-xs px-1.5 py-0.5 rounded bg-indigo-900/50 text-indigo-400">
                persistent
              </PillBadge>
            )}
            <SchedulerStatusBadge status={state.status} />
          </>
        }
      />

      <TokenSummaryCards
        cost={stateMetrics.token_cost}
        inputTokens={stateMetrics.input_tokens}
        outputTokens={stateMetrics.output_tokens}
        totalTokens={stateMetrics.total_tokens}
        cacheReadTokens={stateMetrics.cache_read_tokens}
        cacheCreationTokens={stateMetrics.cache_creation_tokens}
        extraCards={
          <>
            <MetricCard
              label="Status"
              value={<SchedulerStatusBadge status={state.status} />}
            />
            <MetricCard
              label="Session"
              valueClassName="text-sm font-mono truncate"
              value={state.session_id}
            />
            <MetricCard
              label="Children"
              valueClassName="text-lg font-medium"
              value={children.length}
            />
          </>
        }
      />

      <SectionCard
        title="Details"
        className="p-4"
        headerClassName="mb-2"
        titleClassName="text-sm font-medium"
      >
        <InfoRow label="Task">
          <UserInputDetail input={state.task} showContext={true} />
        </InfoRow>
        {state.parent_id && (
          <InfoRow label="Parent">
            <MonoLink
              href={`/scheduler/${state.parent_id}`}
              className="font-mono text-xs text-blue-400 hover:text-blue-300"
            >
              {state.parent_id}
            </MonoLink>
          </InfoRow>
        )}
        {state.agent_config_id && (
          <InfoRow label="Config ID">
            <MonoText>{state.agent_config_id}</MonoText>
          </InfoRow>
        )}
        <InfoRow label="Signal Propagated">
          <span>{state.signal_propagated ? "Yes" : "No"}</span>
        </InfoRow>
        <InfoRow label="Run Metrics">
          <span className="text-zinc-400">
            runs {stateMetrics.run_count} | steps {stateMetrics.step_count} | tools {stateMetrics.tool_calls_count} | total {formatTokenCount(stateMetrics.total_tokens)} | duration {formatDurationMs(stateMetrics.duration_ms)}
          </span>
        </InfoRow>
        <InfoRow label="Created">
          <span className="text-zinc-400">
            {formatLocalDateTime(state.created_at)}
          </span>
        </InfoRow>
        <InfoRow label="Updated">
          <span className="text-zinc-400">
            {formatLocalDateTime(state.updated_at)}
          </span>
        </InfoRow>
      </SectionCard>

      {state.result_summary && (
        <SectionCard
          title="Result Summary"
          className="p-4"
          headerClassName="mb-2"
          titleClassName="text-sm font-medium"
        >
          <p className="text-sm whitespace-pre-wrap text-zinc-300">
            {state.result_summary}
          </p>
        </SectionCard>
      )}

      <ControlPanel state={state} onAction={refresh} />

      <WakeConditionCard wc={state.wake_condition} />

      {Object.keys(state.config_overrides).length > 0 && (
        <SectionCard
          title="Config Overrides"
          className="p-4"
          headerClassName="mb-2"
          titleClassName="text-sm font-medium"
        >
          <pre className="text-xs bg-zinc-800/50 rounded px-3 py-2 overflow-auto max-h-48 font-mono">
            {JSON.stringify(state.config_overrides, null, 2)}
          </pre>
        </SectionCard>
      )}

      <ChildrenTable items={children} />
    </div>
  );
}
